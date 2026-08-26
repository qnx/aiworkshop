/**
 * Copyright (c) 2026, BlackBerry Limited. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "handtrack_gui.hpp"

#include <gdk/gdkkeysyms.h>  // GDK_KEY_* are not pulled in by gtk/gtk.h

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <opencv2/imgproc.hpp>

namespace {

// How long a hand or visitor should live before getting removed on no
// detection.
constexpr auto SOURCE_TTL = std::chrono::seconds(3);

const int CONN[][2] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
    {7, 8},   {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14},
    {14, 15}, {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17}};

constexpr double FILL_ALPHA = 0.28;  // enough tint to group the hand, light
                                     // enough to see it through
constexpr double CORNER_R = 10.0;
constexpr double DOT_R = 7.0;    // visitor dot radius
constexpr double JOINT_R = 4.0;  // landmark dot radius
constexpr double LINE_W = 2.0;   // skeleton and box edge
constexpr double TAG_PX = 15.0;  // text size
constexpr double TAG_PAD = 4.0;  // plate padding around the text
constexpr double TAG_GAP = 5.0;  // between a visitor dot and its tag

// Colours are carried as OpenCV BGR scalars because that is what the messages
// decode into; cairo wants RGB in 0..1.
void set_colour(cairo_t* cr, const cv::Scalar& bgr, double alpha = 1.0) {
  cairo_set_source_rgba(cr, bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0,
                        alpha);
}

// Rounded rect as a cairo path: four corner arcs joined into one subpath.
void rounded_path(cairo_t* cr, double x, double y, double w, double h,
                  double r) {
  r = std::min(r, std::min(w, h) / 2);
  cairo_new_sub_path(cr);
  cairo_arc(cr, x + w - r, y + r, r, -G_PI / 2, 0);
  cairo_arc(cr, x + w - r, y + h - r, r, 0, G_PI / 2);
  cairo_arc(cr, x + r, y + h - r, r, G_PI / 2, G_PI);
  cairo_arc(cr, x + r, y + r, r, G_PI, 3 * G_PI / 2);
  cairo_close_path(cr);
}

// Plate height for n lines. Known before the text is measured, so a caller can
// centre a tag on something before it knows how wide it will be.
double tag_height(cairo_t* cr, size_t lines) {
  cairo_font_extents_t fe;
  cairo_font_extents(cr, &fe);
  return fe.height * lines + 2 * TAG_PAD;
}

// One tag: a rounded plate in `colour` carrying white lines of text, anchored
// at its top-left. The plate stays opaque -- text over the blended box fill is
// what makes a label unreadable exactly when the hand is over something busy.
//
// `bounds` non-null drops the tag whole when it would not fit inside the
// widget: half a name at the window edge reads as a different name, and a
// clipped score reads as a different number.
bool draw_tag(cairo_t* cr, double x, double y,
              const std::vector<std::string>& lines, const cv::Scalar& colour,
              const cv::Size* bounds) {
  cairo_font_extents_t fe;
  cairo_font_extents(cr, &fe);
  double text_w = 0;
  for (const auto& l : lines) {
    cairo_text_extents_t te;
    cairo_text_extents(cr, l.c_str(), &te);
    text_w = std::max(text_w, te.width);
  }
  const double bw = text_w + 2 * TAG_PAD, bh = tag_height(cr, lines.size());
  if (bounds &&
      (x < 0 || y < 0 || x + bw > bounds->width || y + bh > bounds->height)) {
    return false;
  }

  rounded_path(cr, x, y, bw, bh, CORNER_R / 2);
  set_colour(cr, colour);
  cairo_fill(cr);

  cairo_set_source_rgb(cr, 1, 1, 1);
  for (size_t i = 0; i < lines.size(); ++i) {
    cairo_move_to(cr, x + TAG_PAD, y + TAG_PAD + fe.ascent + i * fe.height);
    cairo_show_text(cr, lines[i].c_str());
  }
  // show_text leaves a current point past the end of the text, and an arc
  // started while one exists is joined to it by a line. Clear it.
  cairo_new_path(cr);
  return true;
}

// Fallback when a peer expresses no colour preference. Indexed by source id so
// a peer keeps the same colour for as long as it is on the mesh.
cv::Scalar palette(int32_t id) {
  static const cv::Scalar P[] = {{0, 255, 255}, {255, 128, 0}, {255, 0, 255},
                                 {0, 128, 255}, {128, 255, 0}, {255, 0, 128}};
  return P[static_cast<uint32_t>(id) % (sizeof(P) / sizeof(P[0]))];
}

}  // namespace

HandtrackGui::~HandtrackGui() {
  if (tick_id_) g_source_remove(tick_id_);
  if (loop_) g_main_loop_unref(loop_);
}

int HandtrackGui::open(int width, int height, const char* title) {
  if (width <= 0 || height <= 0) return -1;
  width_ = width;
  height_ = height;
  frame_.assign(static_cast<size_t>(width_) * height_ * 4, 0);

  if (!gtk_init_check()) {
    fprintf(stderr, "gtk_init_check failed -- no display backend available.\n");
    return -2;
  }

  window_ = gtk_window_new();
  gtk_window_set_title(GTK_WINDOW(window_), title);
  gtk_window_set_default_size(GTK_WINDOW(window_), width_, height_);

  area_ = gtk_drawing_area_new();
  gtk_drawing_area_set_content_width(GTK_DRAWING_AREA(area_), width_);
  gtk_drawing_area_set_content_height(GTK_DRAWING_AREA(area_), height_);
  gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(area_),
                                 &HandtrackGui::on_draw, this, nullptr);
  gtk_window_set_child(GTK_WINDOW(window_), area_);

  loop_ = g_main_loop_new(nullptr, FALSE);

  // Key bindings: f (or F11) toggles fullscreen, q / Escape quits. GTK4 routes
  // keys through an event controller rather than a widget signal.
  GtkEventController* keys = gtk_event_controller_key_new();
  g_signal_connect(keys, "key-pressed",
                   G_CALLBACK(+[](GtkEventControllerKey*, guint keyval, guint,
                                  GdkModifierType, gpointer d) -> gboolean {
                     auto* self = static_cast<HandtrackGui*>(d);
                     switch (keyval) {
                       case GDK_KEY_f:
                       case GDK_KEY_F:
                       case GDK_KEY_F11:
                         self->toggle_fullscreen();
                         return TRUE;
                       case GDK_KEY_q:
                       case GDK_KEY_Escape:
                         self->quit();
                         return TRUE;
                       default:
                         return FALSE;
                     }
                   }),
                   this);
  gtk_widget_add_controller(window_, keys);

  // Closing the window ends the loop.
  g_signal_connect_swapped(
      window_, "close-request", G_CALLBACK(+[](gpointer d) -> gboolean {
        static_cast<HandtrackGui*>(d)->quit();
        return FALSE;  // let the default handler destroy it
      }),
      this);
  return 0;
}

void HandtrackGui::update_hands(const std::vector<Hand>& hands) {
  std::lock_guard<std::mutex> lk(hands_mtx_);
  hands_ = hands;
}

HandtrackGui::HandsCallback HandtrackGui::hands_callback() {
  return [this](const std::vector<Hand>& hands) { update_hands(hands); };
}

void HandtrackGui::update_visitor(const HandVisitor& visitor) {
  std::lock_guard<std::mutex> lk(visitors_mtx_);
  visitors_[visitor.id] = Visitor{visitor, std::chrono::steady_clock::now()};
}

void HandtrackGui::show_frame(const cv::Mat& bgr) {
  if (bgr.empty() || width_ <= 0) return;
  if (bgr.cols == width_ && bgr.rows == height_)
    bgr.copyTo(canvas_);
  else
    cv::resize(bgr, canvas_, {width_, height_});

  // cairo RGB24 wants a 32-bit pixel with the top byte unused, in native byte
  // order -- on little-endian that is exactly BGRX.
  cv::cvtColor(canvas_, bgrx_, cv::COLOR_BGR2BGRA);

  const size_t n = static_cast<size_t>(width_) * height_ * 4;
  std::lock_guard<std::mutex> lk(mtx_);
  if (frame_.size() != n) frame_.resize(n);
  memcpy(frame_.data(), bgrx_.data, n);
  have_frame_ = true;
}

void HandtrackGui::set_status(const std::string& text) {
  std::lock_guard<std::mutex> lk(mtx_);
  status_ = text;
}

void HandtrackGui::on_draw(GtkDrawingArea*, cairo_t* cr, int w, int h,
                           gpointer data) {
  auto* self = static_cast<HandtrackGui*>(data);

  std::string status;
  {
    std::lock_guard<std::mutex> lk(self->mtx_);
    if (!self->have_frame_) return;
    status = self->status_;

    // The staged buffer is already cairo's RGB24 layout, so this is a blit,
    // not a conversion.
    cairo_surface_t* surf = cairo_image_surface_create_for_data(
        self->frame_.data(), CAIRO_FORMAT_RGB24, self->width_, self->height_,
        self->width_ * 4);
    if (cairo_surface_status(surf) != CAIRO_STATUS_SUCCESS) {
      cairo_surface_destroy(surf);
      return;
    }

    // Scale to the widget if the user resized the window.
    cairo_save(cr);
    if (w != self->width_ || h != self->height_) {
      cairo_scale(cr, static_cast<double>(w) / self->width_,
                  static_cast<double>(h) / self->height_);
    }
    cairo_set_source_surface(cr, surf, 0, 0);
    cairo_paint(cr);
    cairo_surface_destroy(surf);
    cairo_restore(cr);
  }

  self->draw_overlay(cr, w, h);

  if (status.empty()) return;

  // Drawn in widget pixels, not frame pixels, so the label is unaffected by the
  // capture resolution. The font shrinks until the line fits rather than being
  // clipped at the window edge.
  const double margin = 6.0;
  cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL,
                         CAIRO_FONT_WEIGHT_BOLD);
  double size = 14.0;
  cairo_text_extents_t ext;
  for (;;) {
    cairo_set_font_size(cr, size);
    cairo_text_extents(cr, status.c_str(), &ext);
    if (ext.width <= w - 2 * margin || size <= 6.0) break;
    size -= 0.5;
  }

  // Dark plate behind the text so it stays readable over a bright frame.
  const double bar_h = ext.height + 2 * margin;
  cairo_set_source_rgba(cr, 0, 0, 0, 0.55);
  cairo_rectangle(cr, 0, 0, w, bar_h);
  cairo_fill(cr);

  cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);
  cairo_move_to(cr, margin, margin + ext.height);
  cairo_show_text(cr, status.c_str());
}

void HandtrackGui::draw_overlay(cairo_t* cr, int w, int h) {
  // Copy out under the locks and draw after releasing them: a producer should
  // never wait on the display thread. This is the only reader of either
  // container, so expiring dead visitors here costs nothing until something
  // actually draws.
  std::vector<Hand> hands;
  std::vector<HandVisitor> visitors;
  {
    std::lock_guard<std::mutex> lk(hands_mtx_);
    hands = hands_;
  }
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(visitors_mtx_);
    visitors.reserve(visitors_.size());
    for (auto it = visitors_.begin(); it != visitors_.end();) {
      if (now - it->second.seen > SOURCE_TTL) {
        it = visitors_.erase(it);
      } else {
        visitors.push_back(it->second.visitor);
        ++it;
      }
    }
  }
  if (hands.empty() && visitors.empty()) return;

  // Landmarks are in frame pixels, everything drawn here is in widget pixels.
  // Only positions are scaled: line widths, dot radii and text hold their size
  // whatever the window does.
  const double sx = static_cast<double>(w) / width_;
  const double sy = static_cast<double>(h) / height_;
  const cv::Size bounds(w, h);
  const auto map = [&](const cv::Point2f& p) {
    return std::make_pair(p.x * sx, p.y * sy);
  };

  cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL,
                         CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(cr, TAG_PX);
  cairo_set_line_width(cr, LINE_W);

  // Visitors first, so our own hands land on top of them.
  //
  // A dot where the peer sees a hand -- the mesh carries a point per hand, not
  // landmarks, so there is no skeleton to draw -- and their name over their
  // confidence beside it. Two narrow lines instead of one wide one: a wide tag
  // runs off the window far sooner, and one that would is dropped rather than
  // clipped.
  for (const auto& v : visitors) {
    const cv::Scalar colour = v.colour[0] < 0 ? palette(v.id) : v.colour;
    const std::string label = v.label.empty() ? std::to_string(v.id) : v.label;
    for (const auto& info : v.pts) {
      const auto [x, y] = map(info.pt);
      if (x < 0 || y < 0 || x >= w || y >= h) continue;

      cairo_new_sub_path(cr);
      cairo_arc(cr, x, y, DOT_R, 0, 2 * G_PI);
      set_colour(cr, colour);
      cairo_fill_preserve(cr);
      // White rim: a dark peer colour over a dark frame is otherwise invisible.
      cairo_set_source_rgb(cr, 1, 1, 1);
      cairo_set_line_width(cr, 1.0);
      cairo_stroke(cr);
      cairo_set_line_width(cr, LINE_W);

      const std::vector<std::string> lines = {
          label, std::format("{:.2f}", info.score)};
      draw_tag(cr, x + DOT_R + TAG_GAP, y - tag_height(cr, lines.size()) / 2,
               lines, colour, &bounds);
    }
  }

  for (const auto& hand : hands) {
    const cv::Rect box = hand_bbox(hand, cv::Size(width_, height_));
    if (box.empty()) continue;
    const double bx = box.x * sx, by = box.y * sy;
    const double bw = box.width * sx, bh = box.height * sy;

    // Box first: the fill is translucent, so anything under it is washed out
    // by the same 28%. The skeleton goes on top at full strength.
    rounded_path(cr, bx, by, bw, bh, CORNER_R);
    // Toward white, so a saturated edge reads against its own tint.
    set_colour(cr, MY_COLOUR * 0.5 + cv::Scalar(128, 128, 128), FILL_ALPHA);
    cairo_fill(cr);
    rounded_path(cr, bx, by, bw, bh, CORNER_R);
    set_colour(cr, MY_COLOUR);
    cairo_stroke(cr);

    for (const auto& c : CONN) {
      const auto [x0, y0] = map(hand.pts[c[0]]);
      const auto [x1, y1] = map(hand.pts[c[1]]);
      cairo_move_to(cr, x0, y0);
      cairo_line_to(cr, x1, y1);
    }
    cairo_stroke(cr);

    for (const auto& p : hand.pts) {
      const auto [x, y] = map(p);
      cairo_new_sub_path(cr);
      cairo_arc(cr, x, y, JOINT_R, 0, 2 * G_PI);
      cairo_fill(cr);
    }

    // "Larry Right 0.99" -- who, then handedness, then the presence score, so a
    // marginal detection is visible per hand rather than only in the status
    // line. On a shared display "whose hand is that" matters more than the
    // score, so the label leads.
    //
    // Above the box, or below it when there is no room above: a plate inside
    // the box covers the landmarks it labels.
    const char* side = hand.handed > 0.5f ? "Right" : "Left";
    const std::vector<std::string> lines = {
        std::format("{} {} {:.2f}", MY_LABEL, side, hand.score)};
    const double th = tag_height(cr, lines.size());
    const double ty =
        by - th - TAG_GAP >= 0 ? by - th - TAG_GAP : by + bh + TAG_GAP;
    draw_tag(cr, bx, ty, lines, MY_COLOUR, nullptr);
  }
}

gboolean HandtrackGui::on_tick(gpointer data) {
  auto* self = static_cast<HandtrackGui*>(data);
  if (self->area_) gtk_widget_queue_draw(self->area_);
  return G_SOURCE_CONTINUE;
}

void HandtrackGui::run() {
  if (!window_ || !loop_) return;
  gtk_window_present(GTK_WINDOW(window_));

  // Repaint on a timer rather than scheduling from the camera thread: frames
  // arrive faster than they need to be drawn, and this keeps every GTK call on
  // the main thread. ~60 Hz.
  tick_id_ = g_timeout_add(16, &HandtrackGui::on_tick, this);

  g_main_loop_run(loop_);
}

void HandtrackGui::quit() {
  if (loop_ && g_main_loop_is_running(loop_)) g_main_loop_quit(loop_);
}

void HandtrackGui::toggle_fullscreen() {
  if (!window_) return;
  fullscreen_ = !fullscreen_;
  if (fullscreen_)
    gtk_window_fullscreen(GTK_WINDOW(window_));
  else
    gtk_window_unfullscreen(GTK_WINDOW(window_));
}
