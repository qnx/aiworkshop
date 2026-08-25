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

#include <gdk/gdkkeysyms.h>   // GDK_KEY_* are not pulled in by gtk/gtk.h

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>

namespace {

// A source that stops calling update_hands() stops being drawn. Without this a
// peer that drops off the mesh -- or crashes -- leaves its last hand frozen on
// every other display, which reads as a live hand that simply never moves.
constexpr auto SOURCE_TTL = std::chrono::seconds(3);

const int CONN[][2] = {
    {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},{11,12},
    {9,13},{13,14},{14,15},{15,16},{13,17},{17,18},{18,19},{19,20},{0,17}};

constexpr double FILL_ALPHA = 0.28;   // enough tint to group the hand, light
                                      // enough to see it through
constexpr int CORNER_R = 10;


// Toward white, for the translucent fill under a saturated edge.
cv::Scalar lighten(const cv::Scalar& c, double f) {
    return {c[0] + (255 - c[0]) * f, c[1] + (255 - c[1]) * f, c[2] + (255 - c[2]) * f};
}

// OpenCV has no rounded rectangle. Filled: two overlapping rects that leave the
// corners bare, plus a disc in each. Outlined: the four sides, pulled in by the
// radius, plus a quarter-circle arc joining each pair.
void rounded_rect(cv::Mat& img, cv::Rect r, int rad, const cv::Scalar& c, int thickness) {
    rad = std::min(rad, std::min(r.width, r.height) / 2);
    if (rad < 1) { cv::rectangle(img, r, c, thickness, cv::LINE_AA); return; }
    const int x0 = r.x, y0 = r.y, x1 = r.x + r.width - 1, y1 = r.y + r.height - 1;
    const cv::Point tl{x0 + rad, y0 + rad}, tr{x1 - rad, y0 + rad},
                    br{x1 - rad, y1 - rad}, bl{x0 + rad, y1 - rad};

    if (thickness == cv::FILLED) {
        cv::rectangle(img, {x0 + rad, y0, r.width - 2 * rad, r.height}, c, cv::FILLED);
        cv::rectangle(img, {x0, y0 + rad, r.width, r.height - 2 * rad}, c, cv::FILLED);
        for (const auto& p : {tl, tr, br, bl})
            cv::circle(img, p, rad, c, cv::FILLED, cv::LINE_AA);
        return;
    }
    cv::line(img, {tl.x, y0}, {tr.x, y0}, c, thickness, cv::LINE_AA);
    cv::line(img, {bl.x, y1}, {br.x, y1}, c, thickness, cv::LINE_AA);
    cv::line(img, {x0, tl.y}, {x0, bl.y}, c, thickness, cv::LINE_AA);
    cv::line(img, {x1, tr.y}, {x1, br.y}, c, thickness, cv::LINE_AA);
    cv::ellipse(img, tl, {rad, rad}, 180, 0, 90, c, thickness, cv::LINE_AA);
    cv::ellipse(img, tr, {rad, rad}, 270, 0, 90, c, thickness, cv::LINE_AA);
    cv::ellipse(img, br, {rad, rad},   0, 0, 90, c, thickness, cv::LINE_AA);
    cv::ellipse(img, bl, {rad, rad},  90, 0, 90, c, thickness, cv::LINE_AA);
}

// Box plus a "peer2 Right 0.99" tag -- who, then handedness, then the presence
// score, so a marginal detection is visible per hand rather than only in the
// status line. On a shared display "whose hand is that" matters more than the
// score, so the label leads.
void draw_box(cv::Mat& img, const Hand& h, const std::string& label, const cv::Scalar& edge) {
    cv::Rect box = hand_bbox(h, img.size());
    if (box.empty()) return;

    // Blend the tint into the box region only. Compositing a full-frame overlay
    // would cost two 640x480 passes per hand per frame on the display thread;
    // this touches the box and nothing else. The rounded mask is what keeps the
    // blend off the corners -- blending square and outlining round would leave
    // four tinted nubs poking outside the edge.
    cv::Mat roi = img(box);
    cv::Mat tint(roi.size(), roi.type(), lighten(edge, 0.5)), blended;
    cv::addWeighted(tint, FILL_ALPHA, roi, 1.0 - FILL_ALPHA, 0.0, blended);
    cv::Mat mask(roi.size(), CV_8UC1, cv::Scalar(0));
    rounded_rect(mask, {0, 0, box.width, box.height}, CORNER_R, cv::Scalar(255), cv::FILLED);
    blended.copyTo(roi, mask);
    rounded_rect(img, box, CORNER_R, edge, 2);

    const char* side = h.handed > 0.5f ? "Right" : "Left";
    std::string tag = label.empty() ? std::format("{} {:.2f}", side, h.score)
                                    : std::format("{} {} {:.2f}", label, side, h.score);
    int base = 0;
    cv::Size ts = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    // Inside the box, top-left. Clipped to the box, so a hand at the frame edge
    // keeps its label instead of pushing it off-screen.
    // The bar stays opaque: text over the blended fill is what makes a label
    // unreadable exactly when the hand is over something busy.
    cv::Rect bar = cv::Rect{box.x + 2, box.y + 2, ts.width + 6, ts.height + base + 4} & box;
    if (!bar.empty()) rounded_rect(img, bar, CORNER_R / 2, edge, cv::FILLED);
    cv::putText(img, tag, {box.x + 5, box.y + 5 + ts.height}, cv::FONT_HERSHEY_SIMPLEX,
                0.5, {255, 255, 255}, 1, cv::LINE_AA);
}

}  // namespace

HandtrackGui::~HandtrackGui() {
    if (tick_id_) g_source_remove(tick_id_);
    if (loop_) g_main_loop_unref(loop_);
}

int HandtrackGui::open(int width, int height, const char *title) {
    if (width <= 0 || height <= 0) return -1;
    width_ = width;
    height_ = height;
    frame_.assign(static_cast<size_t>(width_) * height_ * 4, 0);

    if (!gtk_init_check()) {
        fprintf(stderr, "gtk_init_check failed -- no display backend available.\n"
                        "GTK4 here uses the qnxscreen GDK backend; Screen must be running.\n");
        return -2;
    }

    window_ = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window_), title);
    gtk_window_set_default_size(GTK_WINDOW(window_), width_, height_);

    area_ = gtk_drawing_area_new();
    gtk_drawing_area_set_content_width(GTK_DRAWING_AREA(area_), width_);
    gtk_drawing_area_set_content_height(GTK_DRAWING_AREA(area_), height_);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(area_), &HandtrackGui::on_draw, this, nullptr);
    gtk_window_set_child(GTK_WINDOW(window_), area_);

    loop_ = g_main_loop_new(nullptr, FALSE);

    // Key bindings: f (or F11) toggles fullscreen, q / Escape quits. GTK4 routes
    // keys through an event controller rather than a widget signal.
    GtkEventController *keys = gtk_event_controller_key_new();
    g_signal_connect(keys, "key-pressed",
                     G_CALLBACK(+[](GtkEventControllerKey *, guint keyval, guint, GdkModifierType,
                                    gpointer d) -> gboolean {
                         auto *self = static_cast<HandtrackGui *>(d);
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
    g_signal_connect_swapped(window_, "close-request", G_CALLBACK(+[](gpointer d) -> gboolean {
                                 static_cast<HandtrackGui *>(d)->quit();
                                 return FALSE;   // let the default handler destroy it
                             }),
                             this);
    return 0;
}

void HandtrackGui::update_hands(const HandSource &source, const std::vector<Hand> &hands) {
    std::lock_guard<std::mutex> lk(hands_mtx_);
    sources_[source.id] = SourceHands{source, hands, std::chrono::steady_clock::now()};
}

HandtrackGui::HandsCallback HandtrackGui::hands_callback() {
    return [this](const HandSource &source, const std::vector<Hand> &hands) {
        update_hands(source, hands);
    };
}

void HandtrackGui::draw_hands(cv::Mat &bgr) {
    // Copy out under the lock: the drawing itself is the slow part and a producer
    // should never wait on it. Expiring here rather than on a timer means a dead
    // source costs nothing until something actually draws.
    std::vector<SourceHands> live;
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(hands_mtx_);
        live.reserve(sources_.size());
        for (auto it = sources_.begin(); it != sources_.end();) {
            if (now - it->second.seen > SOURCE_TTL) it = sources_.erase(it);
            else { live.push_back(it->second); ++it; }
        }
    }

    for (const auto &entry : live) {
        const cv::Scalar colour = ;
        const std::string &label =
            entry.source.label.empty() ? entry.source.id : entry.source.label;

        // Boxes first: the tint is blended, so anything drawn before it would be
        // washed out by the same 28%. The skeleton goes on top at full strength.
        if (h.local) {
            for (const auto &h : entry.hands)  {
                draw_box(bgr, h, label, entry.source.colour);
            }
        }
        for (const auto &h : entry.hands) {
            for (const auto &c : CONN) {
                cv::line(bgr, h.pts[c[0]], h.pts[c[1]], {0, 255, 0}, 2, cv::LINE_AA);
            }

            for (const auto &p : h.pts) {
                cv::circle(bgr, p, 4, entry.source.colour, -1, cv::LINE_AA);
            }
        }
    }
}

void HandtrackGui::show_frame(const cv::Mat &bgr) {
    if (bgr.empty() || width_ <= 0) return;
    if (bgr.cols == width_ && bgr.rows == height_) bgr.copyTo(canvas_);
    else cv::resize(bgr, canvas_, {width_, height_});

    draw_hands(canvas_);

    // cairo RGB24 wants a 32-bit pixel with the top byte unused, in native byte
    // order -- on little-endian that is exactly BGRX.
    cv::cvtColor(canvas_, bgrx_, cv::COLOR_BGR2BGRA);

    const size_t n = static_cast<size_t>(width_) * height_ * 4;
    std::lock_guard<std::mutex> lk(mtx_);
    if (frame_.size() != n) frame_.resize(n);
    memcpy(frame_.data(), bgrx_.data, n);
    have_frame_ = true;
}

void HandtrackGui::set_status(const std::string &text) {
    std::lock_guard<std::mutex> lk(mtx_);
    status_ = text;
}

void HandtrackGui::on_draw(GtkDrawingArea *, cairo_t *cr, int w, int h, gpointer data) {
    auto *self = static_cast<HandtrackGui *>(data);

    std::string status;
    {
        std::lock_guard<std::mutex> lk(self->mtx_);
        if (!self->have_frame_) return;
        status = self->status_;

        // The staged buffer is already cairo's RGB24 layout, so this is a blit,
        // not a conversion.
        cairo_surface_t *surf = cairo_image_surface_create_for_data(
            self->frame_.data(), CAIRO_FORMAT_RGB24, self->width_, self->height_, self->width_ * 4);
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

    if (status.empty()) return;

    // Drawn in widget pixels, not frame pixels, so the label is unaffected by the
    // capture resolution. The font shrinks until the line fits rather than being
    // clipped at the window edge.
    const double margin = 6.0;
    cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
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

gboolean HandtrackGui::on_tick(gpointer data) {
    auto *self = static_cast<HandtrackGui *>(data);
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
    if (fullscreen_) gtk_window_fullscreen(GTK_WINDOW(window_));
    else gtk_window_unfullscreen(GTK_WINDOW(window_));
}
