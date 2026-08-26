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

/**
 * @file handtrack_gui.hpp
 *
 * GTK4 GUI for sensor framework camera with handtrack overlay.
 *
 * Two overlays, drawn from two different sources:
 *   - our own hands, from the local pipeline: full 21-point skeleton and box.
 *   - "visitors", from peers on the network: one dot per hand plus their tag.
 *     The mesh only carries a point per hand, so there is no skeleton to draw.
 */
#pragma once

#include <gtk/gtk.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

#include "handtracking.hpp"

/**
 * Choose your Colour and and label which appears in the gui
 */
inline constexpr const char* MY_LABEL = "NAME";
inline const cv::Scalar MY_COLOUR{255, 0, 0};  // BGR

/// One peer's hands as they arrive off the network.
struct HandVisitor {
  int32_t id = 0;
  std::string label;
  cv::Scalar colour{-1, -1, -1};  // negative == no preference, pick for them

  struct Info {
    cv::Point2f pt;  // window coordinates
    float score = 0.f;
    float handedness = 0.5f;
  };
  std::vector<Info> pts;
};

class HandtrackGui {
 public:
  /// A sink for our own landmarks, so the inference worker can stay ignorant of
  /// the GUI.
  using HandsCallback = std::function<void(const std::vector<Hand>&)>;

  HandtrackGui() = default;
  ~HandtrackGui();

  HandtrackGui(const HandtrackGui&) = delete;
  HandtrackGui& operator=(const HandtrackGui&) = delete;

  /**
   * @brief Create the window and drawing area at the given size.
   *
   * Must be called on the main thread, before run().
   *
   * @return 0 on success, negative on failure.
   */
  int open(int width, int height, const char* title = "handtrack");

  /// Gets the drawing size in <width, height> format
  std::pair<int, int> get_size() const {
    return std::make_pair(width_, height_);
  }

  int width() const { return width_; }
  int height() const { return height_; }

  /**
   * @brief Stage a camera frame for display.
   *
   * Takes BGR at any size, rescaled to the window if it does not match. The
   * overlays are drawn over it on the GTK thread, so this is video only. Call
   * it from one thread only, normally the capture callback.
   */
  void show_frame(const cv::Mat& bgr);

  /// Replace our own hands with the newest set. Safe from any thread.
  void update_hands(const std::vector<Hand>& hands);

  /**
   * @brief Replace one peer's visitor with its newest set. Safe from any
   * thread.
   *
   * An empty pts means "this peer sees no hands", which is different from a
   * peer going silent: a peer that stops publishing is dropped from the display
   * after a couple of seconds rather than leaving a dot frozen on screen.
   *
   * Points must already be in window coordinates -- the caller decoding the
   * message is responsible for scaling them.
   */
  void update_visitor(const HandVisitor& visitor);

  /// update_hands() as a callable, for handing to a producer.
  HandsCallback hands_callback();

  /**
   * @brief Set the status line drawn over the frame. Safe from any thread.
   *
   * Drawn in widget coordinates rather than baked into the frame, so it stays
   * legible at any window size and shrinks to fit rather than being clipped.
   */
  void set_status(const std::string& text);

  /// Run the GTK main loop. Blocks until the window closes or quit() is called.
  void run();

  /// Ask the main loop to exit. Safe to call from any thread.
  void quit();

  /// Toggle fullscreen. Bound to `f` / F11; main-thread only.
  void toggle_fullscreen();

 private:
  struct Visitor {
    HandVisitor visitor;
    std::chrono::steady_clock::time_point seen;
  };

  static void on_draw(GtkDrawingArea* area, cairo_t* cr, int w, int h,
                      gpointer data);
  static gboolean on_tick(gpointer data);

  /// Draw both overlays -- our hands, and the visitors -- over the staged
  /// frame, in widget pixels, on the GTK thread.
  void draw_overlay(cairo_t* cr, int w, int h);

  GtkWidget* window_ = nullptr;
  GtkWidget* area_ = nullptr;
  GMainLoop* loop_ = nullptr;
  guint tick_id_ = 0;

  std::mutex mtx_;
  std::vector<uint8_t> frame_;  // staged BGRX, width_*height_*4
  bool have_frame_ = false;
  std::string status_;

  std::mutex hands_mtx_;
  std::vector<Hand> hands_;

  std::mutex visitors_mtx_;
  std::map<int32_t, Visitor> visitors_;

  // Scratch for show_frame(), reused per frame. Capture thread only.
  cv::Mat canvas_, bgrx_;

  int width_ = 0;
  int height_ = 0;
  bool fullscreen_ = false;
};
