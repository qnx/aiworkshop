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
  */
#pragma once

#include "handtracking.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <gtk/gtk.h>

/**
 * @brief Who a set of hands came from, and how they want to be drawn.
 *
 * Deliberately free of any ROS type: the GUI does not care whether a source is
 * this machine's own camera or a peer decoded off a topic.
 */
struct HandSource {
    std::string id;      ///< Unique ID. generated on launch
    std::string label;   ///< Drawn on the overlay; falls back to id when empty.
    bool local = false;  ///< If this is generated locally or over the network.
    cv::Scalar color{-1, -1, -1}; ///< Preferred colour, in BGR.
};

class HandtrackGui {
   public:
    /// A sink for landmarks. Anything that produces hands -- the local worker, a
    /// subscription carrying a peer's hands -- can hold one of these and stay
    /// ignorant of the GUI.
    using HandsCallback =
        std::function<void(const HandSource &source, const std::vector<Hand> &hands)>;

    HandtrackGui() = default;
    ~HandtrackGui();

    HandtrackGui(const HandtrackGui &) = delete;
    HandtrackGui &operator=(const HandtrackGui &) = delete;

    /**
     * @brief Create the window and drawing area at the given size.
     *
     * Must be called on the main thread, before run().
     *
     * @return 0 on success, negative on failure.
     */
    int open(int width, int height, const char *title = "handtrack");

    /// Gets the drawing size in <width, height> format
    std::pair<int, int> get_size() const { return std::make_pair(width_, height_); }

    int width() const { return width_; }
    int height() const { return height_; }

    /**
     * @brief Draw a camera frame plus every source's hands, and stage it.
     *
     * Takes BGR at any size (rescaled to the window if it does not match, though
     * landmarks are assumed to be in window coordinates). Composites on the
     * calling thread -- call it from one thread only, normally the capture
     * callback.
     */
    void show_frame(const cv::Mat &bgr);

    /**
     * @brief Replace one source's hands with its newest set. Safe from any thread.
     *
     * An empty vector means "this source sees no hands", which is different from
     * a source going silent: a source that stops calling is dropped from the
     * display after a couple of seconds rather than leaving a hand frozen on
     * screen forever.
     *
     * Landmarks must already be in window coordinates -- a caller relaying a
     * peer's hands is responsible for scaling them.
     */
    void update_hands(const HandSource &source, const std::vector<Hand> &hands);

    /// update_hands() as a callable, for handing to a producer.
    HandsCallback hands_callback();

    /**
     * @brief Set the status line drawn over the frame. Safe from any thread.
     *
     * Drawn in widget coordinates rather than baked into the frame, so it stays
     * legible at any window size and shrinks to fit rather than being clipped.
     */
    void set_status(const std::string &text);

    /// Run the GTK main loop. Blocks until the window closes or quit() is called.
    void run();

    /// Ask the main loop to exit. Safe to call from any thread.
    void quit();

    /// Toggle fullscreen. Bound to `f` / F11; main-thread only.
    void toggle_fullscreen();

   private:
    struct SourceHands {
        HandSource source;
        std::vector<Hand> hands;
        std::chrono::steady_clock::time_point seen;
    };

    static void on_draw(GtkDrawingArea *area, cairo_t *cr, int w, int h, gpointer data);
    static gboolean on_tick(gpointer data);

    /// Draw every live source's hands onto a BGR image, in place.
    void draw_hands(cv::Mat &bgr);

    GtkWidget *window_ = nullptr;
    GtkWidget *area_ = nullptr;
    GMainLoop *loop_ = nullptr;
    guint tick_id_ = 0;

    std::mutex mtx_;
    std::vector<uint8_t> frame_;   // staged BGRX, width_*height_*4
    bool have_frame_ = false;
    std::string status_;

    std::mutex hands_mtx_;
    std::map<std::string, SourceHands> sources_;

    // Scratch for show_frame(), reused per frame. Capture thread only.
    cv::Mat canvas_, bgrx_;

    int width_ = 0;
    int height_ = 0;
    bool fullscreen_ = false;
};
