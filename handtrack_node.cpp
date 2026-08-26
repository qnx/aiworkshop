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

// handtrack_node.cpp — ROS 2 node: camera in, hand landmarks out.
// Camera API capture (RpiCam, callback style) -> native TFLite hand pipeline
// (HandTrack) -> published on a topic, and pushed to HandtrackGui for display.
//
// Everything visual lives in HandtrackGui; this file is capture, inference and
// ROS plumbing. Our own hands go to the GUI through a callback; a peer's arrive
// on the topic and go to the GUI as visitors.
//
// Three threads, and GTK owns the one main() runs on:
//   - camera thread: RpiCam invokes our frame callback per viewfinder frame. It
//     hands the newest frame to the worker and passes the frame to the GUI,
//     never blocking on the pipeline.
//   - worker thread: runs inference, publishes, and pushes hands to the GUI.
//   - executor thread: rclcpp::spin, so the node answers `ros2 node`/`ros2
//   param`
//     while GTK's main loop has the original thread.
// Video stays at camera rate; the skeleton lags by roughly one inference.
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <format>
#include <handtrack/msg/hand_set.hpp>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <set>
#include <std_msgs/msg/color_rgba.hpp>
#include <string>
#include <thread>
#include <vector>

#include "handtrack_gui.hpp"
#include "handtracking.hpp"
#include "rpi_cam.hpp"

constexpr int DEFAULT_CAM_UNIT = 5;  // USB camera for RPI5
constexpr const char* TOPIC = "/handtrack/hands";

// Cleared once the window closes or rclcpp is shut down (Ctrl-C). rclcpp
// installs its own SIGINT handler, so there is no signal() call here -- the GTK
// timer at the bottom of main() watches rclcpp::ok() and closes the window.
static std::atomic<bool> g_run{true};

// Frame handoff: one slot, newest wins. Frames the worker misses are dropped.
static std::mutex g_in_mtx;
static std::condition_variable g_in_cv;
static cv::Mat g_in_frame;
static bool g_in_ready = false;

// Inference results go straight to the GUI and the topic, so nothing is kept
// here but the numbers the status line needs.
static std::atomic<long> g_infers{0};
// Cumulative, not the last reading: the status line reports the mean over the
// same window it measures the rates over, so the two agree.
static std::atomic<long> g_infer_us_total{0};
static std::atomic<int> g_hand_count{0};
static std::atomic<float> g_best_score{0.f};

static int32_t g_source_id = 0;

using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// ColorRGBA -> the GUI's BGR scalar. Alpha 0 means the sender had no
// preference, which HandVisitor signals with a negative component.
static cv::Scalar to_bgr(const std_msgs::msg::ColorRGBA& c) {
  if (c.a <= 0.f) return {-1, -1, -1};
  return {c.b * 255.0, c.g * 255.0, c.r * 255.0};
}

// The reverse, for publishing MY_COLOUR.
static std_msgs::msg::ColorRGBA to_rgba(const cv::Scalar& bgr) {
  std_msgs::msg::ColorRGBA c;
  c.b = static_cast<float>(bgr[0] / 255.0);
  c.g = static_cast<float>(bgr[1] / 255.0);
  c.r = static_cast<float>(bgr[2] / 255.0);
  c.a = 1.f;
  return c;
}

// One message per inference, so an empty hands[] is a positive "no hands this
// frame" rather than silence.
//
// A peer draws a dot, not a skeleton, so only the centre of each hand goes on
// the wire -- normalized against the frame it was found in, so a peer with a
// different camera resolution can scale it to its own window without knowing
// anything about ours.
static handtrack::msg::HandSet to_msg(const std::vector<Hand>& hands,
                                      cv::Size frame,
                                      const rclcpp::Time& stamp) {
  handtrack::msg::HandSet msg;
  msg.header.stamp = stamp;
  msg.source_id = g_source_id;
  msg.label = MY_LABEL;
  msg.colour = to_rgba(MY_COLOUR);
  msg.hands.reserve(hands.size());

  const double fw = frame.width, fh = frame.height;
  for (const auto& h : hands) {
    handtrack::msg::Hand out;
    const cv::Rect box = hand_bbox(h, frame);
    out.point.x = (box.x + box.width / 2.0) / fw;
    out.point.y = (box.y + box.height / 2.0) / fh;
    out.score = h.score;
    out.handedness = h.handed;
    msg.hands.push_back(out);
  }
  return msg;
}

static void infer_worker(
    HandTrack* pipe, rclcpp::Node* node,
    rclcpp::Publisher<handtrack::msg::HandSet>::SharedPtr pub,
    HandtrackGui::HandsCallback on_hands) {
  for (;;) {
    cv::Mat frame;
    {
      std::unique_lock<std::mutex> lk(g_in_mtx);
      g_in_cv.wait(lk, [] { return g_in_ready || !g_run.load(); });
      if (!g_run.load()) return;
      frame = std::move(g_in_frame);
      g_in_ready = false;
    }

    const auto started = Clock::now();
    auto hands = pipe->detect(frame, 2);
    const auto finished = Clock::now();

    // Out to the mesh, where peers draw us as a visitor, and to our own
    // overlay, which draws the full skeleton.
    pub->publish(to_msg(hands, frame.size(), node->now()));

    float best = 0.f;
    for (const auto& h : hands) best = std::max(best, h.score);
    g_hand_count.store(static_cast<int>(hands.size()));
    g_best_score.store(best);
    on_hands(hands);

    g_infer_us_total += std::chrono::duration_cast<std::chrono::microseconds>(
                            finished - started)
                            .count();
    ++g_infers;
  }
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("handtrack_node");

  const int unit =
      node->declare_parameter<int>("camera_unit", DEFAULT_CAM_UNIT);

  auto pub =
      node->create_publisher<handtrack::msg::HandSet>(TOPIC, rclcpp::QoS(1));

  // Generate random source id
  // In theory it is possible for collision but the likelyhood is so little
  // lets pretend that it could never happen.
  srand(time(NULL));
  g_source_id = rand();

  // Camera first: the window is opened at the camera's native size, so the
  // frame is never scaled up only for inference to scale it back down.
  RpiCam cam(static_cast<camera_unit_t>(unit));
  int err = cam.init();
  if (err != EOK) {
    RCLCPP_ERROR(node->get_logger(), "camera init failed: %s", strerror(err));
    return 1;
  }
  auto [cw, ch] = cam.get_frame_size();

  HandtrackGui gui;
  if (gui.open(cw, ch) != 0) {
    RCLCPP_ERROR(node->get_logger(), "gtk display open failed");
    return 1;
  }
  const int DW = gui.width(), DH = gui.height();

  RCLCPP_INFO(node->get_logger(), "camera unit %d  %dx%d  window %dx%d", unit,
              cw, ch, DW, DH);
  if ((cw | ch) & 1) {
    RCLCPP_ERROR(node->get_logger(), "need even camera dimensions for NV12");
    return 1;
  }

  // Find the ML model from our ros2 directory
  std::string model_dir;
  try {
    model_dir = ament_index_cpp::get_package_share_directory("handtrack") + "/";
  } catch (const std::exception& e) {
    RCLCPP_WARN(node->get_logger(),
                "package share dir not found (%s), using ./models", e.what());
  }
  // The model paths come from HandTrackConfig's defaults and are relative to
  // the share directory, so they need resolving before the pipeline loads them.
  HandTrackConfig cfg;

  // Params for model
  cfg.det_th = node->declare_parameter<double>("det_th", cfg.det_th);
  cfg.presence_th =
      node->declare_parameter<double>("presence_th", cfg.presence_th);
  cfg.track_scale =
      node->declare_parameter<double>("track_scale", cfg.track_scale);
  cfg.tracking = node->declare_parameter<bool>("tracking", cfg.tracking);
  cfg.redetect_every =
      node->declare_parameter<int>("redetect_every", cfg.redetect_every);
  cfg.mirror = node->declare_parameter<bool>("mirror", cfg.mirror);
  cfg.threads = node->declare_parameter<int>("threads", cfg.threads);
  cfg.det_model = model_dir + cfg.det_model;
  cfg.lm_model = model_dir + cfg.lm_model;

  HandTrack pipe(cfg);
  RCLCPP_INFO(node->get_logger(), "hand pipeline ready; source %d (%s) on %s",
              g_source_id, MY_LABEL, pub->get_topic_name());

  // The other half of the mesh. Created here rather than beside the publisher
  // because scaling a peer's normalized points needs the window size, which
  // only exists once the GUI is open.
  const auto on_hands = gui.hands_callback();
  const cv::Size window(DW, DH);
  auto sub = node->create_subscription<handtrack::msg::HandSet>(
      TOPIC, rclcpp::QoS(10),
      [&gui, window](const handtrack::msg::HandSet::SharedPtr msg) {
        // Our own message came back to us; we already draw those as hands.
        if (msg->source_id == g_source_id) return;

        HandVisitor visitor;
        visitor.id = msg->source_id;
        visitor.label = msg->label;
        visitor.colour = to_bgr(msg->colour);
        visitor.pts.reserve(msg->hands.size());
        for (const auto& in : msg->hands) {
          HandVisitor::Info info;
          info.pt = cv::Point2f(static_cast<float>(in.point.x * window.width),
                                static_cast<float>(in.point.y * window.height));
          info.score = in.score;
          info.handedness = in.handedness;
          visitor.pts.push_back(info);
        }
        gui.update_visitor(visitor);
      });

  std::thread worker(infer_worker, &pipe, node.get(), pub, on_hands);

  // GTK takes this thread at gui.run(), so the executor needs its own.
  std::thread executor([node] { rclcpp::spin(node); });

  // Rates are measured over a rolling window, so every figure in the status
  // line describes the same recent stretch of time.
  constexpr double REPORT_SEC = 0.5;
  auto win_start = Clock::now();
  long frames = 0, win_frames = 0, win_infers = 0, win_us = 0, reports = 0;
  cv::Mat disp;

  // Runs on the camera library's thread for every viewfinder frame. RpiCam has
  // already converted NV12 to BGR at camera resolution, and the window is
  // opened at that same size, so this is normally a copy rather than a rescale.
  auto on_frame = [&](const cv::Mat& bgr) {
    if (!g_run.load()) return;

    // Resized here rather than in the GUI because inference runs on this same
    // frame, and the landmarks it returns have to be in window coordinates
    // for every source's overlay to line up.
    if (bgr.cols == DW && bgr.rows == DH)
      bgr.copyTo(disp);
    else
      cv::resize(bgr, disp, {DW, DH});

    // Hand off a private copy; disp is reused on the next frame.
    cv::Mat next = disp.clone();
    {
      std::lock_guard<std::mutex> lk(g_in_mtx);
      g_in_frame = std::move(next);
      g_in_ready = true;
    }
    g_in_cv.notify_one();

    ++frames;
    const auto now = Clock::now();
    const double dt = secs(win_start, now);
    if (dt >= REPORT_SEC) {
      const long infers = g_infers.load(), us = g_infer_us_total.load();
      const long ran = infers - win_infers;
      // Mean over the window, matching the rates beside it. ai fps is capped
      // by the camera -- one inference per delivered frame -- so it tracks fps
      // until inference is the slower of the two.
      const double ms = ran ? (us - win_us) / 1000.0 / ran : 0.0;
      // Show the best score alongside the count: a detection sitting just over
      // the threshold looks the same as a solid one in a bare hand count.
      const std::string label = std::format(
          "Camera {:.1f} fps  AI Inference {:.1f} fps ({:.0f} ms)  hands {} "
          "({:.2f})",
          (frames - win_frames) / dt, ran / dt, ms, g_hand_count.load(),
          g_best_score.load());
      // Drawn by GTK in widget coordinates, so the line is unaffected by the
      // capture resolution.
      gui.set_status(label);

      // Throttled: the status line is already on the window, and a node's
      // stdout usually ends up in a launch log nobody is watching live.
      if (++reports % 20 == 0)
        RCLCPP_INFO(node->get_logger(), "%s", label.c_str());

      win_start = now;
      win_frames = frames;
      win_infers = infers;
      win_us = us;
    }

    // The GUI draws every source's hands over this frame, ours included.
    gui.show_frame(disp);
  };

  err = cam.start(on_frame);
  if (err != EOK) {
    RCLCPP_ERROR(node->get_logger(), "camera start failed: %s", strerror(err));
    g_run = false;
    g_in_cv.notify_all();
    worker.join();
    rclcpp::shutdown();
    executor.join();
    return 1;
  }

  // GTK owns this thread from here: frames arrive on the camera thread and are
  // staged by gui.show(), while the GTK loop repaints and services the
  // window. A low-frequency timer closes the window when rclcpp goes down
  // (Ctrl-C, or `ros2 lifecycle`-style external shutdown), since a signal
  // handler cannot safely call into GLib itself.
  g_timeout_add(
      100,
      +[](gpointer d) -> gboolean {
        if (!g_run.load() || !rclcpp::ok()) {
          static_cast<HandtrackGui*>(d)->quit();
          return G_SOURCE_REMOVE;
        }
        return G_SOURCE_CONTINUE;
      },
      &gui);

  gui.run();

  // Closing the window shuts the node down too, so either exit path converges
  // here: stop feeding the worker, then let the executor fall out of spin().
  g_run = false;
  cam.stop();            // no more callbacks after this returns
  g_in_cv.notify_all();  // g_run is already false; wake the worker to exit.
  worker.join();
  rclcpp::shutdown();  // returns immediately if Ctrl-C already did it
  executor.join();
  return 0;
}
