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
// ROS plumbing. The GUI takes hands keyed by source, so the same callback that
// carries our own landmarks will carry a peer's once the mesh exists here.
//
// Three threads, and GTK owns the one main() runs on:
//   - camera thread: RpiCam invokes our frame callback per viewfinder frame. It
//     hands the newest frame to the worker and passes the frame to the GUI,
//     never blocking on the pipeline.
//   - worker thread: runs inference, publishes, and pushes hands to the GUI.
//   - executor thread: rclcpp::spin, so the node answers `ros2 node`/`ros2 param`
//     while GTK's main loop has the original thread.
// Video stays at camera rate; the skeleton lags by roughly one inference.
#include "handtracking.hpp"
#include "rpi_cam.hpp"
#include "handtrack_gui.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <handtrack/msg/hand_set.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <format>
#include <mutex>
#include <set>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

constexpr int DISPLAY_PRIO = 12;
constexpr int INFER_PRIO   = 8;
constexpr int DEFAULT_CAM_UNIT = 1;
constexpr const char* TOPIC = "/handtrack/hands";

static void set_prio(int prio) {
    struct sched_param sp;
    int policy = 0;
    if (pthread_getschedparam(pthread_self(), &policy, &sp) != 0) return;
    sp.sched_priority = prio;
    if (pthread_setschedparam(pthread_self(), SCHED_RR, &sp) != 0)
        fprintf(stderr, "warn: could not set thread priority %d\n", prio);
}

// Cleared once the window closes or rclcpp is shut down (Ctrl-C). rclcpp installs
// its own SIGINT handler, so there is no signal() call here -- the GTK timer at
// the bottom of main() watches rclcpp::ok() and closes the window.
static std::atomic<bool> g_run{true};

// Frame handoff: one slot, newest wins. Frames the worker misses are dropped.
static std::mutex              g_in_mtx;
static std::condition_variable g_in_cv;
static cv::Mat                 g_in_frame;
static bool                    g_in_ready = false;

// Inference results go straight to the GUI and the topic, so nothing is kept
// here but the numbers the status line needs.
static std::atomic<long>  g_infers{0};
static std::atomic<long>  g_infer_us{0};
static std::atomic<int>   g_hand_count{0};
static std::atomic<float> g_best_score{0.f};

static double elapsed(const struct timespec& a, const struct timespec& b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

// ColorRGBA -> the GUI's BGR scalar. Alpha 0 means the sender had no preference,
// which HandSource signals with a negative component.
static cv::Scalar to_bgr(const std_msgs::msg::ColorRGBA& c) {
    if (c.a <= 0.f) return {-1, -1, -1};
    return {c.b * 255.0, c.g * 255.0, c.r * 255.0};
}

// One message per inference, so an empty hands[] is a positive "no hands this
// frame" rather than silence.
//
// Landmarks and the bounding box are normalized against the frame they were
// found in, so a peer with a different camera resolution can scale them to its
// own window without knowing anything about ours.
static handtrack::msg::HandSet to_msg(const std::vector<Hand>& hands, cv::Size frame,
                                      const rclcpp::Time& stamp, const std::string& source_id,
                                      const std::string& label,
                                      const std_msgs::msg::ColorRGBA& color) {
    handtrack::msg::HandSet msg;
    msg.header.stamp = stamp;
    msg.source_id = source_id;
    msg.label = label;
    msg.color = color;
    msg.hands.reserve(hands.size());

    const double fw = frame.width, fh = frame.height;
    for (const auto& h : hands) {
        handtrack::msg::Hand out;
        for (size_t j = 0; j < h.pts.size(); ++j) {
            out.landmarks[j].x = h.pts[j].x / fw;
            out.landmarks[j].y = h.pts[j].y / fh;
        }
        const cv::Rect box = hand_bbox(h, frame);
        out.bbox = {static_cast<float>(box.x / fw), static_cast<float>(box.y / fh),
                    static_cast<float>(box.width / fw), static_cast<float>(box.height / fh)};
        out.score = h.score;
        out.handedness = h.handed;
        msg.hands.push_back(out);
    }
    return msg;
}

// The reverse, for a peer's hands: normalized back up into our own window, since
// that is the space the GUI draws in.
static std::vector<Hand> from_msg(const handtrack::msg::HandSet& msg, cv::Size frame) {
    std::vector<Hand> hands;
    hands.reserve(msg.hands.size());
    for (const auto& in : msg.hands) {
        Hand h;
        for (size_t j = 0; j < h.pts.size(); ++j)
            h.pts[j] = cv::Point2f(static_cast<float>(in.landmarks[j].x * frame.width),
                                   static_cast<float>(in.landmarks[j].y * frame.height));
        h.score = in.score;
        h.handed = in.handedness;
        hands.push_back(h);
    }
    return hands;
}

static void infer_worker(HandTrack* pipe, rclcpp::Node* node,
                         rclcpp::Publisher<handtrack::msg::HandSet>::SharedPtr pub,
                         HandtrackGui::HandsCallback on_hands, HandSource self,
                         std_msgs::msg::ColorRGBA color) {
    set_prio(INFER_PRIO);
    for (;;) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(g_in_mtx);
            g_in_cv.wait(lk, [] { return g_in_ready || !g_run.load(); });
            if (!g_run.load()) return;
            frame = std::move(g_in_frame);
            g_in_ready = false;
        }

        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        auto hands = pipe->detect(frame, 2);
        clock_gettime(CLOCK_MONOTONIC, &b);

        pub->publish(to_msg(hands, frame.size(), node->now(), self.id, self.label, color));

        // Our own hands are just another source as far as the GUI is concerned;
        // a peer's arrive through the same callback with a different id.
        float best = 0.f;
        for (const auto& h : hands) best = std::max(best, h.score);
        g_hand_count.store(static_cast<int>(hands.size()));
        g_best_score.store(best);
        on_hands(self, hands);

        g_infer_us.store(static_cast<long>(elapsed(a, b) * 1e6));
        ++g_infers;
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    set_prio(DISPLAY_PRIO);

    auto node = std::make_shared<rclcpp::Node>("handtrack_node");
    // CAM_UNIT still works as the default so the existing invocation keeps
    // running; --ros-args -p camera_unit:=N wins over it.
    const char* env = getenv("CAM_UNIT");
    const int unit = node->declare_parameter<int>(
        "camera_unit", env && *env ? atoi(env) : DEFAULT_CAM_UNIT);

    // Identity carried on every message, so a display can say whose hands it is
    // drawing. Defaults to the node's fully-qualified name, which is already
    // unique on a graph; override when running several nodes per machine.
    HandSource self;
    self.id = "Larry TODO change"
    self.local = true;
    self.label = "Workshop"
    self.color = to_bgr(color);

    auto pub = node->create_publisher<handtrack::msg::HandSet>(TOPIC, rclcpp::QoS(1));

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

    RCLCPP_INFO(node->get_logger(), "camera unit %d  %dx%d  window %dx%d", unit, cw, ch, DW, DH);
    if ((cw | ch) & 1) {
        RCLCPP_ERROR(node->get_logger(), "need even camera dimensions for NV12");
        return 1;
    }

    // Find the ML model from our ros2 directory
    std::string model_dir;
    try {
        model_dir = ament_index_cpp::get_package_share_directory("handtrack") + "/";
    } catch (const std::exception& e) {
        RCLCPP_WARN(node->get_logger(), "package share dir not found (%s), using ./models", e.what());
    }
    HandTrack pipe(model_dir + det_model_path(), model_dir + lm_model_path());
    RCLCPP_INFO(node->get_logger(), "hand pipeline ready; source '%s' on %s",
                self.id.c_str(), pub->get_topic_name());

    // The other half of the mesh. Created here rather than beside the publisher
    // because scaling a peer's normalized landmarks needs the window size, which
    // only exists once the GUI is open.
    const auto on_hands = gui.hands_callback();
    const cv::Size window(DW, DH);
    auto seen = std::make_shared<std::set<std::string>>();
    auto logger = node->get_logger();
    auto sub = node->create_subscription<handtrack::msg::HandSet>(
        TOPIC, rclcpp::QoS(10),
        [on_hands, window, self, seen, logger](const handtrack::msg::HandSet::SharedPtr msg) {
            // We got our own message ignore.
            if (msg->source_id == self.id) return;
            HandSource src;
            src.local = false;
            src.id = msg->source_id;
            src.label = msg->label.empty() ? msg->source_id : msg->label;
            src.color = to_bgr(msg->color);
            
            // Insert them into our seen list
            if (seen->insert(src.id).second) {
                RCLCPP_INFO(logger, "peer '%s' joined", src.label.c_str());
            }
            on_hands(src, from_msg(*msg, window));
        });

    std::thread worker(infer_worker, &pipe, node.get(), pub, on_hands, self, color);
    // GTK takes this thread at gui.run(), so the executor needs its own.
    std::thread executor([node] { rclcpp::spin(node); });

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    long frames = 0;
    cv::Mat disp;

    // Runs on the camera library's thread for every viewfinder frame. RpiCam has
    // already converted NV12 to BGR at camera resolution, and the window is opened
    // at that same size, so this is normally a copy rather than a rescale.
    auto on_frame = [&](const cv::Mat& bgr) {
        static thread_local bool prio_set = (set_prio(DISPLAY_PRIO), true);
        (void)prio_set;
        if (!g_run.load()) return;

        // Resized here rather than in the GUI because inference runs on this same
        // frame, and the landmarks it returns have to be in window coordinates
        // for every source's overlay to line up.
        if (bgr.cols == DW && bgr.rows == DH) bgr.copyTo(disp);
        else cv::resize(bgr, disp, {DW, DH});

        // Hand off a private copy; disp is reused on the next frame.
        cv::Mat next = disp.clone();
        {
            std::lock_guard<std::mutex> lk(g_in_mtx);
            g_in_frame = std::move(next);
            g_in_ready = true;
        }
        g_in_cv.notify_one();

        ++frames;
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double sec = elapsed(t0, now);
        // Show the best score alongside the count: a detection sitting just over
        // the threshold looks the same as a solid one in a bare hand count.
        std::string label = std::format("{:.1f} fps  ai {:.1f} fps ({} ms)  hands {} ({:.2f})",
                                        frames / sec, g_infers.load() / sec,
                                        g_infer_us.load() / 1000, g_hand_count.load(),
                                        g_best_score.load());
        // The label is drawn by GTK in widget coordinates, not baked into the
        // frame: at 320x240 capture a fixed-size cv::putText ran off the edge.
        gui.set_status(label);

        // The GUI draws every source's hands over this frame, ours included.
        gui.show_frame(disp);

        // Throttled: the status line is already on the window, and a node's
        // stdout usually ends up in a launch log nobody is watching live.
        if (frames % 150 == 0)
            RCLCPP_INFO(node->get_logger(), "%s", label.c_str());
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
    g_timeout_add(100, +[](gpointer d) -> gboolean {
        if (!g_run.load() || !rclcpp::ok()) {
            static_cast<HandtrackGui *>(d)->quit();
            return G_SOURCE_REMOVE;
        }
        return G_SOURCE_CONTINUE;
    }, &gui);

    gui.run();

    // Closing the window shuts the node down too, so either exit path converges
    // here: stop feeding the worker, then let the executor fall out of spin().
    g_run = false;
    cam.stop();             // no more callbacks after this returns
    g_in_cv.notify_all();   // g_run is already false; wake the worker to exit.
    worker.join();
    rclcpp::shutdown();     // returns immediately if Ctrl-C already did it
    executor.join();
    return 0;
}
