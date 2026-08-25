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

// handtracking.cpp — native hand-landmark pipeline on raw TFLite + OpenCV.
//
// Two stage: palm_detection_lite.tflite (192, SSD-anchor palm detection) ->
// rotated ROI -> hand_landmark_lite.tflite (224) -> 21 landmarks in image space.
#include "handtracking.hpp"

#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>

namespace {

constexpr int   DET_SIZE = 192;
constexpr int   LM_SIZE  = 224;
constexpr float NMS_TH   = 0.3f;
constexpr float ROI_SCALE = 2.6f;
constexpr float ROI_SHIFT_Y = -0.5f;

// Two independent gates, and both matter for false positives.
//
// DET_TH filters the palm detector: how sure it is a hand is *there*. Raising it
// suppresses hand-shaped background before any crop work happens, so it is also
// the cheaper of the two.
//
// PRESENCE_TH filters the landmark model's presence output: given this crop,
// how sure it is the crop actually contains a hand. This is the one that rejects
// a confident-but-wrong palm box, so it is the stronger false-positive gate.
//
// Both default to 0.5 upstream. DET_TH is raised because palm-detector scores for
// a real hand sit well above 0.9, so 0.7 costs nothing and drops weak boxes early.
// PRESENCE_TH stays at 0.5 and is a true probability -- see the note on the
// presence head in run_landmarks(). Tune at runtime rather than guessing:
//   HAND_DET_TH=0.8 HAND_PRESENCE_TH=0.7 ./handtrack
float env_th(const char* name, float dflt) {
    const char* v = getenv(name);
    if (!v || !*v) return dflt;
    float f = std::strtof(v, nullptr);
    if (f <= 0.f || f > 1.f) {
        fprintf(stderr, "warn: %s=%s out of range (0,1], using %.2f\n", name, v, dflt);
        return dflt;
    }
    return f;
}

const float DET_TH      = env_th("HAND_DET_TH", 0.7f);
const float PRESENCE_TH = env_th("HAND_PRESENCE_TH", 0.5f);

// Landmark-derived ROIs are looser than detector-derived ones: the crop must
// still contain the hand after it has moved for a frame. 2.0 here against 2.6 for
// the detector rect, which is already the wider box.
const float TRACK_ROI_SCALE = [] {
    const char* v = getenv("HAND_TRACK_SCALE");
    if (!v || !*v) return 2.0f;
    float f = std::strtof(v, nullptr);
    return (f >= 1.0f && f <= 5.0f) ? f : 2.0f;
}();

// Skip the palm detector while a hand stays tracked. Set HAND_TRACKING=0 to run
// detection on every frame (the original behaviour) for comparison.
const bool TRACKING = [] {
    const char* v = getenv("HAND_TRACKING");
    return !(v && v[0] == '0');
}();

// Tracking alone can only ever follow the hands it already has: a hand entering
// the frame is invisible until tracking drops. So when we are holding fewer hands
// than the caller asked for, re-run the detector every N frames to look for new
// ones. At 30 fps the default finds a new hand within about half a second while
// costing one detector pass (~57 ms) per 15 frames, i.e. ~4 ms/frame amortized.
// Once max_hands are tracked, the detector stops running entirely.
const int REDETECT_EVERY = [] {
    const char* v = getenv("HAND_REDETECT");
    int n = (v && *v) ? atoi(v) : 15;
    return (n >= 1 && n <= 1000) ? n : 15;
}();

// The model's handedness label assumes a *mirrored* (selfie) image. This camera
// points outward and nothing in the pipeline flips the frame, so the sense is
// inverted here to report real-world handedness. Set HAND_MIRROR=1 if your feed
// is mirrored -- or simply if left and right come out backwards on your board;
// this is one convention I cannot verify without a hand in front of the lens.
const bool MIRRORED = [] {
    const char* v = getenv("HAND_MIRROR");
    return v && v[0] == '1';
}();

std::vector<cv::Point2f> gen_anchors() {
    const int strides[4] = {8, 16, 16, 16};
    const int num_layers = 4;
    std::vector<cv::Point2f> a;
    int layer = 0;
    while (layer < num_layers) {
        int last = layer, per = 0;
        while (last < num_layers && strides[last] == strides[layer]) { per += 2; ++last; }
        int stride = strides[layer];
        int fm = (DET_SIZE + stride - 1) / stride;
        for (int y = 0; y < fm; ++y)
            for (int x = 0; x < fm; ++x) {
                float xc = (x + 0.5f) / fm, yc = (y + 0.5f) / fm;
                for (int k = 0; k < per; ++k) a.push_back({xc, yc});
            }
        layer = last;
    }
    return a;
}

double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

inline float sigmoid(float x) {
    if (x < -100.f) x = -100.f;
    return 1.f / (1.f + std::exp(-x));
}

// Turn the landmark model's handedness output into P(right hand).
//
// That head is already activated in this export (the conversion names it
// activation_handedness), so the value arrives as a probability and must not be
// pushed through sigmoid a second time -- that would squash every value into
// [0.5, 0.73] and make every hand read the same way. A value outside [0,1] can
// only be a logit, so activate that one.
//
// The model's raw value is P(left) under its mirrored-image convention, which
// makes it P(right) for this un-mirrored feed. See MIRRORED above.
float handedness_right(float raw) {
    float p = (raw >= 0.f && raw <= 1.f) ? raw : sigmoid(raw);
    return MIRRORED ? 1.f - p : p;
}

struct Det {
    float score, cx, cy, w, h;
    float kx[7], ky[7];   // keypoints in letterbox-normalized [0,1]
};

float iou(const Det& a, const Det& b) {
    float ax0 = a.cx - a.w / 2, ay0 = a.cy - a.h / 2, ax1 = a.cx + a.w / 2, ay1 = a.cy + a.h / 2;
    float bx0 = b.cx - b.w / 2, by0 = b.cy - b.h / 2, bx1 = b.cx + b.w / 2, by1 = b.cy + b.h / 2;
    float ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    float ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    float iw = std::max(0.f, ix1 - ix0), ih = std::max(0.f, iy1 - iy0);
    float inter = iw * ih;
    float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0.f;
}

}  // namespace

// Leave a core for the display thread rather than taking all four.
const int TFLITE_THREADS = [] {
    const char* v = getenv("HAND_THREADS");
    int n = (v && *v) ? atoi(v) : 3;
    return (n >= 1 && n <= 8) ? n : 3;
}();

// The next frame's crop is derived from the current landmarks instead of
// re-running the detector: hands move little between frames at video rate. The
// rect is the landmark bounding box measured in the hand's own rotated frame,
// squared off and expanded.
HandRoi roi_from_landmarks(const Hand& h) {
    const cv::Point2f& wrist = h.pts[0];
    const cv::Point2f& mid_mcp = h.pts[9];
    float angle = CV_PI / 2.f - std::atan2(-(mid_mcp.y - wrist.y), mid_mcp.x - wrist.x);

    // Measure the bounding box in the rotated frame so the box tracks the hand's
    // orientation rather than the image axes.
    float ca = std::cos(-angle), sa = std::sin(-angle);
    float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
    for (const auto& p : h.pts) {
        float rx = p.x * ca - p.y * sa;
        float ry = p.x * sa + p.y * ca;
        minx = std::min(minx, rx); maxx = std::max(maxx, rx);
        miny = std::min(miny, ry); maxy = std::max(maxy, ry);
    }
    float cx_rot = (minx + maxx) / 2.f, cy_rot = (miny + maxy) / 2.f;

    // Rotate the centre back into image space.
    float cb = std::cos(angle), sb = std::sin(angle);
    HandRoi r;
    r.center = cv::Point2f(cx_rot * cb - cy_rot * sb, cx_rot * sb + cy_rot * cb);
    r.angle = angle;
    r.size = std::max(maxx - minx, maxy - miny) * TRACK_ROI_SCALE;
    return r;
}

// Lite only: benchmarked on this target at 87.1 ms/frame vs 123.8 ms for the full
// models (1.4x), with the same presence score and landmarks within a few pixels
// on the test image. The full models are not worth their cost here.
std::string det_model_path() { return "models/palm_detection_lite.tflite"; }
std::string lm_model_path()  { return "models/hand_landmark_lite.tflite"; }

cv::Rect hand_bbox(const Hand& h, cv::Size frame) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const auto& p : h.pts) {
        x0 = std::min(x0, p.x); x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y); y1 = std::max(y1, p.y);
    }
    float m = 0.10f * std::max(x1 - x0, y1 - y0);
    cv::Rect r(cv::Point(std::lround(x0 - m), std::lround(y0 - m)),
               cv::Point(std::lround(x1 + m), std::lround(y1 + m)));
    return r & cv::Rect({0, 0}, frame);
}

bool HandTrack::load(const std::string& path,
                     std::unique_ptr<tflite::FlatBufferModel>& model,
                     std::unique_ptr<tflite::Interpreter>& interp) {
    model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!model) {
        fprintf(stderr, "error: cannot load model %s\n", path.c_str());
        return false;
    }
    tflite::ops::builtin::BuiltinOpResolverWithXNNPACK resolver;
    if (tflite::InterpreterBuilder(*model, resolver)(&interp, TFLITE_THREADS) != kTfLiteOk || !interp) {
        fprintf(stderr, "error: cannot build interpreter for %s\n", path.c_str());
        return false;
    }
    if (interp->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "error: AllocateTensors failed for %s\n", path.c_str());
        return false;
    }
    return true;
}

HandTrack::HandTrack(const std::string& det_path, const std::string& lm_path)
    : anchors_(gen_anchors()) {
    load(det_path, det_model_, det_);
    load(lm_path, lm_model_, lm_);
}


std::vector<Hand> HandTrack::detect(const cv::Mat& bgr, int max_hands) {
    if (!det_ || !lm_) return {};

    last_det_ms = 0.0;
    last_lm_ms = 0.0;
    last_lm_runs = 0;
    ran_detector = false;

    // Tracking can only follow hands it already holds. If we are short of
    // max_hands, sweep with the detector every REDETECT_EVERY frames so a hand
    // entering the frame gets picked up; once max_hands are tracked, no sweep is
    // needed and the detector never runs.
    ++since_detect_;
    const bool want_more = static_cast<int>(tracked_.size()) < max_hands;
    const bool sweep_due = want_more && since_detect_ >= REDETECT_EVERY;

    // Fast path: the previous frame produced confident landmarks, so crop from
    // those instead of paying for the palm detector. If every ROI comes back
    // below the presence threshold the hand was lost, and we fall through to a
    // full detection on this same frame rather than dropping a frame.
    if (TRACKING && !tracked_.empty() && !sweep_due) {
        std::vector<Hand> hands;
        for (const auto& roi : tracked_) {
            Hand h;
            if (run_landmarks(bgr, roi, h)) hands.push_back(h);
            if ((int)hands.size() >= max_hands) break;
        }
        if (!hands.empty()) {
            tracked_.clear();
            for (const auto& h : hands) tracked_.push_back(roi_from_landmarks(h));
            return hands;
        }
        tracked_.clear();
    }

    ran_detector = true;
    since_detect_ = 0;
    const int W = bgr.cols, H = bgr.rows;

    // ---- palm detection: letterbox to 192x192 RGB, normalized [0,1] ----
    float scale = std::min((float)DET_SIZE / W, (float)DET_SIZE / H);
    int nw = std::lround(W * scale), nh = std::lround(H * scale);
    int px = (DET_SIZE - nw) / 2, py = (DET_SIZE - nh) / 2;
    // Downscale before converting colour so BGR2RGB touches 192x192 rather than
    // the whole frame, then let convertTo fill the tensor with SIMD.
    cv::Mat resized;
    cv::resize(bgr, resized, {nw, nh});
    cv::Mat canvas(DET_SIZE, DET_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect(px, py, nw, nh)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    cv::Mat in_mat(DET_SIZE, DET_SIZE, CV_32FC3, det_->typed_input_tensor<float>(0));
    canvas.convertTo(in_mat, CV_32F, 1 / 255.0);
    double t_det0 = now_ms();
    det_->Invoke();
    last_det_ms = now_ms() - t_det0;
    last_lm_ms = 0.0;
    last_lm_runs = 0;

    const float* boxes = det_->typed_output_tensor<float>(0);    // [2016][18]
    const float* scores = det_->typed_output_tensor<float>(1);   // [2016]

    std::vector<Det> dets;
    for (size_t i = 0; i < anchors_.size(); ++i) {
        float s = sigmoid(scores[i]);
        if (s < DET_TH) continue;
        const float* b = boxes + i * 18;
        Det d;
        d.score = s;
        d.cx = b[0] / DET_SIZE + anchors_[i].x;
        d.cy = b[1] / DET_SIZE + anchors_[i].y;
        d.w = b[2] / DET_SIZE;
        d.h = b[3] / DET_SIZE;
        for (int k = 0; k < 7; ++k) {
            d.kx[k] = b[4 + k * 2] / DET_SIZE + anchors_[i].x;
            d.ky[k] = b[4 + k * 2 + 1] / DET_SIZE + anchors_[i].y;
        }
        dets.push_back(d);
    }

    // ---- NMS ----
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
    std::vector<Det> keep;
    for (const auto& d : dets) {
        bool ok = true;
        for (const auto& k : keep)
            if (iou(d, k) > NMS_TH) { ok = false; break; }
        if (ok) keep.push_back(d);
        if ((int)keep.size() >= max_hands) break;
    }

    // Map a letterbox-normalized point to source pixels.
    auto to_px = [&](float x, float y) {
        return cv::Point2f((x * DET_SIZE - px) / scale, (y * DET_SIZE - py) / scale);
    };

    std::vector<Hand> hands;
    for (const auto& d : keep) {
        cv::Point2f kp0 = to_px(d.kx[0], d.ky[0]);
        cv::Point2f kp2 = to_px(d.kx[2], d.ky[2]);
        cv::Point2f center = to_px(d.cx, d.cy);
        float box_w = d.w * DET_SIZE / scale;   // source px
        float box_h = d.h * DET_SIZE / scale;

        // rotation so wrist->middle-MCP points up (target 90deg).
        float angle = CV_PI / 2.f - std::atan2(-(kp2.y - kp0.y), kp2.x - kp0.x);

        // shift center along rotation, then square-long * scale.
        float long_side = std::max(box_w, box_h);
        center.x += long_side * (-ROI_SHIFT_Y * std::sin(angle));   // shift_x=0
        center.y += long_side * ( ROI_SHIFT_Y * std::cos(angle));
        float S = long_side * ROI_SCALE;

        Hand hand;
        if (run_landmarks(bgr, HandRoi{center, angle, S}, hand)) hands.push_back(hand);
    }

    // Seed tracking for the next frame.
    tracked_.clear();
    if (TRACKING)
        for (const auto& h : hands) tracked_.push_back(roi_from_landmarks(h));

    return hands;
}

// Crop the oriented ROI, run the landmark model, and map the 21 points back into
// source-image coordinates. Returns false if the crop did not contain a hand.
bool HandTrack::run_landmarks(const cv::Mat& bgr, const HandRoi& roi, Hand& out) {
    // Affine mapping source -> 224x224 crop (dst = M*[src;1]).
    float s224 = LM_SIZE / roi.size;
    float ca = std::cos(-roi.angle) * s224, sa = std::sin(-roi.angle) * s224;
    cv::Mat M = (cv::Mat_<double>(2, 3) <<
        ca, -sa, LM_SIZE / 2.0 - (ca * roi.center.x - sa * roi.center.y),
        sa,  ca, LM_SIZE / 2.0 - (sa * roi.center.x + ca * roi.center.y));
    cv::Mat crop;
    cv::warpAffine(bgr, crop, M, {LM_SIZE, LM_SIZE}, cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat crop_rgb;
    cv::cvtColor(crop, crop_rgb, cv::COLOR_BGR2RGB);
    cv::Mat lin_mat(LM_SIZE, LM_SIZE, CV_32FC3, lm_->typed_input_tensor<float>(0));
    crop_rgb.convertTo(lin_mat, CV_32F, 1 / 255.0);
    double t_lm0 = now_ms();
    lm_->Invoke();
    last_lm_ms += now_ms() - t_lm0;
    ++last_lm_runs;

    // Outputs, in the order the export declares them: landmarks, hand flag,
    // handedness, world landmarks. Older exports stop at two, so the handedness
    // head is only read when it is actually there.
    //
    // The hand flag is already a probability -- this export activates it (the
    // conversion names the head activation_handflag, and it reads 0.003-0.016 on
    // random-noise input, which no logit head would). Do not sigmoid it again:
    // that squashes the whole head into [0.5, 0.73], which is where the old
    // "a clean thumbs-up only scores 0.73" reading came from.
    const float* lms = lm_->typed_output_tensor<float>(0);   // [63]
    float presence = *lm_->typed_output_tensor<float>(1);
    if (presence < PRESENCE_TH) return false;

    cv::Mat Minv;
    cv::invertAffineTransform(M, Minv);
    out.score = presence;
    out.handed = lm_->outputs().size() > 2
                     ? handedness_right(*lm_->typed_output_tensor<float>(2))
                     : 0.5f;
    for (int j = 0; j < 21; ++j) {
        double cx = lms[j * 3], cy = lms[j * 3 + 1];   // in crop pixels [0,224]
        double sx = Minv.at<double>(0, 0) * cx + Minv.at<double>(0, 1) * cy + Minv.at<double>(0, 2);
        double sy = Minv.at<double>(1, 0) * cx + Minv.at<double>(1, 1) * cy + Minv.at<double>(1, 2);
        out.pts[j] = cv::Point2f((float)sx, (float)sy);
    }
    return true;
}
