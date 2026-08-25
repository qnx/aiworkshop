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

// handtracking.hpp — native hand-landmark pipeline (TFLite + OpenCV).
#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// tflite::FlatBufferModel and tflite::Interpreter are aliases into TFLite's
// internal impl namespace, not classes, so they cannot be forward declared
// without hardcoding that detail. Include the headers instead.
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

struct Hand {
    std::array<cv::Point2f, 21> pts;   // landmark pixels in the input image
    float score = 0.f;
    // P(right hand); < 0.5 means left, and how far it sits from 0.5 is the
    // model's confidence in the call. See handtracking.cpp for the mirroring
    // caveat and the HAND_MIRROR knob.
    float handed = 0.5f;
};

// Model paths, relative to the working directory. Lite models only.
std::string det_model_path();
std::string lm_model_path();

/**
 * @brief Axis-aligned bounding box around a hand's landmarks.
 *
 * The 21 points sit inside the hand's outline, so a tight box clips fingertips
 * and the palm edge -- this pads by a fraction of the box itself, which keeps
 * the margin sane at any hand distance, then clips to @p frame since landmarks
 * can land outside it.
 */
cv::Rect hand_bbox(const Hand& h, cv::Size frame);

/// An oriented square crop region in source-image pixels.
struct HandRoi {
    cv::Point2f center;
    float angle;   // radians, same convention as the detection path
    float size;    // side length of the square
};

class HandTrack {
public:
    HandTrack(const std::string& det_path, const std::string& lm_path);
    // Detect hands in a BGR image; returns up to max_hands hands.
    std::vector<Hand> detect(const cv::Mat& bgr, int max_hands = 2);

    // Wall time of each stage in the last detect() call, for profiling. The
    // landmark figure is the total across all crops that were run.
    double last_det_ms = 0.0;
    double last_lm_ms = 0.0;
    int last_lm_runs = 0;
    // Whether the last detect() had to run the palm detector, or tracked the
    // previous frame's ROI instead.
    bool ran_detector = false;

private:
    /**
     * @brief Load one model and build its interpreter.
     *
     * XNNPack comes from BuiltinOpResolverWithXNNPACK rather than an explicit
     * delegate: tflite-runtime-dev ships no xnnpack_delegate.h, but the resolver
     * is exported by the library and wires the delegate up during the build.
     */
    bool load(const std::string& path, std::unique_ptr<tflite::FlatBufferModel>& model,
              std::unique_ptr<tflite::Interpreter>& interp);

    /// Crop the ROI, run the landmark model, map the points back to source pixels.
    bool run_landmarks(const cv::Mat& bgr, const HandRoi& roi, Hand& out);

    // Each FlatBufferModel owns its mapped file and must outlive the interpreter
    // built from it. Declared first so they are destroyed last.
    std::unique_ptr<tflite::FlatBufferModel> det_model_, lm_model_;
    std::unique_ptr<tflite::Interpreter> det_, lm_;

    std::vector<cv::Point2f> anchors_;   // SSD anchor centres, normalized

    // ROIs carried over from the previous frame. Non-empty means the last frame
    // produced confident landmarks, so the palm detector can be skipped.
    std::vector<HandRoi> tracked_;

    // Frames since the palm detector last ran, for the periodic sweep that finds
    // hands entering the frame while others are already tracked.
    int since_detect_ = 0;
};
