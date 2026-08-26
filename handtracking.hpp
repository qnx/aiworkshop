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
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

struct Hand {
  std::array<cv::Point2f, 21> pts;  // landmark pixels in the input image
  float score = 0.f;
  // P(right hand); < 0.5 means left, and how far it sits from 0.5 is the
  // model's confidence in the call. See handtracking.cpp for the mirroring
  // caveat and HandTrackConfig::mirror.
  float handed = 0.5f;
};

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
  float angle;  // radians, same convention as the detection path
  float size;   // side length of the square
};

/**
 * @brief Tuning knobs for the pipeline.
 *
 * handtrack_node declares each numeric and boolean knob as a ros2 parameter
 * under the same name, so they are set on the command line:
 *
 *   ros2 run handtrack handtrack_node --ros-args -p det_th:=0.8 -p mirror:=true
 *
 * The model paths are not parameters; they take the defaults below.
 *
 * Out-of-range values are rejected with a warning and fall back to the default,
 * so a typo on the command line degrades to the default behaviour rather than
 * to a pipeline that silently detects nothing.
 */
struct HandTrackConfig {
  // Two independent gates, and both matter for false positives.
  //
  // det_th filters the palm detector: how sure it is a hand is *there*. Raising
  // it suppresses hand-shaped background before any crop work happens, so it is
  // also the cheaper of the two.
  //
  // presence_th filters the landmark model's presence output: given this crop,
  // how sure it is the crop actually contains a hand. This is the one that
  // rejects a confident-but-wrong palm box, so it is the stronger
  // false-positive gate.
  //
  // Both default to 0.5 upstream. det_th is raised because palm-detector scores
  // for a real hand sit well above 0.9, so 0.7 costs nothing and drops weak
  // boxes early. presence_th stays at 0.5 and is a true probability -- see the
  // note on the presence head in run_landmarks().
  float det_th = 0.7f;       // (0, 1]
  float presence_th = 0.5f;  // (0, 1]

  // Scale of the crop derived from the previous frame's landmarks.
  // Landmark-derived ROIs are looser than detector-derived ones: the crop must
  // still contain the hand after it has moved for a frame. 2.0 here against 2.6
  // for the detector rect, which is already the wider box.
  float track_scale = 2.0f;  // [1, 5]

  // Skip the palm detector while a hand stays tracked. false runs the detector
  // on every frame.
  bool tracking = true;

  // Tracking alone can only ever follow the hands it already has: a hand
  // entering the frame is invisible until tracking drops. So when we are
  // holding fewer hands than the caller asked for, re-run the detector every N
  // frames to look for new ones. At 30 fps the default finds a new hand within
  // about half a second while costing one detector pass (~57 ms) per 15 frames,
  // i.e. ~4 ms/frame amortized. Once max_hands are tracked, the detector stops
  // running entirely.
  int redetect_every = 15;  // [1, 1000]

  // The model's handedness label assumes a *mirrored* (selfie) image. This
  // camera points outward and nothing in the pipeline flips the frame, so the
  // sense is inverted to report real-world handedness. Set this if your feed is
  // mirrored -- or simply if left and right come out backwards on your board;
  // this is one convention I cannot verify without a hand in front of the lens.
  bool mirror = false;

  // TFLite interpreter threads. Defaults to leaving a core for the display
  // thread rather than taking all four.
  int threads = 3;  // [1, 8]

  // Model files, resolved by the caller against the package share directory.
  //
  // The lite models run at 87.1 ms/frame on this target against 123.8 ms for
  // the full ones (1.4x), with the same presence score and landmarks within a
  // few pixels on the test image.
  std::string det_model = "models/palm_detection_lite.tflite";
  std::string lm_model = "models/hand_landmark_lite.tflite";
};

class HandTrack {
 public:
  explicit HandTrack(const HandTrackConfig& cfg = {});
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
   * @brief Load TFlite model
   */
  bool load(const std::string& path,
            std::unique_ptr<tflite::FlatBufferModel>& model,
            std::unique_ptr<tflite::Interpreter>& interp);

  /**
   * Run the landmark detection on a hand given ROI.
   *
   * @param bgr input image
   * @param roi The ROI where the hand should be cropped from
   * @param[out] out The output hand (only valid if return = true)
   *
   * @return true if the landmark succeeded.
   */
  bool run_landmarks(const cv::Mat& bgr, const HandRoi& roi, Hand& out);

  // Clamped to the documented ranges by the constructor.
  HandTrackConfig cfg_;

  // Each FlatBufferModel owns its mapped file and must outlive the interpreter
  // built from it. Declared first so they are destroyed last.
  std::unique_ptr<tflite::FlatBufferModel> det_model_, lm_model_;
  std::unique_ptr<tflite::Interpreter> det_, lm_;

  std::vector<cv::Point2f> anchors_;  // SSD anchor centres, normalized

  // ROIs carried over from the previous frame. Non-empty means the last frame
  // produced confident landmarks, so the palm detector can be skipped.
  std::vector<HandRoi> tracked_;

  // Frames since the palm detector last ran, for the periodic sweep that finds
  // hands entering the frame while others are already tracked.
  int since_detect_ = 0;
};
