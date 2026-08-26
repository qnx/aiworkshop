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
#include "rpi_cam.hpp"

#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <opencv2/imgproc.hpp>

RpiCam::RpiCam(camera_unit_t camera_id) : id_(camera_id) {}

int RpiCam::init() {
    int err = camera_open(id_, CAMERA_MODE_RO, &handle_);
    if (err != EOK) {
        fprintf(stderr, "camera_open(unit %d) failed: %s\n", static_cast<int>(id_), strerror(err));
        return err;
    }

    if (camera_has_feature(handle_, CAMERA_FEATURE_VFWINDOW) == false) {
        fprintf(stderr, "Viewfinder is not supported for this camera\n");
        return ENOTSUP;
    }

    camera_frametype_t frame_types[10];
    uint32_t frame_types_num = 0;
    camera_get_supported_vf_frame_types(handle_, 10, &frame_types_num, frame_types);

    // Prefer NV12 (half the bytes per pixel of YCbYCr), but take YCbYCr if that is
    // all the camera offers -- cam_cb() handles both.
    camera_frametype_t chosen = CAMERA_FRAMETYPE_UNSPECIFIED;
    for (uint32_t i = 0; i < frame_types_num; i++) {
        if (frame_types[i] == CAMERA_FRAMETYPE_NV12) { chosen = CAMERA_FRAMETYPE_NV12; break; }
        if (frame_types[i] == CAMERA_FRAMETYPE_YCBYCR) chosen = CAMERA_FRAMETYPE_YCBYCR;
    }
    if (chosen == CAMERA_FRAMETYPE_UNSPECIFIED) {
        fprintf(stderr, "Camera doesn't have supported frametype NV12 or YCBYCR (%u types offered)\n",
                frame_types_num);
        return ENOTSUP;
    }

    // Request the format we actually found, not an assumed one.
    camera_set_vf_property(handle_, CAMERA_IMGPROP_FORMAT, chosen);
    camera_set_vf_property(handle_, CAMERA_IMGPROP_CREATEWINDOW, 0);

    // Pick the smallest viewfinder resolution at or above 720p. Every stage
    // downstream is per-pixel -- the NV12->BGR conversion, the window blit, and
    // the letterbox/crop feeding two 192/224-input models -- so capturing at
    // sensor resolution only to throw pixels away is pure cost. The 720p floor
    // is for the operator, not the models: below it the window is too small to
    // read the overlay, and the models still oversample at 1280x720.
    constexpr long MIN_PIXELS = 1280 * 720;
    camera_res_t resolutions[32];
    uint32_t res_num = 0;
    if (camera_get_supported_vf_resolutions(handle_, 32, &res_num, resolutions) == EOK && res_num > 0) {
        auto px = [](const camera_res_t &r) {
            return static_cast<long>(r.width) * r.height;
        };
        uint32_t best = 0;
        for (uint32_t i = 1; i < res_num; i++) {
            const long a = px(resolutions[i]), b = px(resolutions[best]);
            const bool a_ok = a >= MIN_PIXELS, b_ok = b >= MIN_PIXELS;
            // Prefer 720p or better; among those the smallest. If nothing on
            // offer reaches 720p, take the largest rather than the smallest.
            if (a_ok != b_ok ? a_ok : (a_ok ? a < b : a > b)) best = i;
        }
        int rw = static_cast<int>(resolutions[best].width);
        int rh = static_cast<int>(resolutions[best].height);
        int err_res = camera_set_vf_property(handle_, CAMERA_IMGPROP_WIDTH, rw,
                                             CAMERA_IMGPROP_HEIGHT, rh);
        if (err_res != EOK) {
            fprintf(stderr, "warn: could not select %dx%d (%s), keeping camera default\n",
                    rw, rh, strerror(err_res));
        }
    } else {
        fprintf(stderr, "warn: could not enumerate viewfinder resolutions, keeping default\n");
    }

    // Read back what the camera actually settled on rather than assuming the
    // request was honoured.
    camera_get_vf_property(handle_, CAMERA_IMGPROP_WIDTH, &width_, CAMERA_IMGPROP_HEIGHT, &height_);
    printf("camera: %dx%d %s\n", width_, height_,
           chosen == CAMERA_FRAMETYPE_NV12 ? "NV12" : "YCbYCr");

    return EOK;
}

int RpiCam::start(FrameCb callback) {
    if (handle_ == CAMERA_HANDLE_INVALID) {
        return ENODEV;
    }
    cb_ = callback;

    return camera_start_viewfinder(handle_, RpiCam::cam_cb, NULL, this);
}

int RpiCam::stop() {
    if (handle_ == CAMERA_HANDLE_INVALID) {
        return EOK;
    }
    camera_stop_viewfinder(handle_);
    camera_close(handle_);
    handle_ = CAMERA_HANDLE_INVALID;
    return EOK;
}

std::pair<int, int> RpiCam::get_frame_size() const {
    return std::make_pair(width_, height_);
}

void RpiCam::cam_cb(camera_handle_t handle, camera_buffer_t *buf, void *data) {
    RpiCam *cam = reinterpret_cast<RpiCam *>(data);

    if (!cam->cb_) {
        fprintf(stderr, "No callback set for RpiCam\n");
        return;
    }

    // Create our output image based on our camera info
    cv::Mat brg_img(cam->height_, cam->width_, CV_8UC3);
    if (buf->frametype == CAMERA_FRAMETYPE_NV12) {
        /**
         * @brief NV12 uses a 4:2:0 YUV format with 2 planes.
         *
         * The first plane is the Y plane (or grayscale) which is a 1 to 1 mapping between the source and dst image
         *     this means we can just do a blind copy of the entire image
         * The second plan is the UV plane which contains 1 UV entry for every 2x2 kernel in the src image. This means
         *     that there is half has many lines and columns as the first plan (but the columns ends up being equal as
         * its a tuple of 2 values interlaced)
         *
         * The Kernel can be described with the following 4 pixels
         * Y    Y
         *   UV
         * Y    Y
         *
         * @note when creating the plans we provide the memory from camapi to avoid the double copy
         */
        cv::Mat Y_img(buf->framedesc.nv12.height, buf->framedesc.nv12.width, CV_8UC1, buf->framebuf);
        cv::Mat UV_img(buf->framedesc.nv12.height / 2, buf->framedesc.nv12.width / 2, CV_8UC2,
                       buf->framebuf + buf->framedesc.nv12.uv_offset);
        cv::cvtColorTwoPlane(Y_img, UV_img, brg_img, cv::COLOR_YUV2BGR_NV12);
    } else if (buf->frametype == CAMERA_FRAMETYPE_YCBYCR) {
        cv::Mat yuyv_img(buf->framedesc.ycbycr.height, buf->framedesc.ycbycr.width, CV_8UC2, buf->framebuf);
        cv::cvtColor(yuyv_img, brg_img, cv::COLOR_YUV2BGR_YUY2);
    } else {
        fprintf(stderr, "Unsupported Frametype %d\n", static_cast<int>(buf->frametype));
        return;
    }

    cam->cb_(brg_img);
}
