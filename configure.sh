#!/bin/bash

#
# This script can be used during workshops to lower the amount of network traffic needed.
# It will search /boot for /boot/aiworkshop_artifacts/[models|apks] and install any included
# packages as well as copy the models to the current working directory
# If the APKs are not found they will instead be installed though the `apk` tool
# Models on the other hand are downloaded during the build step and can be safely skipped.
#

LOCAL_ARTIFACTS=${LOCAL_ARTIFACTS:-/boot/aiworkshop}
LOCAL_APK_PATH=${LOCAL_APK_PATH:-${LOCAL_ARTIFACTS}/apks}
LOCAL_MODELS_PATH=${LOCAL_MODELS_PATH:-${LOCAL_ARTIFACTS}/models}

# Check for apks
if [ -d "${LOCAL_APK_PATH}" ]; then
    echo "Installing Local APKS from ${LOCAL_APK_PATH}"
    apk add /boot/aiworkshop/apks/*
else
    echo "Installing apk dependencies using network"
    apk add opencv-dev tflite-runtime-dev tflite-runtime flatbuffers-dev gtk4-dev ros2-jazzy python3-dev python3-numpy python3-numpy-dev qnx-sensor-framework-dev
fi

if [ -d "${LOCAL_MODELS_PATH}" ]; then
    echo "Found Local TFLite Models from ${LOCAL_MODELS_PATH} copying"
    cp -r "${LOCAL_MODELS_PATH}" ./models
else
    echo "TFLite Models will be installed during build"
fi
