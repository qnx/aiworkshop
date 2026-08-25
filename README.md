# TFLite Handtrack

Native QNX hand-landmark tracking with sensor framework and TFLite.

## Prerequisite
### Build and Runtime Dependencies
In order to build the project a few dependencies need to be installed in order to support building on the target. in order to install dependencies run the provided `configure.sh` script
```bash
./configure.sh
```

### Camera
> For RPI5 QSTI starting in 8.0.5 multiple camera unit have been exposed in a single running Sensor Framework instance making this step unneeded
A camera must be configured through sensor framework for this demo.

The demo has been verified with the RPI Camera Module 3 and Logitech C270 USB camera

#### QSTI 8.0.5 and on
Depending on which camera you are using when running you will need to specific a different unit where the numbers have been provided below

##### RPI5

##### RPI4

#### QSTI 8.0.4 and earlier
The camera can be selected
```bash
sudo vim /etc/startup/post_startup.sh
```

Commenting any sensor commands and uncommenting USB Camera (or module 3 depending on your camera)
```
sensor -U 521:521 -r /data/share/sensor -c /system/etc/config/sensor/usb_camera.conf
```
> NOTE: in order for this change to take affect you must reboot

### Models

MediaPipe's hand landmarking lite models (palm_detection_lite.tflite and hand_landmark_lite.tflite) are used for landmark detection. These will automatically be downloaded during the build process.

Detection is gated twice, and both gates are tuned with environment variables rather than CLI options:

```sh
HAND_DET_TH=0.8 ./handtrack        # palm-detector confidence (default 0.7)
HAND_PRESENCE_TH=0.6 ./handtrack   # landmark presence (default 0.5)
```

`HAND_DET_TH` drops weak palm boxes before any crop work. `HAND_PRESENCE_TH` is the stronger gate: it rejects a confident-but-wrong box once the landmark model has looked at the crop. Both are true probabilities -- a clean hand scores ~0.99.

## Overlay

Each hand gets a translucent rounded box tagged `label Right 0.99` -- who it belongs to, the handedness call, then the presence score, so a marginal detection is visible per hand and not just in the status line.

## Build
> Before running the build script ./configure.sh should be run to install all dependencies
A build script has been provided to make building easier but really all it is doing is sourcing the installed ROS2 install and running `colcon build`. For more information about how the build works see `build.sh`
```bash
./build.sh
```

## Running the demo
> When running on the RPI5 you may need to specify the camera unit when running.
```sh
./start -p camera_unit:=1
```
