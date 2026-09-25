# Two PX4 UAVs: camera-based follower

This package turns **UAV 2** into a vision-only interceptor. Both vehicles use
PX4's built-in `gz_x500_mono_cam` model. UAV 2 runs a drone-trained YOLO model
on its forward camera, estimates range from the detected X500 bounding-box
width, predicts UAV 1 motion from consecutive detections, and sends lead-pursuit
PX4 Offboard velocity setpoints. The controller does not
subscribe to UAV 1 pose, GPS, or telemetry.

It is intended for SITL first. Do not use it on real aircraft until you add a
reliable target detector, geofence, obstacle avoidance, loss-of-target action,
and independently test the failsafes.

## Prerequisites

* PX4 SITL built with Gazebo (`make px4_sitl` once)
* ROS 2, `px4_msgs`, Micro XRCE-DDS Agent, `ros_gz_image`, NumPy, and Ultralytics
* A YOLO `.pt` model trained to detect a class named `drone`, `uav`, or
  `quadcopter`. A standard COCO YOLO model is not sufficient: COCO has no drone
  category. Set `target_class_names` in `follower.yaml` if your model differs.
* Build the `ament_cmake` package once:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros_ws/kamikaze
colcon build --packages-select uav_vision_follow --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

## Install dependencies

From the workspace root, run the installer once:

```bash
./scripts/install.sh
source install/setup.bash
```

The installer supports ROS 2 Jazzy on Debian/Ubuntu. It installs the ROS image,
Gazebo bridge, OpenCV, NumPy, Ultralytics, and build dependencies, validates the
bundled drone detector, and builds this workspace. PX4 SITL and
`MicroXRCEAgent` must be installed separately.

## One-command tmuxinator startup

The project file starts both PX4 vehicles, the XRCE-DDS Agent, image bridge, and
vision follower. It builds this package first. With PX4 at `~/PX4-Autopilot` and
the standard camera topic, the only command you need is:

```bash
tmuxinator start -p ~/ros_ws/kamikaze/src/uav_vision_follow/tmuxinator/uav_vision_follow.yml
```

If PX4 is elsewhere or `gz topic -l | grep '/image$'` shows a different camera
topic, set the two values before starting:

```bash
export PX4_DIR=/path/to/PX4-Autopilot
export UAV_1_GZ_IMAGE_TOPIC=/world/baylands/model/x500_mono_cam_1/link/camera_link/sensor/imager/image
export UAV_2_GZ_IMAGE_TOPIC=/world/baylands/model/x500_mono_cam_2/link/camera_link/sensor/imager/image
export DRONE_YOLO_MODEL=$HOME/ros_ws/kamikaze/src/uav_vision_follow/models/drone_yolo.pt
tmuxinator start -p ~/ros_ws/kamikaze/src/uav_vision_follow/tmuxinator/uav_vision_follow.yml
```

The `sitl` tmux window contains the leader and follower PX4 consoles. The `ros`
window contains the DDS Agent, camera bridge, and vision controller. The short
delays in the latter panes give Gazebo time to create the follower camera.

## Start the two vehicles

Open four terminals. In each one, source your normal PX4 and ROS environments.
The first vehicle starts Gazebo in the Baylands world. The second uses standalone
mode to connect to that same Gazebo server. Replace `~/PX4-Autopilot` with your
actual PX4 checkout.

Terminal 1 — leader (UAV 1):

```bash
cd ~/PX4-Autopilot
PX4_SYS_AUTOSTART=4001 PX4_GZ_WORLD=baylands PX4_SIM_MODEL=gz_x500_mono_cam \
  ./build/px4_sitl_default/bin/px4 -i 1
```

Terminal 2 — camera follower (UAV 2), initially 8 m behind the leader:

```bash
cd ~/PX4-Autopilot
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=baylands PX4_SYS_AUTOSTART=4001 \
  PX4_GZ_MODEL_POSE="-8,0,0,0,0,0" \
  PX4_SIM_MODEL=gz_x500_mono_cam ./build/px4_sitl_default/bin/px4 -i 2
```

Terminal 3 — one XRCE agent serves both PX4 instances:

```bash
MicroXRCEAgent udp4 -p 8888
```

PX4 exposes the second instance as `/px4_2`; that matches the default package
configuration. Confirm it before flying:

```bash
ros2 topic list | grep '^/px4_2/'
gz topic -l | grep '/image$'
```

## Bridge the follower camera

The built-in `x500_mono_cam` camera sensor is named `imager`. The two model
instances publish these Gazebo topics:

```bash
gz topic -l | grep '/sensor/imager/image$'
```

Tmuxinator bridges them into these stable, per-UAV ROS topics:

```text
/uav_1/camera/image   # leader / PX4 instance 1
/uav_2/camera/image   # follower / PX4 instance 2
```

To bridge manually, run one process for each topic:

```bash
ros2 run ros_gz_image image_bridge \
  /world/baylands/model/x500_mono_cam_1/link/camera_link/sensor/imager/image \
  --ros-args -r /world/baylands/model/x500_mono_cam_1/link/camera_link/sensor/imager/image:=/uav_1/camera/image

ros2 run ros_gz_image image_bridge \
  /world/baylands/model/x500_mono_cam_2/link/camera_link/sensor/imager/image \
  --ros-args -r /world/baylands/model/x500_mono_cam_2/link/camera_link/sensor/imager/image:=/uav_2/camera/image
```

Verify that the image has a positive rate:

```bash
ros2 topic list | grep '^/uav_[12]/camera/image$'
ros2 topic hz /uav_1/camera/image
ros2 topic hz /uav_2/camera/image
```

To see either live stream, use separate terminals (or select the topic in the
GUI):

```bash
ros2 run rqt_image_view rqt_image_view /uav_1/camera/image
ros2 run rqt_image_view rqt_image_view /uav_2/camera/image
```

## Start following

Terminal 4:

```bash
ros2 launch uav_vision_follow vision_follow.launch.py \
  model_path:="$DRONE_YOLO_MODEL"
```

Set the model path before launch:

```bash
export DRONE_YOLO_MODEL=/absolute/path/to/drone_yolo.pt
```

The node first commands UAV 2 to hold 5 m above its local origin using the PX4
NED setpoint `z = -5.0`. After 50 position setpoints it requests Offboard mode
and arms UAV 2 automatically for SITL. Once Offboard is active, the controller
uses the camera image to follow UAV 1. Take off UAV 1 manually and move it
slowly; UAV 2 will then follow once the detector sees it. To disable automatic
arming, pass `auto_arm:=false`:

```bash
ros2 launch uav_vision_follow vision_follow.launch.py \
  model_path:="$DRONE_YOLO_MODEL" auto_arm:=false
```

Tune [follower.yaml](config/follower.yaml) for your camera, model, and intercept
radius. In
particular, its horizontal FOV, `target_size_m`, and `target_class_names` must
match your setup. If target detection disappears for 0.5 s, the commanded
velocity immediately becomes zero; PX4's own Offboard-loss failsafe remains the
ultimate safety mechanism.

The computer-vision input for UAV 2 is `/uav_2/camera/image` (`sensor_msgs/msg/Image`).
View what UAV 2 sees with:

```bash
ros2 run rqt_image_view rqt_image_view /uav_2/camera/image
```

To see the YOLO overlay with bounding boxes, class labels, and confidence
values, open the annotated topic:

```bash
ros2 run rqt_image_view rqt_image_view /uav_2/camera/detections
```

The controller also prints a `CV status` line every two seconds. `images` and
`inferences` should increase while the camera bridge is working; `detections`
increases only when the configured YOLO classes are found. `last_detection=never`
means the camera is being processed but no matching target is visible. Confirm
the bridge independently with:

```bash
ros2 topic hz /uav_2/camera/image
ros2 topic info -v /uav_2/camera/image
```

UAV 2 predicts target motion for up to `prediction_horizon_s` and stops at
`intercept_distance_m` (2 m by default). This is intentional: do not use an
autonomous collision as an interception mechanism.

## Why this differs from `make px4_sitl gz_x500_baylands`

That convenient make target launches a single plain X500. PX4 multi-vehicle
SITL instead starts one PX4 process per vehicle, each with its own `-i` instance
number, and every later vehicle uses `PX4_GZ_STANDALONE=1`. PX4 provides the
`gz_x500_mono_cam` model for the required follower camera. See the official
[PX4 multi-vehicle guide](https://docs.px4.io/main/en/sim_gazebo_gz/multi_vehicle_simulation)
and [Gazebo vehicle list](https://docs.px4.io/main/en/sim_gazebo_gz/vehicles).
