# UAV Interceptor -- Control + RL scaffold

Assumes you already have a working `target_sim` node. This adds the two
pieces that were missing: a PX4 offboard control interface, and an RL
guidance policy that drives it.

## Layout

```
src/
  uav_common_msg/        TargetState.msg -- the interface between
                          target_sim/perception and everything downstream
  uav_control/            C++ (rclcpp)
    offboard_control_node   PX4 arming/offboard handshake, converts
                             ~/cmd_vel (TwistStamped) -> TrajectorySetpoint
    rl_inference_node       Loads the exported ONNX policy, builds the
                             observation from TargetState + odometry, runs
                             inference, publishes ~/cmd_vel, with a watchdog
                             that holds position if the target/odometry goes
                             stale
  uav_vision_detect/       C++ (rclcpp)
    yolo_detector           ONNX Runtime + OpenCV: letterbox preprocess,
                             inference, decode, NMS -- no ROS types, testable
                             standalone
    vision_detect_node      Camera + CameraInfo + odometry in; runs
                             yolo_detector, estimates range via apparent
                             size, rotates the bearing into NED using own
                             attitude, publishes TargetState on
                             /uav_intercept/target_state_vision
training/
  uav_rl_training/         Python, NOT a ROS2 package -- run outside colcon
    ros_bridge.py           rclpy node used only during training, mirrors
                             rl_inference_node's topic contract exactly
    intercept_env.py        Gymnasium env wrapping ros_bridge
    train_ppo.py             SB3 PPO training loop
    export_onnx.py          SB3 checkpoint -> ONNX, with a numerical sanity check
  uav_vision_training/      Python, NOT a ROS2 package
    capture_dataset.py      Auto-labels a YOLO dataset by projecting
                             target_sim's ground-truth TargetState into the
                             camera frame -- no manual annotation
    split_dataset.py        train/val split + dataset.yaml
    train_yolo.py            ultralytics fine-tuning wrapper
    export_yolo_onnx.py     Checkpoint -> ONNX, with an output-shape check
                             against what yolo_detector.cpp expects
```

## Why the split

`offboard_control_node` is the only thing that talks to PX4. Both training
(`ros_bridge.py`, in Python) and deployed inference (`rl_inference_node`, in
C++) send it the exact same message (`geometry_msgs/TwistStamped` on
`/offboard_control/cmd_vel`), so the policy sees the same actuation dynamics
in both. Nothing about training changes when you swap in the real inference
node later.

## Build order

1. **Build and sanity check `offboard_control_node` alone.** Launch PX4 SITL
   + Gazebo, run just this node, and publish a manual `TwistStamped` from the
   CLI (`ros2 topic pub /offboard_control/cmd_vel ...`) to confirm the drone
   arms, goes offboard, and follows velocity commands. Get this rock solid
   before anything RL-related.
2. **Wire up `TargetState`.** Make target_sim (or a small bridge on top of
   it) publish `uav_common_msg/msg/TargetState` on
   `/uav_intercept/target_state`, using ground-truth relative position/
   velocity for now. This is the only integration point the RL code assumes.
3. **Train.** With PX4 SITL + Gazebo + target_sim + `offboard_control_node`
   all running:
   ```
   cd training/uav_rl_training
   pip install -r requirements.txt
   python3 train_ppo.py --timesteps 500000 --logdir runs/exp1
   ```
4. **Export.**
   ```
   python3 export_onnx.py --model runs/exp1/ppo_intercept_final.zip \
       --out ../../src/uav_control/models/policy.onnx
   ```
5. **Deploy.**
   ```
   colcon build --symlink-install
   source install/setup.bash
   ros2 launch uav_control rl_guidance.launch.py
   ```

## Vision detection

`uav_vision_detect` publishes the same `TargetState` message as ground truth
does, on a different topic (`/uav_intercept/target_state_vision`) so you can
run it alongside the ground-truth path and compare before switching over.

**Prerequisite this repo can't set up for you**: your interceptor's Gazebo
model needs a camera sensor publishing `sensor_msgs/Image` and
`sensor_msgs/CameraInfo` over ROS2 (via `ros_gz_bridge` or PX4's camera
plugin, depending on how your model is set up). `uav_vision_detect` and
`capture_dataset.py` both assume these topics already exist; neither creates
the camera.

You said you don't have a trained model yet, so the order is:

1. **Add the camera** to the interceptor's SDF/model, confirm
   `ros2 topic echo /camera/camera_info` shows sane intrinsics before
   anything else.
2. **Capture an auto-labeled dataset.** With PX4 SITL + Gazebo + target_sim
   running (ground truth flowing on `/uav_intercept/target_state`), fly the
   interceptor around -- manually, via a scripted pattern, or your RL
   policy -- so the camera sees the target from varied angles/ranges:
   ```
   cd training/uav_vision_training
   pip install -r requirements.txt
   python3 capture_dataset.py --out dataset_raw --target-size-m 0.35
   # Ctrl+C when you've got a few thousand frames
   python3 split_dataset.py --src dataset_raw --dst dataset
   ```
   `--target-size-m` must match `uav_vision_detect`'s `target_size_m` param
   later -- it's the same apparent-size assumption used in both directions.
3. **Train and export**:
   ```
   python3 train_yolo.py --data dataset/dataset.yaml --epochs 100
   python3 export_yolo_onnx.py --weights runs_yolo/target_drone/weights/best.pt \
       --out ../../src/uav_vision_detect/models/target_yolo.onnx
   ```
4. **Run detection** and sanity check against ground truth before trusting it:
   ```
   ros2 launch uav_vision_detect vision_detect.launch.py
   ros2 topic echo /uav_intercept/target_state_vision
   # compare range/relative_position against /uav_intercept/target_state
   ```
5. **Switch the RL pipeline over** by remapping at launch instead of editing
   code:
   ```
   ros2 launch uav_control rl_guidance.launch.py \
       --ros-args -r rl_inference_node:/uav_intercept/target_state:=/uav_intercept/target_state_vision
   ```
   Expect the policy to need re-training or at least fine-tuning once it's
   seeing vision noise instead of ground truth -- that's the sim-to-"noisy
   estimate" gap mentioned back in the original build order.

**Known limitations, by design, not oversight:**
- Range comes from apparent size (`target_size_m` + bbox size), which is
  noisy and breaks if the target is partially occluded or viewed edge-on.
  A depth camera would give a cleaner range signal if accuracy turns out to
  matter more than simplicity.
- Only the single highest-confidence detection is used -- no multi-target
  handling.
- The camera-mount math (`apply_camera_mount` in `vision_detect_node.cpp`,
  and its inverse in `capture_dataset.py`) assumes a forward-facing,
  zero-offset camera by default. If yours is mounted differently, set
  `mount_q_wxyz_offset` in `config/params.yaml` -- and update the inverse in
  `capture_dataset.py` to match, or your auto-labels will be wrong.

## Things you'll need to adapt

- **`ONNXRUNTIME_ROOTDIR`**: `uav_control/CMakeLists.txt` looks for ONNX
  Runtime headers/libs under `/usr/local` by default. Download a release
  from the onnxruntime repo and pass
  `-DONNXRUNTIME_ROOTDIR=/path/to/it` to `colcon build --cmake-args`, or
  install it under `/usr/local` directly.
- **Topic names**: `rl_inference_node` and `ros_bridge.py` both assume
  target_sim publishes on `/uav_intercept/target_state`. Remap or edit if
  yours differs.
- **Episode resets during training** (see `intercept_env.py` docstring):
  this scaffold does a "soft reset" -- it does not teleport the interceptor
  back to a start pose between episodes, because that needs a Gazebo-level
  service call this scaffold doesn't wire up yet. Fine for getting the loop
  running; add a `/world/<world>/set_pose` call (via `ros_gz_bridge` or
  gz-transport Python bindings) before serious training runs, or your
  episodes will start from wherever the drone physically ended up.
- **No classical fallback yet**: `rl_inference_node`'s watchdog currently
  just holds position if the target/odometry go stale, rather than falling
  back to a PNG-style controller like the reference repo does. Worth adding
  a `uav_png_guidance` package later as both a safety net and a BC teacher
  if you want to pretrain before PPO.
- **Reward shaping** in `intercept_env.py` is a reasonable starting point,
  not a tuned one -- expect the "hover near, don't commit" failure mode
  early on; the `ent_coef` and closing-velocity term in `train_ppo.py` are
  there to fight exactly that.
