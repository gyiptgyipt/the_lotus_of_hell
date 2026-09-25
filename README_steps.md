# UAV Interceptor -- Full Setup & Run Guide

This covers every step from a bare machine to a trained, deployed RL
interceptor with vision-based detection. Follow it in order the first time
through -- each stage is meant to be validated before you move to the next,
because debugging PX4 + Gazebo + RL all at once is miserable, and debugging
them one at a time isn't.

---

## 0. Prerequisites

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 22.04 | matches ROS2 Humble target |
| ROS 2 | Humble | [install guide](https://docs.ros.org/en/humble/Installation.html) |
| Gazebo | Harmonic (gz-sim 8.x) | ships with recent PX4 |
| PX4 Firmware | v1.16 | `git clone https://github.com/PX4/PX4-Autopilot.git --recursive` |
| Micro XRCE-DDS Agent | v2.4.x | bridges ROS2 <-> PX4 uORB topics |
| OpenCV | 4.x | usually already present with ROS2 desktop install |
| cv_bridge | matching ROS2 distro | `sudo apt install ros-humble-cv-bridge` |
| ONNX Runtime | 1.18.x (C++ release) | manual download, see 0.3 below |
| Python | 3.10 | for training scripts, separate from the ROS2 C++ build |

### 0.1 ROS2 + PX4 + Gazebo

If you don't already have these:

```bash
# ROS2 Humble desktop (skip if already installed)
sudo apt update && sudo apt install ros-humble-desktop

# PX4 Autopilot
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh
cd ~/PX4-Autopilot && git checkout v1.16.0 && git submodule update --init --recursive

# Micro XRCE-DDS Agent
git clone -b v2.4.2 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git ~/Micro-XRCE-DDS-Agent
cd ~/Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install && sudo ldconfig
```

### 0.2 ROS2 packages this workspace needs

```bash
sudo apt install ros-humble-cv-bridge ros-humble-image-transport \
  python3-colcon-common-extensions python3-rosdep
```

You also need `px4_msgs` and `px4_ros_com` in your workspace's `src/` --
clone them from the PX4 org if you don't already have them from your
existing setup:

```bash
cd ~/uav_intercept_ws/src
git clone https://github.com/PX4/px4_msgs.git -b release/1.16
git clone https://github.com/PX4/px4_ros_com.git -b release/1.16
```

### 0.3 ONNX Runtime (C++)

Both `uav_control` (RL policy inference) and `uav_vision_detect` (YOLO
inference) link against the ONNX Runtime C++ library directly, not through
pip.

```bash
cd /opt
sudo wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
sudo tar xzf onnxruntime-linux-x64-1.18.0.tgz
# Note the path -- you'll pass it as ONNXRUNTIME_ROOTDIR when building
```

### 0.4 Python environment for training

Training (RL and YOLO) happens outside colcon, in plain Python, but needs
`rclpy` and `cv_bridge` importable so the training bridge can talk to ROS2.
Easiest path: use the system Python that ROS2 Humble already set up, or a
venv with `--system-site-packages`.

```bash
python3 -m venv ~/uav_train_venv --system-site-packages
source ~/uav_train_venv/bin/activate
pip install -r ~/uav_intercept_ws/training/uav_rl_training/requirements.txt
pip install -r ~/uav_intercept_ws/training/uav_vision_training/requirements.txt
```

---

## 1. Workspace layout

```
uav_intercept_ws/
  src/
    uav_common_msg/        TargetState.msg -- shared interface
    uav_control/            offboard_control_node (PX4 interface)
                             rl_inference_node (ONNX RL policy)
    uav_vision_detect/       uav_vision_detect (ONNX YOLO detection)
    <your target_sim package>
    px4_msgs/                (cloned in step 0.2)
    px4_ros_com/              (cloned in step 0.2)
  training/
    uav_rl_training/          PPO training + ONNX export (Python, standalone)
    uav_vision_training/      Dataset capture + YOLO training + export (Python, standalone)
```

Drop your existing `target_sim` package into `src/` alongside the others if
it isn't there already.

## 2. Build

```bash
cd ~/uav_intercept_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args \
  -DONNXRUNTIME_ROOTDIR=/opt/onnxruntime-linux-x64-1.18.0
source install/setup.bash
```

If this fails on `uav_control` or `uav_vision_detect` specifically, it's
almost always the ONNX Runtime path -- double check
`-DONNXRUNTIME_ROOTDIR` points at the directory containing `include/` and
`lib/libonnxruntime.so`.

---

## 3. Validate the control layer alone (no RL, no vision yet)

Don't skip this -- offboard mode is finicky and you want to know it works
before RL or vision are in the loop to confuse debugging.

**Terminal 1 -- start the sim:**
```bash
cd ~/PX4-Autopilot
./run_swarm.sh    # or: make px4_sitl gz_x500 if you're only flying one vehicle
```

**Terminal 2 -- the DDS bridge:**
```bash
MicroXRCEAgent udp4 -p 8888
```

**Terminal 3 -- offboard control node:**
```bash
source ~/uav_intercept_ws/install/setup.bash
ros2 run uav_control offboard_control_node
```

Watch it arm and climb to `takeoff_altitude_m` (default 3m) using its
built-in warmup climb. Then, **Terminal 4**, command it manually:

```bash
ros2 topic pub /offboard_control/cmd_vel geometry_msgs/msg/TwistStamped \
  "{twist: {linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {z: 0.0}}}" -r 20
```

You should see the drone move forward at 1 m/s in Gazebo. Stop the publisher
(Ctrl+C) and confirm it holds position (the node's watchdog zeroes velocity
after 0.5s with no fresh command) rather than drifting on the last command.

**If it won't arm / rejects offboard mode:** almost always means setpoints
weren't streaming fast enough before the mode switch was requested, or
QGroundControl (if running) is fighting for control -- close QGC or put it
in a mode that doesn't contest offboard.

---

## 4. Wire up TargetState (ground truth)

Make your `target_sim` node publish `uav_common_msg/msg/TargetState` on
`/uav_intercept/target_state`. Ground-truth values (you already have the
target's simulated position/velocity, so this is just repackaging):

- `relative_position` = target_position - interceptor_position, NED, meters
- `relative_velocity` = target_velocity - interceptor_velocity, NED, m/s
- `range` = norm(relative_position)
- `closing_velocity` = -(relative_position . relative_velocity) / range
- `valid` = true whenever the target exists in the sim (always, for ground truth)

You'll need the interceptor's own position/velocity too, which you can get
by subscribing to `/fmu/out/vehicle_odometry` inside target_sim, or by
computing the relative state directly from Gazebo's ground-truth model
poses if target_sim already has access to both.

**Check it:**
```bash
ros2 topic echo /uav_intercept/target_state
```
Confirm `range` and `relative_position` look sane as the target moves.

---

## 5. Train the RL policy against ground truth

With PX4 SITL + Gazebo + target_sim + `offboard_control_node` all running
(same as sections 3 and 4, combined):

```bash
source ~/uav_train_venv/bin/activate
cd ~/uav_intercept_ws/training/uav_rl_training
python3 train_ppo.py --timesteps 500000 --logdir runs/exp1
```

Watch `runs/exp1` with TensorBoard (`tensorboard --logdir runs/exp1`) --
expect the reward to be noisy early and the classic "hovers near the
target without committing" failure mode before it learns to close the last
few meters. If it never breaks out of that, see the note in
`intercept_env.py` about `ent_coef` and the closing-velocity reward term.

Read `intercept_env.py`'s docstring on episode resets before a long
training run -- the default is a "soft reset" that does not teleport the
interceptor back to a start pose between episodes.

**Export once training looks reasonable:**
```bash
python3 export_onnx.py --model runs/exp1/ppo_intercept_final.zip \
    --out ../../src/uav_control/models/policy.onnx
```

**Deploy and test against ground truth:**
```bash
cd ~/uav_intercept_ws && colcon build --symlink-install --packages-select uav_control
source install/setup.bash
ros2 launch uav_control rl_guidance.launch.py
```
Watch `/uav_intercept/target_state` range converge as the interceptor
closes in.

---

## 6. Add a camera to the interceptor model

This step happens entirely in your Gazebo model files, not in this
workspace -- add a camera sensor to the interceptor's SDF/xacro and bridge
it to ROS2 (`ros_gz_bridge` or PX4's camera plugin, depending on your
setup) so that these exist:

```bash
ros2 topic echo /camera/camera_info   # confirm sane fx, fy, cx, cy
ros2 topic hz /camera/image_raw       # confirm it's actually publishing
```

Don't proceed until both of these look right -- everything from here on
assumes them.

---

## 7. Capture an auto-labeled vision dataset

With PX4 SITL + Gazebo + target_sim running (ground truth flowing on
`/uav_intercept/target_state`, as in section 4) and the camera confirmed
working (section 6), fly the interceptor around -- manually via the
`cmd_vel` publish trick from section 3, or by re-running the RL policy from
section 5 -- so the camera sees the target from varied angles and ranges:

```bash
source ~/uav_train_venv/bin/activate
cd ~/uav_intercept_ws/training/uav_vision_training
python3 capture_dataset.py --out dataset_raw --target-size-m 0.35
# let it run until you have a few thousand saved frames, then Ctrl+C
```

`--target-size-m` is the target drone's characteristic physical size (e.g.
wingspan/diagonal) -- keep this number consistent with `uav_vision_detect`'s
`target_size_m` param later; both directions use the same apparent-size
assumption.

```bash
python3 split_dataset.py --src dataset_raw --dst dataset
```

---

## 8. Train and export the YOLO detector

```bash
python3 train_yolo.py --data dataset/dataset.yaml --epochs 100
python3 export_yolo_onnx.py \
  --weights runs_yolo/target_drone/weights/best.pt \
  --out ../../src/uav_vision_detect/models/target_yolo.onnx
```

The export script checks the ONNX output shape is `[1, 5, N]` (single
class, no built-in NMS) and warns loudly if it isn't -- don't ignore that
warning, `yolo_detector.cpp`'s decoder assumes that exact layout.

---

## 9. Run detection and sanity check it

```bash
cd ~/uav_intercept_ws && colcon build --symlink-install --packages-select uav_vision_detect
source install/setup.bash
ros2 launch uav_vision_detect vision_detect.launch.py
```

`uav_vision_detect` publishes to `/uav_intercept/target_state_vision` (a
different topic from ground truth on purpose) so you can compare them side
by side before trusting it:

```bash
ros2 topic echo /uav_intercept/target_state_vision
# compare range / relative_position against:
ros2 topic echo /uav_intercept/target_state
```

Expect the vision range estimate to be noisier than ground truth, especially
at long range or when the target is viewed edge-on -- that's the apparent-
size method's known limitation, not a bug to chase down.

---

## 10. Switch the RL pipeline to vision

Remap at launch rather than editing code:

```bash
ros2 launch uav_control rl_guidance.launch.py \
    --ros-args -r /uav_intercept/target_state:=/uav_intercept/target_state_vision
```

The policy was trained on clean ground truth, so expect degraded
performance at first -- this is the real sim-to-"noisy estimate" gap. If it
doesn't hold up:
- Fine-tune the existing PPO checkpoint with vision noise injected into
  `intercept_env.py`'s observations (easiest first step -- add Gaussian
  noise matching what you observe vision producing vs. ground truth in
  section 9, before doing a full vision-in-the-loop retrain), or
- Retrain from scratch with `uav_vision_detect` running live instead of
  ground truth feeding the training bridge (slower, but no train/deploy
  mismatch).

---

## 11. Full system checklist

Everything running together, in order:
1. PX4 SITL + Gazebo (`run_swarm.sh`)
2. `MicroXRCEAgent udp4 -p 8888`
3. your `target_sim` node
4. `uav_vision_detect` (if using vision) or confirm target_sim's ground
   truth is flowing (if not yet switching to vision)
5. `uav_control`'s `rl_guidance.launch.py` (brings up both
   `offboard_control_node` and `rl_inference_node` together)

## Known limitations (by design, not oversight)

- **Episode resets during training** are "soft" -- no Gazebo-level teleport
  of the interceptor between episodes. See `intercept_env.py`'s docstring
  for how to add a `/world/<world>/set_pose` call.
- **No classical fallback guidance** (e.g. PNG) wired into the RL watchdog
  yet -- it holds position on a stale/invalid target rather than falling
  back to a hand-coded controller.
- **Vision range** comes from apparent size, which degrades with occlusion
  or edge-on viewing angles. A depth camera would give a cleaner signal if
  this turns out to matter.
- **Single-target only** -- both the RL policy and the detector assume one
  target drone in the scene.
