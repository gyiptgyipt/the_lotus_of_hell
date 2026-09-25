"""
Gymnasium environment for the intercept task. Talks to PX4 SITL + target_sim
through ros_bridge.TrainingBridge. Run PX4 SITL, the Gazebo world, your
target_sim node, and uav_control/offboard_control_node *before* starting
training -- this env does not launch the simulator itself.

About episode resets (read before training for real):
This env's reset() calls /uav_intercept/reset_episode (a std_srvs/Trigger) if
present, which is expected to respawn the target at a random start pose. It
does NOT teleport the interceptor back to a start position -- PX4 SITL/Gazebo
don't expose a clean ROS2 service for that out of the box. Two practical ways
to close this gap:
  1. Use the Gazebo Transport `/world/<world>/set_pose` service (via
     ros_gz_bridge or gz-transport Python bindings) to teleport both vehicles
     at the start of each episode.
  2. Keep episodes short and let the interceptor's start position drift
     across an episode block, periodically restarting the whole PX4+Gazebo
     stack (slower, but zero extra plumbing).
Until you wire in (1), this env works as a "soft reset": stats/step counters
reset, but the interceptor continues from wherever it physically is.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ros_bridge import BridgeHandle, OBS_DIM, ACTION_DIM

MAX_SPEED_MPS = 8.0
MAX_YAW_RATE_RPS = 1.0
CAPTURE_RADIUS_M = 1.0
MAX_RANGE_M = 60.0          # episode terminates as "lost" beyond this
CONTROL_HZ = 20.0
MAX_EPISODE_STEPS = 800     # 40s at 20Hz


class InterceptEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._bridge = BridgeHandle()
        self._dt = 1.0 / CONTROL_HZ
        self._step_count = 0
        self._prev_range = None

    def _wait_for_fresh_state(self, timeout_sec: float = 5.0) -> np.ndarray:
        start = time.time()
        while time.time() - start < timeout_sec:
            obs = self._bridge.node.get_observation()
            if obs[11] > 0.5:  # target_valid flag
                return obs
            time.sleep(0.05)
        # Fall back to whatever we have; downstream code treats a stale/invalid
        # target as "not visible" via the valid flag, so this is safe, just
        # likely to end the episode quickly via the reward's invalid-target case.
        return self._bridge.node.get_observation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._bridge.node.send_hold()
        self._bridge.node.call_reset_service()
        time.sleep(0.2)  # let target_sim's respawn propagate before we read state

        obs = self._wait_for_fresh_state()
        self._step_count = 0
        self._prev_range = float(obs[9] * 50.0)  # undo pos_norm to get meters
        return obs, {}

    def step(self, action: np.ndarray):
        self._bridge.node.send_action(action, MAX_SPEED_MPS, MAX_YAW_RATE_RPS)
        time.sleep(self._dt)

        obs = self._bridge.node.get_observation()
        raw = self._bridge.node.get_raw_state()
        self._step_count += 1

        terminated = False
        truncated = False
        reward = 0.0

        if not raw.valid:
            # Lost the target (or it was never valid this episode) -- small
            # per-step penalty, let the episode run to truncation rather than
            # ending immediately so the policy learns to re-acquire.
            reward -= 0.1
        else:
            range_m = raw.range_m
            closing = raw.closing_velocity

            reward -= range_m / MAX_RANGE_M            # distance shaping, in [~-1, 0]
            reward += 0.05 * np.clip(closing, -5.0, 5.0)  # reward closing speed, penalize opening
            reward -= 0.01 * float(np.sum(np.square(action)))  # small control-effort penalty

            if range_m <= CAPTURE_RADIUS_M:
                reward += 100.0
                terminated = True
            elif range_m >= MAX_RANGE_M:
                reward -= 20.0
                terminated = True

            self._prev_range = range_m

        if self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        if terminated or truncated:
            self._bridge.node.send_hold()

        return obs, float(reward), terminated, truncated, {"range_m": raw.range_m}

    def close(self):
        self._bridge.node.send_hold()
        self._bridge.shutdown()
