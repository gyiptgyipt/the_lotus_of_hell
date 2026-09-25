"""
ROS2 bridge for training. Runs an rclpy node on a background thread and
exposes plain thread-safe Python methods (get_observation / send_action /
call_reset_service) so the Gym env in intercept_env.py doesn't need to know
anything about ROS2 directly.

Topic/message contract must match uav_control/src/rl_inference_node.cpp
exactly, since that's the node this bridge is standing in for during
training:
  sub  /uav_intercept/target_state   uav_common_msg/msg/TargetState
  sub  /fmu/out/vehicle_odometry     px4_msgs/msg/VehicleOdometry
  pub  /offboard_control/cmd_vel     geometry_msgs/msg/TwistStamped
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistStamped
from px4_msgs.msg import VehicleOdometry
from uav_common_msg.msg import TargetState

try:
    from std_srvs.srv import Trigger
except ImportError:  # pragma: no cover
    Trigger = None


POS_NORM = 50.0
VEL_NORM = 20.0
OBS_DIM = 12
ACTION_DIM = 4


@dataclass
class RawState:
    relative_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    relative_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    own_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    range_m: float = 999.0
    closing_velocity: float = 0.0
    valid: bool = False
    target_state_age: float = 999.0


class TrainingBridge(Node):
    def __init__(self):
        super().__init__("uav_rl_training_bridge")

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._lock = threading.Lock()
        self._state = RawState()
        self._last_target_state_time = 0.0

        self.create_subscription(
            TargetState, "/uav_intercept/target_state", self._on_target_state, 10
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odometry, px4_qos
        )
        self._cmd_pub = self.create_publisher(TwistStamped, "/offboard_control/cmd_vel", 10)

        self._reset_client = None
        if Trigger is not None:
            self._reset_client = self.create_client(Trigger, "/uav_intercept/reset_episode")

    # -- subscriptions --------------------------------------------------
    def _on_target_state(self, msg: TargetState):
        with self._lock:
            self._state.relative_position = np.array(
                [msg.relative_position.x, msg.relative_position.y, msg.relative_position.z],
                dtype=np.float32,
            )
            self._state.relative_velocity = np.array(
                [msg.relative_velocity.x, msg.relative_velocity.y, msg.relative_velocity.z],
                dtype=np.float32,
            )
            self._state.range_m = float(msg.range)
            self._state.closing_velocity = float(msg.closing_velocity)
            self._state.valid = bool(msg.valid)
            self._last_target_state_time = time.time()

    def _on_odometry(self, msg: VehicleOdometry):
        with self._lock:
            self._state.own_velocity = np.array(msg.velocity, dtype=np.float32)

    # -- public API used by the Gym env ----------------------------------
    def get_observation(self) -> np.ndarray:
        """Same normalization as uav_control/rl_inference_node.cpp -- keep in sync."""
        with self._lock:
            s = self._state
            age = time.time() - self._last_target_state_time
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = s.relative_position / POS_NORM
        obs[3:6] = s.relative_velocity / VEL_NORM
        obs[6:9] = s.own_velocity / VEL_NORM
        obs[9] = s.range_m / POS_NORM
        obs[10] = s.closing_velocity / VEL_NORM
        obs[11] = 1.0 if (s.valid and age < 0.5) else 0.0
        return obs

    def get_raw_state(self) -> RawState:
        with self._lock:
            return RawState(**vars(self._state))

    def send_action(self, action: np.ndarray, max_speed_mps: float, max_yaw_rate_rps: float):
        """action is length-4, each component in [-1, 1] (matches the ONNX policy's output space)."""
        action = np.clip(action, -1.0, 1.0)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link_ned"
        msg.twist.linear.x = float(action[0]) * max_speed_mps
        msg.twist.linear.y = float(action[1]) * max_speed_mps
        msg.twist.linear.z = float(action[2]) * max_speed_mps
        msg.twist.angular.z = float(action[3]) * max_yaw_rate_rps
        self._cmd_pub.publish(msg)

    def send_hold(self):
        self._cmd_pub.publish(TwistStamped())

    def call_reset_service(self, timeout_sec: float = 2.0) -> bool:
        """
        Calls /uav_intercept/reset_episode if you've implemented it (e.g. in
        target_sim, to respawn the target at a random start pose). This does
        NOT reset the interceptor's own position -- PX4 SITL / Gazebo don't
        expose that cleanly over ROS2 by default. See training/README section
        "About episode resets" for options.
        """
        if self._reset_client is None or not self._reset_client.service_is_ready():
            return False
        future = self._reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        return future.done() and future.result() is not None and future.result().success


class BridgeHandle:
    """Owns the background executor thread so intercept_env.py can stay a plain Gym env."""

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = TrainingBridge()
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._executor.shutdown()
        self.node.destroy_node()
