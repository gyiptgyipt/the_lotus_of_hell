#!/usr/bin/python3
"""Vision-only lead-pursuit interceptor for a PX4 multicopter.

The image is the only source for leader-relative motion.  PX4 local attitude is
used only to rotate camera/body velocity requests into PX4's NED world frame.
"""
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleAttitude, VehicleCommand, VehicleLocalPosition)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class VisionFollower(Node):
    TAKEOFF_ALTITUDE_M = 5.0

    def __init__(self):
        super().__init__('vision_follower')
        self.declare_parameter('image_topic', '/uav_2/camera/image')
        self.declare_parameter('debug_image_topic', '/uav_2/camera/detections')
        self.declare_parameter('px4_namespace', '/px4_2')
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_class_names', ['drone', 'uav', 'quadcopter'])
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('inference_size', 640)
        self.declare_parameter('desired_distance_m', 4.0)
        self.declare_parameter('target_size_m', 0.50)
        self.declare_parameter('horizontal_fov_rad', 1.3962634)
        self.declare_parameter('max_speed_mps', 1.5)
        self.declare_parameter('max_yaw_rate_rps', 0.6)
        self.declare_parameter('range_gain', 0.55)
        self.declare_parameter('lateral_gain', 1.0)
        self.declare_parameter('vertical_gain', 0.8)
        self.declare_parameter('yaw_gain', 0.8)
        self.declare_parameter('min_target_area_px', 200)
        self.declare_parameter('lost_target_timeout_s', 0.5)
        self.declare_parameter('intercept_distance_m', 2.0)
        self.declare_parameter('prediction_horizon_s', 1.2)
        self.declare_parameter('relative_velocity_alpha', 0.35)
        self.p = {name: self.get_parameter(name).value for name in [
            'model_path', 'target_class_names', 'confidence_threshold', 'inference_size',
            'desired_distance_m', 'target_size_m', 'horizontal_fov_rad',
            'max_speed_mps', 'max_yaw_rate_rps', 'range_gain', 'lateral_gain',
            'vertical_gain', 'yaw_gain', 'min_target_area_px', 'lost_target_timeout_s',
            'intercept_distance_m', 'prediction_horizon_s', 'relative_velocity_alpha']}
        ns = self.get_parameter('px4_namespace').value.rstrip('/')
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.mode_pub = self.create_publisher(
            OffboardControlMode, ns + '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, ns + '/fmu/in/trajectory_setpoint', px4_qos)
        self.command_pub = self.create_publisher(
            VehicleCommand, ns + '/fmu/in/vehicle_command', px4_qos)
        self.create_subscription(
            VehicleAttitude, ns + '/fmu/out/vehicle_attitude', self.attitude_cb, px4_qos)
        self.create_subscription(VehicleLocalPosition, ns + '/fmu/out/vehicle_local_position',
                                 self.local_position_cb, px4_qos)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.image_cb, qos_profile_sensor_data)
        self.debug_image_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, qos_profile_sensor_data)
        self.model = self.load_model()
        model_names = {str(name).lower() for name in self.model.names.values()}
        configured_names = {str(name).lower() for name in self.p['target_class_names']}
        if not model_names.intersection(configured_names):
            self.get_logger().error(
                f'YOLO classes {sorted(model_names)} do not include any configured '
                f'target class {sorted(configured_names)}. Replace model_path or '
                'set target_class_names to classes trained to detect drones.')
        self.q = None
        self.own_velocity_ned = np.zeros(3)
        self.target = None  # (normalised horizontal error, vertical error, distance, timestamp)
        self.relative_position_ned = None
        self.target_velocity_ned = np.zeros(3)
        self.last_relative_position_ned = None
        self.last_detection_time = None
        self.image_count = 0
        self.inference_count = 0
        self.detection_count = 0
        self.last_image_time = None
        self.last_status_log = 0.0
        self.last_detection_log = 0.0
        self.setpoint_count = 0
        self.offboard_started = False
        self.timer = self.create_timer(0.05, self.control_tick)  # PX4 requires a continuous >2 Hz stream
        self.get_logger().info(
            f'YOLO loaded. Reading camera images from '
            f'{self.get_parameter("image_topic").value}; waiting for UAV 1.')

    def load_model(self):
        path = self.p['model_path']
        if not path:
            raise RuntimeError(
                'Set model_path to drone-trained YOLO .pt weights in follower.yaml or at launch.')
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                'Missing ultralytics. Install it for ROS system Python: '
                '/usr/bin/python3 -m pip install --user ultralytics') from error
        try:
            return YOLO(path)
        except Exception as error:
            raise RuntimeError(f'Cannot load YOLO weights at {path}: {error}') from error

    @staticmethod
    def now_us():
        return int(time.time() * 1_000_000)

    def attitude_cb(self, msg):
        # PX4 VehicleAttitude q is the Hamilton quaternion rotating body FRD -> NED.
        self.q = msg.q

    def local_position_cb(self, msg):
        # PX4 local velocity fields are in the NED frame.
        self.own_velocity_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=float)

    def image_cb(self, msg):
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            channels = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
            # Respect Image.step: Gazebo images normally have no row padding, but
            # this also works when a bridge inserts it.
            rows = raw.reshape(msg.height, msg.step)[:, :msg.width * channels]
            image = rows.reshape(msg.height, msg.width, channels)
            if msg.encoding == 'rgb8':
                rgb = image
            elif msg.encoding == 'bgr8':
                rgb = image[:, :, ::-1]
            elif msg.encoding == 'rgba8':
                rgb = image[:, :, :3]
            elif msg.encoding == 'bgra8':
                rgb = image[:, :, [2, 1, 0]]
            else:
                raise ValueError(f'unsupported encoding {msg.encoding}; use rgb8 or bgr8')
        except Exception as error:
            self.get_logger().warn(f'Cannot convert camera image: {error}')
            return
        self.image_count += 1
        self.last_image_time = time.monotonic()
        try:
            result = self.model(
                rgb, imgsz=self.p['inference_size'],
                conf=self.p['confidence_threshold'], verbose=False)[0]
        except Exception as error:
            self.get_logger().error(f'YOLO inference failed: {error}')
            return
        self.inference_count += 1
        allowed = {name.lower() for name in self.p['target_class_names']}
        candidates = []
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id]).lower()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            width, height = x2 - x1, y2 - y1
            area = width * height
            confidence = float(box.conf[0].item())
            color = (0, 220, 0) if class_name in allowed else (120, 120, 120)
            cv2.rectangle(
                annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(
                annotated, label, (int(x1), max(20, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if class_name in allowed and area >= self.p['min_target_area_px']:
                candidates.append((confidence, area, x1, y1, width, height))
        if not len(result.boxes):
            cv2.putText(
                annotated, 'NO DETECTIONS', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        debug_msg = Image()
        debug_msg.header = msg.header
        debug_msg.height, debug_msg.width = msg.height, msg.width
        debug_msg.encoding = 'bgr8'
        debug_msg.is_bigendian = 0
        debug_msg.step = msg.width * 3
        debug_msg.data = annotated.tobytes()
        self.debug_image_pub.publish(debug_msg)
        if not candidates:
            return
        # Prefer a close, high-confidence drone when more than one is visible.
        confidence, _, x, y, box_w, box_h = max(
            candidates, key=lambda item: item[0] * item[1])
        self.detection_count += 1
        image_height, image_width = rgb.shape[:2]
        u_error = ((x + box_w / 2) - image_width / 2) / (image_width / 2)
        v_error = ((y + box_h / 2) - image_height / 2) / (image_height / 2)
        # Pinhole range from known X500 width and detected bounding-box width.
        focal_px = image_width / (2 * math.tan(self.p['horizontal_fov_rad'] / 2))
        distance = self.p['target_size_m'] * focal_px / max(box_w, 1)
        detected_at = time.monotonic()
        self.target = (u_error, v_error, distance, detected_at)
        if detected_at - self.last_detection_log >= 1.0:
            self.get_logger().info(
                f'Target detected: confidence={confidence:.2f}, '
                f'box={box_w:.0f}x{box_h:.0f}px, range={distance:.1f}m, '
                f'image_error=({u_error:+.2f}, {v_error:+.2f})')
            self.last_detection_log = detected_at

        # Camera optical coordinates approximately match PX4 body FRD here:
        # forward, right, down. Convert the YOLO box centre into a 3-D bearing.
        if self.q is None:
            return
        vertical_fov = 2 * math.atan(math.tan(self.p['horizontal_fov_rad'] / 2)
                                     * image_height / image_width)
        relative_body = np.array([
            distance,
            distance * math.tan(u_error * self.p['horizontal_fov_rad'] / 2),
            distance * math.tan(v_error * vertical_fov / 2),
        ])
        relative_ned = np.array(self.body_to_ned(relative_body))
        if self.last_relative_position_ned is not None and self.last_detection_time is not None:
            dt = detected_at - self.last_detection_time
            if 0.02 < dt < 1.0:
                # Relative velocity plus our velocity estimates UAV 1's NED velocity.
                measured_velocity = self.own_velocity_ned + (relative_ned - self.last_relative_position_ned) / dt
                alpha = self.p['relative_velocity_alpha']
                self.target_velocity_ned = alpha * measured_velocity + (1 - alpha) * self.target_velocity_ned
        self.relative_position_ned = relative_ned
        self.last_relative_position_ned = relative_ned
        self.last_detection_time = detected_at

    def body_to_ned(self, body):
        if self.q is None:
            return [0.0, 0.0, 0.0]
        w, x, y, z = self.q
        # Rotation matrix for q, where axes are PX4 body FRD and local NED.
        rotation = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        return (rotation @ np.array(body)).tolist()

    def publish_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.now_us()
        msg.command = command
        msg.param1, msg.param2 = float(param1), float(param2)
        msg.target_system = msg.source_system = 1
        msg.target_component = msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    def control_tick(self):
        now = time.monotonic()
        mode = OffboardControlMode()
        mode.timestamp = self.now_us()
        mode.position = not self.offboard_started
        mode.velocity = self.offboard_started
        mode.acceleration = mode.attitude = mode.body_rate = False
        self.mode_pub.publish(mode)

        if now - self.last_status_log >= 2.0:
            image_age = 'never' if self.last_image_time is None else f'{now - self.last_image_time:.1f}s ago'
            target_age = 'never' if self.last_detection_time is None else f'{now - self.last_detection_time:.1f}s ago'
            self.get_logger().info(
                f'CV status: images={self.image_count}, inferences={self.inference_count}, '
                f'detections={self.detection_count}, last_image={image_age}, '
                f'last_detection={target_age}, attitude={"ready" if self.q is not None else "missing"}, '
                f'offboard={"active" if self.offboard_started else "preparing"}')
            self.last_status_log = now

        # PX4 needs a stream of position setpoints before accepting OFFBOARD.
        # NED uses a negative Z value for altitude above the local origin.
        if not self.offboard_started:
            setpoint = TrajectorySetpoint()
            setpoint.timestamp = self.now_us()
            setpoint.position = [0.0, 0.0, -self.TAKEOFF_ALTITUDE_M]
            setpoint.velocity = [math.nan, math.nan, math.nan]
            setpoint.acceleration = [math.nan, math.nan, math.nan]
            setpoint.yaw = 0.0
            setpoint.yawspeed = math.nan
            self.setpoint_pub.publish(setpoint)
            self.setpoint_count += 1

            if self.setpoint_count >= 50:
                self.publish_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                if self.get_parameter('auto_arm').value:
                    self.publish_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.offboard_started = True
                self.get_logger().info(
                    'Requested OFFBOARD and arm for UAV 2 at 5 m altitude.'
                    if self.get_parameter('auto_arm').value
                    else 'Requested OFFBOARD for UAV 2 at 5 m altitude; arm is disabled.')
            return

        # A stale/absent target means stop. Never continue on a previous intercept vector.
        velocity = [0.0, 0.0, 0.0]
        yaw_rate = 0.0
        if (self.target and self.relative_position_ned is not None
                and now - self.target[3] <= self.p['lost_target_timeout_s']):
            u, _, distance, _ = self.target
            separation = self.p['intercept_distance_m']
            if distance > separation:
                # Lead pursuit: aim where UAV 1 should be after a bounded prediction horizon,
                # then add its estimated velocity as feed-forward.
                closing_speed = clamp(self.p['range_gain'] * (distance - separation), 0.0,
                                      self.p['max_speed_mps'])
                eta = clamp(distance / max(closing_speed, 0.2), 0.0,
                            self.p['prediction_horizon_s'])
                predicted_relative = self.relative_position_ned + self.target_velocity_ned * eta
                predicted_norm = np.linalg.norm(predicted_relative)
                pursuit = (predicted_relative / predicted_norm * closing_speed
                           if predicted_norm > 0.01 else np.zeros(3))
                requested = self.target_velocity_ned + pursuit
                requested_norm = np.linalg.norm(requested)
                if requested_norm > self.p['max_speed_mps']:
                    requested *= self.p['max_speed_mps'] / requested_norm
                velocity = requested.tolist()
            # At the intercept radius, command zero. Set intercept_distance_m lower
            # only after testing; this avoids a deliberate simulated collision.
            yaw_rate = clamp(self.p['yaw_gain'] * u,
                             -self.p['max_yaw_rate_rps'], self.p['max_yaw_rate_rps'])
        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self.now_us()
        setpoint.position = [math.nan, math.nan, math.nan]
        setpoint.velocity = velocity
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.yaw = math.nan
        setpoint.yawspeed = yaw_rate
        self.setpoint_pub.publish(setpoint)


def main():
    rclpy.init()
    node = VisionFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
