"""
Auto-labels a YOLO dataset by re-using the ground-truth TargetState that
target_sim already publishes: instead of hand-drawing boxes, project the
known 3D relative position into the camera frame and back out a bounding box
from the target's known physical size. This is the exact inverse of what
uav_vision_detect does at inference time (project 3D -> 2D here; 2D -> 3D
there), so keep target_size_m and the camera-mount assumption in sync with
uav_vision_detect/config/params.yaml.

Run this with the SAME sim setup you trained/validated the RL loop against
(PX4 SITL + Gazebo + target_sim, ground truth on /uav_intercept/target_state),
but drive the interceptor around manually or let target_sim/your RL policy
fly it -- the more varied the interceptor's viewpoints, the better the
detector generalizes.

Usage:
  ros2 run ...  (or just: python3 capture_dataset.py --out dataset_raw)
  # fly/simulate for a while, Ctrl+C to stop
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from px4_msgs.msg import VehicleOdometry
from uav_common_msg.msg import TargetState

CLASS_ID = 0  # single class: target_drone


def conjugate(q_wxyz):
    w, x, y, z = q_wxyz
    return (w, -x, -y, -z)


def rotate_by_quaternion(q_wxyz, v):
    w, x, y, z = q_wxyz
    vx, vy, vz = v
    uvx = 2.0 * (y * vz - z * vy)
    uvy = 2.0 * (z * vx - x * vz)
    uvz = 2.0 * (x * vy - y * vx)
    uuvx = y * uvz - z * uvy
    uuvy = z * uvx - x * uvz
    uuvz = x * uvy - y * uvx
    return (vx + w * uvx + uuvx, vy + w * uvy + uuvy, vz + w * uvz + uuvz)


class CaptureNode(Node):
    def __init__(self, out_dir: str, target_size_m: float, save_every_n: int, neg_rate: float):
        super().__init__("uav_vision_dataset_capture")
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, "images")
        self.labels_dir = os.path.join(out_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.target_size_m = target_size_m
        self.save_every_n = max(1, save_every_n)
        self.neg_rate = neg_rate

        self.bridge = CvBridge()
        self.have_camera_info = False
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.own_q = (1.0, 0.0, 0.0, 0.0)
        self.have_odom = False
        self.latest_target = None
        self.frame_count = 0
        self.saved_count = 0

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, "camera/image_raw", self._on_image, 10)
        self.create_subscription(CameraInfo, "camera/camera_info", self._on_camera_info, 10)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odom, px4_qos)
        self.create_subscription(TargetState, "/uav_intercept/target_state", self._on_target, 10)

        self.get_logger().info(f"Capturing to {out_dir} (every {self.save_every_n} frames)")

    def _on_camera_info(self, msg: CameraInfo):
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.have_camera_info = True

    def _on_odom(self, msg: VehicleOdometry):
        self.own_q = tuple(msg.q)
        self.have_odom = True

    def _on_target(self, msg: TargetState):
        self.latest_target = msg

    def _on_image(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % self.save_every_n != 0:
            return
        if not (self.have_camera_info and self.have_odom and self.latest_target is not None):
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = cv_image.shape[:2]

        label_line = self._project_to_label(w, h)

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        fname = f"{stamp_ns}"

        if label_line is None:
            if np.random.rand() > self.neg_rate:
                return  # skip most background frames, keep the dataset target-heavy
            label_text = ""  # empty label file = valid YOLO "no object" example
        else:
            label_text = label_line

        cv2.imwrite(os.path.join(self.images_dir, f"{fname}.jpg"), cv_image)
        with open(os.path.join(self.labels_dir, f"{fname}.txt"), "w") as f:
            f.write(label_text)
        self.saved_count += 1
        if self.saved_count % 50 == 0:
            self.get_logger().info(f"Saved {self.saved_count} frames")

    def _project_to_label(self, img_w: int, img_h: int):
        t = self.latest_target
        if not t.valid:
            return None
        rel_pos_ned = (t.relative_position.x, t.relative_position.y, t.relative_position.z)
        range_m = math.sqrt(sum(c * c for c in rel_pos_ned))
        if range_m < 0.5 or range_m > 80.0:
            return None

        q_ned_to_body = conjugate(self.own_q)   # own_q is body->NED, so conjugate is NED->body
        v_body = rotate_by_quaternion(q_ned_to_body, rel_pos_ned)

        # Inverse of the camera-mount permutation in uav_vision_detect
        # (body_x=opt_z, body_y=opt_x, body_z=opt_y), assuming default
        # identity mount_q_wxyz_offset -- keep this in sync if you change
        # that param.
        opt_x, opt_y, opt_z = v_body[1], v_body[2], v_body[0]

        if opt_z <= 0.3:  # behind or right at the camera plane
            return None

        u = self.fx * (opt_x / opt_z) + self.cx
        v = self.fy * (opt_y / opt_z) + self.cy

        apparent_size_px = (self.target_size_m * self.fx) / range_m
        half = apparent_size_px / 2.0

        x0, y0, x1, y1 = u - half, v - half, u + half, v + half
        # require the box center in-frame with a margin; partial-edge boxes
        # are still useful training signal so we don't require full containment
        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            return None

        x0c, y0c = max(0.0, x0), max(0.0, y0)
        x1c, y1c = min(float(img_w), x1), min(float(img_h), y1)
        bw, bh = x1c - x0c, y1c - y0c
        if bw < 3 or bh < 3:
            return None
        cx_n = (x0c + bw / 2.0) / img_w
        cy_n = (y0c + bh / 2.0) / img_h
        bw_n = bw / img_w
        bh_n = bh / img_h

        return f"{CLASS_ID} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="dataset_raw")
    parser.add_argument("--target-size-m", type=float, default=0.35)
    parser.add_argument("--save-every-n", type=int, default=3, help="save 1 of every N frames")
    parser.add_argument("--neg-rate", type=float, default=0.05,
                         help="fraction of no-target frames to keep as negatives")
    args = parser.parse_args()

    rclpy.init()
    node = CaptureNode(args.out, args.target_size_m, args.save_every_n, args.neg_rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Done. Saved {node.saved_count} labeled frames to {args.out}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
