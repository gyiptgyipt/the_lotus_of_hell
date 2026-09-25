#include "uav_vision_detect/vision_detect_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <cmath>

namespace uav_vision_detect
{

VisionDetectNode::VisionDetectNode() : Node("uav_vision_detect")
{
  this->declare_parameter<std::string>("model_path", "");
  this->declare_parameter<double>("target_size_m", 0.35);   // characteristic size (e.g. diagonal/wingspan) of the target drone
  this->declare_parameter<double>("conf_threshold", 0.4);
  this->declare_parameter<double>("nms_iou_threshold", 0.45);
  this->declare_parameter<int>("model_input_size", 640);
  this->declare_parameter<int>("max_missed_frames", 5);
  this->declare_parameter<double>("velocity_smoothing_alpha", 0.3);
  this->declare_parameter<std::vector<double>>("mount_q_wxyz_offset", {1.0, 0.0, 0.0, 0.0});

  target_size_m_ = this->get_parameter("target_size_m").as_double();
  conf_threshold_ = static_cast<float>(this->get_parameter("conf_threshold").as_double());
  nms_iou_threshold_ = static_cast<float>(this->get_parameter("nms_iou_threshold").as_double());
  model_input_size_ = this->get_parameter("model_input_size").as_int();
  max_missed_frames_ = this->get_parameter("max_missed_frames").as_int();
  velocity_smoothing_alpha_ = this->get_parameter("velocity_smoothing_alpha").as_double();

  auto mount_q = this->get_parameter("mount_q_wxyz_offset").as_double_array();
  if (mount_q.size() == 4) {
    mount_q_wxyz_offset_ = {mount_q[0], mount_q[1], mount_q[2], mount_q[3]};
  }

  std::string model_path = this->get_parameter("model_path").as_string();
  if (model_path.empty()) {
    RCLCPP_FATAL(this->get_logger(), "No model_path parameter set. Point it at the exported YOLO .onnx.");
    throw std::runtime_error("model_path not set");
  }
  detector_ = std::make_unique<YoloDetector>(model_path, model_input_size_, conf_threshold_, nms_iou_threshold_);

  rclcpp::QoS px4_qos(rclcpp::KeepLast(1));
  px4_qos.best_effort();
  px4_qos.durability_volatile();

  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "camera/image_raw", rclcpp::SensorDataQoS(),
    std::bind(&VisionDetectNode::image_callback, this, std::placeholders::_1));
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera/camera_info", rclcpp::SensorDataQoS(),
    std::bind(&VisionDetectNode::camera_info_callback, this, std::placeholders::_1));
  odometry_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
    "/fmu/out/vehicle_odometry", px4_qos,
    std::bind(&VisionDetectNode::odometry_callback, this, std::placeholders::_1));

  target_state_pub_ = this->create_publisher<uav_common_msg::msg::TargetState>(
    "/uav_intercept/target_state_vision", 10);

  RCLCPP_INFO(this->get_logger(),
    "uav_vision_detect ready. Publishing to /uav_intercept/target_state_vision "
    "-- remap to /uav_intercept/target_state to use this instead of ground truth.");
}

void VisionDetectNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  fx_ = msg->k[0];
  fy_ = msg->k[4];
  cx_ = msg->k[2];
  cy_ = msg->k[5];
  have_camera_info_ = true;
}

void VisionDetectNode::odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
  own_q_wxyz_ = msg->q;   // PX4 convention: rotation from body FRD to reference (NED) frame, order (w,x,y,z)
  have_odometry_ = true;
}

Vec3 VisionDetectNode::rotate_by_quaternion(const std::array<float, 4> & q_wxyz, const Vec3 & v)
{
  // Standard quaternion vector rotation: v' = q * v * q^-1, expanded.
  const double w = q_wxyz[0], x = q_wxyz[1], y = q_wxyz[2], z = q_wxyz[3];
  const double vx = v.x, vy = v.y, vz = v.z;

  const double uvx = 2.0 * (y * vz - z * vy);
  const double uvy = 2.0 * (z * vx - x * vz);
  const double uvz = 2.0 * (x * vy - y * vx);

  const double uuvx = y * uvz - z * uvy;
  const double uuvy = z * uvx - x * uvz;
  const double uuvz = x * uvy - y * uvx;

  return Vec3{
    vx + w * uvx + uuvx,
    vy + w * uvy + uuvy,
    vz + w * uvz + uuvz
  };
}

Vec3 VisionDetectNode::apply_camera_mount(const Vec3 & v_optical)
{
  // Fixed permutation for a forward-facing camera whose optical frame
  // (x-right, y-down, z-forward) is aligned with body FRD (x-forward,
  // y-right, z-down) at zero mount rotation:
  //   body_x = optical_z, body_y = optical_x, body_z = optical_y
  return Vec3{v_optical.z, v_optical.x, v_optical.y};
}

std::optional<Vec3> VisionDetectNode::estimate_relative_position(
  const Detection & det, int /*image_width*/, int /*image_height*/) const
{
  if (!have_camera_info_) {
    return std::nullopt;
  }

  const double u = det.box.x + det.box.width / 2.0;
  const double v = det.box.y + det.box.height / 2.0;

  // Unit ray in the camera optical frame (x-right, y-down, z-forward).
  Vec3 ray{(u - cx_) / fx_, (v - cy_) / fy_, 1.0};
  const double norm = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
  ray.x /= norm; ray.y /= norm; ray.z /= norm;

  // Apparent-size range estimate: assumes fx ~= fy (square-ish pixels) and
  // that target_size_m_ corresponds to the LARGER of the bbox width/height,
  // i.e. the target's characteristic size at whatever aspect it's viewed
  // from. This is a coarse estimate -- expect noisy range, not exact range.
  const double apparent_size_px = std::max(det.box.width, det.box.height);
  if (apparent_size_px < 1.0) {
    return std::nullopt;
  }
  const double range = (target_size_m_ * fx_) / apparent_size_px;

  Vec3 v_optical{ray.x * range, ray.y * range, ray.z * range};
  Vec3 v_body = apply_camera_mount(v_optical);

  std::array<float, 4> mount_q{
    static_cast<float>(mount_q_wxyz_offset_[0]), static_cast<float>(mount_q_wxyz_offset_[1]),
    static_cast<float>(mount_q_wxyz_offset_[2]), static_cast<float>(mount_q_wxyz_offset_[3])};
  v_body = rotate_by_quaternion(mount_q, v_body);

  Vec3 v_ned = rotate_by_quaternion(own_q_wxyz_, v_body);
  return v_ned;
}

void VisionDetectNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  if (!have_odometry_) {
    return;  // need own attitude to convert bearing into NED, nothing useful to publish yet
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    return;
  }

  std::vector<Detection> dets = detector_->detect(cv_ptr->image);

  uav_common_msg::msg::TargetState out;
  out.header.stamp = msg->header.stamp;
  out.header.frame_id = "ned";

  if (dets.empty()) {
    no_detection_count_++;
    out.valid = false;
    target_state_pub_->publish(out);
    if (no_detection_count_ > max_missed_frames_) {
      have_prev_position_ = false;   // stale track, don't finite-difference across the gap when we reacquire
    }
    return;
  }

  const Detection & best = dets.front();  // highest confidence after NMS
  auto rel_pos_opt = estimate_relative_position(best, cv_ptr->image.cols, cv_ptr->image.rows);
  if (!rel_pos_opt) {
    out.valid = false;
    target_state_pub_->publish(out);
    return;
  }
  Vec3 rel_pos = *rel_pos_opt;
  no_detection_count_ = 0;

  rclcpp::Time now = msg->header.stamp;
  Vec3 raw_velocity{0.0, 0.0, 0.0};
  if (have_prev_position_) {
    double dt = (now - prev_time_).seconds();
    if (dt > 1e-3) {
      raw_velocity.x = (rel_pos.x - prev_relative_position_.x) / dt;
      raw_velocity.y = (rel_pos.y - prev_relative_position_.y) / dt;
      raw_velocity.z = (rel_pos.z - prev_relative_position_.z) / dt;
      double a = velocity_smoothing_alpha_;
      smoothed_relative_velocity_.x = a * raw_velocity.x + (1 - a) * smoothed_relative_velocity_.x;
      smoothed_relative_velocity_.y = a * raw_velocity.y + (1 - a) * smoothed_relative_velocity_.y;
      smoothed_relative_velocity_.z = a * raw_velocity.z + (1 - a) * smoothed_relative_velocity_.z;
    }
  } else {
    smoothed_relative_velocity_ = {0.0, 0.0, 0.0};   // no history yet (first detection, or just reacquired)
  }

  double range = std::sqrt(rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z);
  double closing_velocity = 0.0;
  if (range > 1e-3) {
    double range_rate =
      (rel_pos.x * smoothed_relative_velocity_.x +
       rel_pos.y * smoothed_relative_velocity_.y +
       rel_pos.z * smoothed_relative_velocity_.z) / range;
    closing_velocity = -range_rate;
  }

  out.relative_position.x = rel_pos.x;
  out.relative_position.y = rel_pos.y;
  out.relative_position.z = rel_pos.z;
  out.relative_velocity.x = smoothed_relative_velocity_.x;
  out.relative_velocity.y = smoothed_relative_velocity_.y;
  out.relative_velocity.z = smoothed_relative_velocity_.z;
  out.range = static_cast<float>(range);
  out.closing_velocity = static_cast<float>(closing_velocity);
  out.valid = true;

  target_state_pub_->publish(out);

  prev_relative_position_ = rel_pos;
  prev_time_ = now;
  have_prev_position_ = true;
}

}  // namespace uav_vision_detect
