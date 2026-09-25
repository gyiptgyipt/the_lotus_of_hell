#ifndef UAV_VISION_DETECT__VISION_DETECT_NODE_HPP_
#define UAV_VISION_DETECT__VISION_DETECT_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <uav_common_msg/msg/target_state.hpp>
#include <array>
#include <memory>
#include <optional>

#include "uav_vision_detect/yolo_detector.hpp"

namespace uav_vision_detect
{

struct Vec3
{
  double x{0.0}, y{0.0}, z{0.0};
};

class VisionDetectNode : public rclcpp::Node
{
public:
  VisionDetectNode();

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

  // Converts a pixel-space detection + own attitude into a relative-position
  // estimate in NED, using the apparent-size range method.
  std::optional<Vec3> estimate_relative_position(
    const Detection & det, int image_width, int image_height) const;

  static Vec3 rotate_by_quaternion(const std::array<float, 4> & q_wxyz, const Vec3 & v);
  static Vec3 apply_camera_mount(const Vec3 & v_optical);

  std::unique_ptr<YoloDetector> detector_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<uav_common_msg::msg::TargetState>::SharedPtr target_state_pub_;

  bool have_camera_info_{false};
  double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};

  std::array<float, 4> own_q_wxyz_{1.0f, 0.0f, 0.0f, 0.0f};
  bool have_odometry_{false};

  // Simple state for finite-difference velocity + basic temporal smoothing.
  bool have_prev_position_{false};
  Vec3 prev_relative_position_{};
  Vec3 smoothed_relative_velocity_{};
  rclcpp::Time prev_time_;

  int no_detection_count_{0};

  // Params
  double target_size_m_;
  float conf_threshold_;
  float nms_iou_threshold_;
  int model_input_size_;
  int max_missed_frames_;
  double velocity_smoothing_alpha_;
  std::array<double, 4> mount_q_wxyz_offset_{1.0, 0.0, 0.0, 0.0};
};

}  // namespace uav_vision_detect

#endif  // UAV_VISION_DETECT__VISION_DETECT_NODE_HPP_
