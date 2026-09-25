#ifndef UAV_CONTROL__RL_INFERENCE_NODE_HPP_
#define UAV_CONTROL__RL_INFERENCE_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <uav_common_msg/msg/target_state.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <mutex>

namespace uav_control
{

// Observation layout fed to the policy (float32, length kObsDim). Keep this in
// lockstep with training/uav_rl_training/intercept_env.py -- both sides must
// agree on the exact ordering and normalization or the policy will be fed
// garbage at inference time.
//   [0:3]  relative_position / pos_norm      (target - own, NED, m)
//   [3:6]  relative_velocity / vel_norm      (target - own, NED, m/s)
//   [6:9]  own_velocity / vel_norm           (NED, m/s)
//   [9]    range / pos_norm                  (m)
//   [10]   closing_velocity / vel_norm       (m/s, +ve = closing)
//   [11]   target_valid (0.0 or 1.0)
constexpr int kObsDim = 12;
constexpr int kActionDim = 4;   // vx, vy, vz, yaw_rate, each in [-1, 1] out of the policy

class RLInferenceNode : public rclcpp::Node
{
public:
  RLInferenceNode();

private:
  void target_state_callback(const uav_common_msg::msg::TargetState::SharedPtr msg);
  void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void control_timer_callback();

  void run_policy(const std::array<float, kObsDim> & obs, std::array<float, kActionDim> & action_out);
  void publish_hold();
  void publish_action(const std::array<float, kActionDim> & action);

  rclcpp::Subscription<uav_common_msg::msg::TargetState>::SharedPtr target_state_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  // ONNX Runtime handles
  Ort::Env ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;

  std::mutex state_mutex_;
  uav_common_msg::msg::TargetState latest_target_state_;
  bool have_target_state_{false};
  rclcpp::Time last_target_state_time_;

  std::array<float, 3> own_velocity_{0.0f, 0.0f, 0.0f};
  bool have_odometry_{false};

  // Normalization / scaling constants -- must match training.
  double pos_norm_;
  double vel_norm_;
  double max_speed_mps_;
  double max_yaw_rate_rps_;
  double watchdog_timeout_sec_;
  double capture_radius_m_;
};

}  // namespace uav_control

#endif  // UAV_CONTROL__RL_INFERENCE_NODE_HPP_
