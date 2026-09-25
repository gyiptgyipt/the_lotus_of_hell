#ifndef UAV_CONTROL__OFFBOARD_CONTROL_HPP_
#define UAV_CONTROL__OFFBOARD_CONTROL_HPP_

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <chrono>
#include <mutex>

namespace uav_control
{

// Sits between "something that wants the drone to move at velocity (vx,vy,vz,yaw_rate)"
// and PX4's offboard velocity setpoint interface. Neither the RL policy nor the
// training bridge talk to PX4 directly -- they publish TwistStamped on
// ~/cmd_vel and this node does the handshake + streams setpoints at the rate
// PX4 requires.
class OffboardControl : public rclcpp::Node
{
public:
  OffboardControl();

private:
  static constexpr double kControlPeriodSec = 0.05;   // 20 Hz, PX4 requires >2 Hz, we use headroom
  static constexpr int kOffboardSetpointWarmup = 20;   // ticks of streaming before requesting offboard
  static constexpr double kCmdTimeoutSec = 0.5;        // if no cmd_vel received in this long, hold/zero

  void timer_callback();
  void cmd_vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void vehicle_status_callback(const px4_msgs::msg::VehicleStatus::SharedPtr msg);

  void publish_offboard_control_mode();
  void publish_trajectory_setpoint();
  void publish_vehicle_command(uint16_t command, float param1 = 0.0f, float param2 = 0.0f);
  void arm();
  void engage_offboard_mode();

  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
  rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_command_pub_;

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr vehicle_status_sub_;

  rclcpp::TimerBase::SharedPtr timer_;

  uint64_t offboard_setpoint_counter_{0};
  uint8_t nav_state_{0};
  uint8_t arming_state_{0};

  std::mutex cmd_mutex_;
  float cmd_vx_{0.0f};
  float cmd_vy_{0.0f};
  float cmd_vz_{0.0f};
  float cmd_yaw_rate_{0.0f};
  rclcpp::Time last_cmd_time_;

  double takeoff_altitude_m_;   // negative-down NED, so e.g. -3.0 = 3 m above ground
};

}  // namespace uav_control

#endif  // UAV_CONTROL__OFFBOARD_CONTROL_HPP_
