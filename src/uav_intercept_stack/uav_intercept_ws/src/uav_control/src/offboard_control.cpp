#include "uav_control/offboard_control.hpp"

namespace uav_control
{

using namespace std::chrono_literals;
using px4_msgs::msg::VehicleCommand;

OffboardControl::OffboardControl() : Node("offboard_control")
{
  this->declare_parameter<double>("takeoff_altitude_m", 3.0);
  takeoff_altitude_m_ = -std::abs(this->get_parameter("takeoff_altitude_m").as_double());  // NED: up is negative

  rclcpp::QoS px4_qos(rclcpp::KeepLast(1));
  px4_qos.best_effort();
  px4_qos.durability_volatile();

  offboard_control_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
    "/fmu/in/offboard_control_mode", 10);
  trajectory_setpoint_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
    "/fmu/in/trajectory_setpoint", 10);
  vehicle_command_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
    "/fmu/in/vehicle_command", 10);

  vehicle_status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
    "/fmu/out/vehicle_status_v1", px4_qos,
    std::bind(&OffboardControl::vehicle_status_callback, this, std::placeholders::_1));

  cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
    "~/cmd_vel", 10,
    std::bind(&OffboardControl::cmd_vel_callback, this, std::placeholders::_1));

  last_cmd_time_ = this->now();

  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(kControlPeriodSec),
    std::bind(&OffboardControl::timer_callback, this));

  RCLCPP_INFO(this->get_logger(),
    "offboard_control started. Streaming setpoints, will arm + engage offboard after warmup.");
}

void OffboardControl::vehicle_status_callback(const px4_msgs::msg::VehicleStatus::SharedPtr msg)
{
  nav_state_ = msg->nav_state;
  arming_state_ = msg->arming_state;
}

void OffboardControl::cmd_vel_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  // NOTE: this assumes the publisher already sends commands in the NED frame
  // (x=North, y=East, z=Down) to match PX4's TrajectorySetpoint convention.
  cmd_vx_ = static_cast<float>(msg->twist.linear.x);
  cmd_vy_ = static_cast<float>(msg->twist.linear.y);
  cmd_vz_ = static_cast<float>(msg->twist.linear.z);
  cmd_yaw_rate_ = static_cast<float>(msg->twist.angular.z);
  last_cmd_time_ = this->now();
}

void OffboardControl::timer_callback()
{
  if (offboard_setpoint_counter_ == kOffboardSetpointWarmup) {
    engage_offboard_mode();
    arm();
  }

  publish_offboard_control_mode();
  publish_trajectory_setpoint();

  if (offboard_setpoint_counter_ < kOffboardSetpointWarmup + 1) {
    offboard_setpoint_counter_++;
  }
}

void OffboardControl::publish_offboard_control_mode()
{
  px4_msgs::msg::OffboardControlMode msg{};
  msg.position = false;
  msg.velocity = true;
  msg.acceleration = false;
  msg.attitude = false;
  msg.body_rate = false;
  msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
  offboard_control_mode_pub_->publish(msg);
}

void OffboardControl::publish_trajectory_setpoint()
{
  px4_msgs::msg::TrajectorySetpoint msg{};
  msg.position = {NAN, NAN, NAN};
  msg.yaw = NAN;

  bool cmd_fresh;
  float vx, vy, vz, yaw_rate;
  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    cmd_fresh = (this->now() - last_cmd_time_).seconds() < kCmdTimeoutSec;
    vx = cmd_vx_;
    vy = cmd_vy_;
    vz = cmd_vz_;
    yaw_rate = cmd_yaw_rate_;
  }

  if (offboard_setpoint_counter_ < kOffboardSetpointWarmup) {
    // Still warming up: climb to takeoff altitude with a plain velocity command,
    // no external commander is driving yet.
    msg.velocity = {0.0f, 0.0f, static_cast<float>(takeoff_altitude_m_ < 0 ? -0.5 : 0.5)};
    msg.yawspeed = 0.0f;
  } else if (cmd_fresh) {
    msg.velocity = {vx, vy, vz};
    msg.yawspeed = yaw_rate;
  } else {
    // Safety: no recent command from the RL/training bridge -- hold (zero velocity)
    // rather than keep executing a stale command.
    msg.velocity = {0.0f, 0.0f, 0.0f};
    msg.yawspeed = 0.0f;
  }

  msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
  trajectory_setpoint_pub_->publish(msg);
}

void OffboardControl::publish_vehicle_command(uint16_t command, float param1, float param2)
{
  VehicleCommand msg{};
  msg.param1 = param1;
  msg.param2 = param2;
  msg.command = command;
  msg.target_system = 1;
  msg.target_component = 1;
  msg.source_system = 1;
  msg.source_component = 1;
  msg.from_external = true;
  msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
  vehicle_command_pub_->publish(msg);
}

void OffboardControl::arm()
{
  publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0f);
  RCLCPP_INFO(this->get_logger(), "Arm command sent");
}

void OffboardControl::engage_offboard_mode()
{
  // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1, PX4_CUSTOM_MAIN_MODE_OFFBOARD=6
  publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1.0f, 6.0f);
  RCLCPP_INFO(this->get_logger(), "Offboard mode command sent");
}

}  // namespace uav_control
