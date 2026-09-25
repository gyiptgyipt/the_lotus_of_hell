#include "uav_control/rl_inference_node.hpp"
#include <cmath>
#include <algorithm>

namespace uav_control
{

using namespace std::chrono_literals;

RLInferenceNode::RLInferenceNode()
: Node("rl_inference_node"),
  ort_env_(ORT_LOGGING_LEVEL_WARNING, "rl_inference_node"),
  memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
  this->declare_parameter<std::string>("model_path", "");
  this->declare_parameter<double>("pos_norm", 50.0);
  this->declare_parameter<double>("vel_norm", 20.0);
  this->declare_parameter<double>("max_speed_mps", 8.0);
  this->declare_parameter<double>("max_yaw_rate_rps", 1.0);
  this->declare_parameter<double>("watchdog_timeout_sec", 0.3);
  this->declare_parameter<double>("capture_radius_m", 1.0);
  this->declare_parameter<double>("control_rate_hz", 20.0);

  pos_norm_ = this->get_parameter("pos_norm").as_double();
  vel_norm_ = this->get_parameter("vel_norm").as_double();
  max_speed_mps_ = this->get_parameter("max_speed_mps").as_double();
  max_yaw_rate_rps_ = this->get_parameter("max_yaw_rate_rps").as_double();
  watchdog_timeout_sec_ = this->get_parameter("watchdog_timeout_sec").as_double();
  capture_radius_m_ = this->get_parameter("capture_radius_m").as_double();

  std::string model_path = this->get_parameter("model_path").as_string();
  if (model_path.empty()) {
    RCLCPP_FATAL(this->get_logger(),
      "No model_path parameter set. Point it at the exported .onnx policy, e.g. "
      "install/uav_control/share/uav_control/models/policy.onnx");
    throw std::runtime_error("model_path not set");
  }

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  ort_session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr input_name_ptr = ort_session_->GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr output_name_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
  input_name_ = input_name_ptr.get();
  output_name_ = output_name_ptr.get();

  RCLCPP_INFO(this->get_logger(), "Loaded ONNX policy from %s (input='%s', output='%s')",
    model_path.c_str(), input_name_.c_str(), output_name_.c_str());

  rclcpp::QoS px4_qos(rclcpp::KeepLast(1));
  px4_qos.best_effort();
  px4_qos.durability_volatile();

  target_state_sub_ = this->create_subscription<uav_common_msg::msg::TargetState>(
    "/uav_intercept/target_state", 10,
    std::bind(&RLInferenceNode::target_state_callback, this, std::placeholders::_1));

  odometry_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
    "/fmu/out/vehicle_odometry", px4_qos,
    std::bind(&RLInferenceNode::odometry_callback, this, std::placeholders::_1));

  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
    "/offboard_control/cmd_vel", 10);

  double control_rate_hz = this->get_parameter("control_rate_hz").as_double();
  control_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / control_rate_hz),
    std::bind(&RLInferenceNode::control_timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "rl_inference_node ready.");
}

void RLInferenceNode::target_state_callback(const uav_common_msg::msg::TargetState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  latest_target_state_ = *msg;
  have_target_state_ = true;
  last_target_state_time_ = this->now();
}

void RLInferenceNode::odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
  own_velocity_ = {msg->velocity[0], msg->velocity[1], msg->velocity[2]};
  have_odometry_ = true;
}

void RLInferenceNode::control_timer_callback()
{
  bool target_fresh;
  uav_common_msg::msg::TargetState state_copy;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    target_fresh = have_target_state_ &&
      (this->now() - last_target_state_time_).seconds() < watchdog_timeout_sec_;
    state_copy = latest_target_state_;
  }

  if (!target_fresh || !have_odometry_ || !state_copy.valid) {
    // No safe classical fallback (e.g. PNG) is wired in yet -- hold rather than
    // guess. If you add a PNG guidance package, this is the place to call it
    // instead of publish_hold().
    publish_hold();
    return;
  }

  std::array<float, kObsDim> obs{};
  obs[0] = static_cast<float>(state_copy.relative_position.x / pos_norm_);
  obs[1] = static_cast<float>(state_copy.relative_position.y / pos_norm_);
  obs[2] = static_cast<float>(state_copy.relative_position.z / pos_norm_);
  obs[3] = static_cast<float>(state_copy.relative_velocity.x / vel_norm_);
  obs[4] = static_cast<float>(state_copy.relative_velocity.y / vel_norm_);
  obs[5] = static_cast<float>(state_copy.relative_velocity.z / vel_norm_);
  obs[6] = static_cast<float>(own_velocity_[0] / vel_norm_);
  obs[7] = static_cast<float>(own_velocity_[1] / vel_norm_);
  obs[8] = static_cast<float>(own_velocity_[2] / vel_norm_);
  obs[9] = static_cast<float>(state_copy.range / pos_norm_);
  obs[10] = static_cast<float>(state_copy.closing_velocity / vel_norm_);
  obs[11] = 1.0f;

  if (state_copy.range <= capture_radius_m_) {
    RCLCPP_INFO(this->get_logger(), "Capture radius reached (range=%.2fm). Holding.", state_copy.range);
    publish_hold();
    return;
  }

  std::array<float, kActionDim> action{};
  run_policy(obs, action);
  publish_action(action);
}

void RLInferenceNode::run_policy(
  const std::array<float, kObsDim> & obs, std::array<float, kActionDim> & action_out)
{
  std::array<int64_t, 2> input_shape{1, kObsDim};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info_, const_cast<float *>(obs.data()), obs.size(),
    input_shape.data(), input_shape.size());

  const char * input_names[] = {input_name_.c_str()};
  const char * output_names[] = {output_name_.c_str()};

  auto output_tensors = ort_session_->Run(
    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  const float * out_data = output_tensors.front().GetTensorData<float>();
  for (int i = 0; i < kActionDim; ++i) {
    // Policy is trained with a tanh-squashed action space, so outputs should
    // already be in [-1, 1]; clamp defensively in case of extrapolation.
    action_out[i] = std::clamp(out_data[i], -1.0f, 1.0f);
  }
}

void RLInferenceNode::publish_hold()
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = "base_link_ned";
  cmd_vel_pub_->publish(msg);  // all-zero by default construction
}

void RLInferenceNode::publish_action(const std::array<float, kActionDim> & action)
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = "base_link_ned";
  msg.twist.linear.x = action[0] * max_speed_mps_;
  msg.twist.linear.y = action[1] * max_speed_mps_;
  msg.twist.linear.z = action[2] * max_speed_mps_;
  msg.twist.angular.z = action[3] * max_yaw_rate_rps_;
  cmd_vel_pub_->publish(msg);
}

}  // namespace uav_control
