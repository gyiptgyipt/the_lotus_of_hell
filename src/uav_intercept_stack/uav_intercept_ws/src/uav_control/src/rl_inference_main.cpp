#include "uav_control/rl_inference_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<uav_control::RLInferenceNode>());
  rclcpp::shutdown();
  return 0;
}
