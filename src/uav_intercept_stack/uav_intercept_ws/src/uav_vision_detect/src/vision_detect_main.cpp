#include "uav_vision_detect/vision_detect_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<uav_vision_detect::VisionDetectNode>());
  rclcpp::shutdown();
  return 0;
}
