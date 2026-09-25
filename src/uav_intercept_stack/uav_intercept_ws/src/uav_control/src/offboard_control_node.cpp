#include "uav_control/offboard_control.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<uav_control::OffboardControl>());
  rclcpp::shutdown();
  return 0;
}
