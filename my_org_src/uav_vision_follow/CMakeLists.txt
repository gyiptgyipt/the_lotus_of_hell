cmake_minimum_required(VERSION 3.16)
project(uav_vision_follow)

find_package(ament_cmake REQUIRED)

# This is a Python ROS node, installed by CMake rather than setuptools.
install(PROGRAMS
  uav_vision_follow/vision_follower.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME vision_follower
)

install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
