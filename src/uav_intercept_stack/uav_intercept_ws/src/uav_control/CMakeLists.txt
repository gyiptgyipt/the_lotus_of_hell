cmake_minimum_required(VERSION 3.8)
project(uav_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(uav_common_msg REQUIRED)

# --- ONNX Runtime -----------------------------------------------------------
# Point ONNXRUNTIME_ROOTDIR at wherever you extracted the onnxruntime release
# (https://github.com/microsoft/onnxruntime/releases -> onnxruntime-linux-x64-*.tgz)
# e.g. -DONNXRUNTIME_ROOTDIR=/opt/onnxruntime-linux-x64-1.18.0
if(NOT DEFINED ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "/usr/local")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOTDIR}/include ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
  REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS ${ONNXRUNTIME_ROOTDIR}/lib
  REQUIRED)
# -----------------------------------------------------------------------------

# offboard_control: PX4 interface, no ONNX dependency
add_executable(offboard_control_node
  src/offboard_control.cpp
  src/offboard_control_node.cpp
)
target_include_directories(offboard_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(offboard_control_node
  rclcpp px4_msgs geometry_msgs
)

# rl_inference_node: loads the exported ONNX policy
add_executable(rl_inference_node
  src/rl_inference_node.cpp
  src/rl_inference_main.cpp
)
target_include_directories(rl_inference_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(rl_inference_node ${ONNXRUNTIME_LIB})
ament_target_dependencies(rl_inference_node
  rclcpp px4_msgs geometry_msgs uav_common_msg
)

install(TARGETS
  offboard_control_node
  rl_inference_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
