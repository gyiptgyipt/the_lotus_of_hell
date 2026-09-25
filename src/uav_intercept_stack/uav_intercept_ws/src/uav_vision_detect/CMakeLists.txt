cmake_minimum_required(VERSION 3.8)
project(uav_vision_detect)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(uav_common_msg REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT DEFINED ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "/usr/local")
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOTDIR}/include ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
  REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS ${ONNXRUNTIME_ROOTDIR}/lib
  REQUIRED)

add_library(yolo_detector src/yolo_detector.cpp)
target_include_directories(yolo_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(yolo_detector ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

add_executable(uav_vision_detect
  src/vision_detect_node.cpp
  src/vision_detect_main.cpp
)
target_include_directories(uav_vision_detect PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(uav_vision_detect yolo_detector)
ament_target_dependencies(uav_vision_detect
  rclcpp sensor_msgs px4_msgs uav_common_msg cv_bridge OpenCV
)

install(TARGETS
  uav_vision_detect
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()
