cmake_minimum_required(VERSION 3.5)
project(hand_eye_calibration)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(ros2_aruco_interfaces REQUIRED)

install(DIRECTORY launch config rviz docs
  DESTINATION share/${PROJECT_NAME}
)

install (PROGRAMS
  scripts/calibration_aruco_publisher.py
  scripts/evaluate_calibration.py
  scripts/follow_aruco_marker.py
  scripts/handeye_publisher.py
  scripts/visualize_aruco_marker.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
