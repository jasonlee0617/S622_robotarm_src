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
  scripts/vision_quality_gate.py
  scripts/sample_manager.py
  scripts/calibration_validator.py
  scripts/collector_config.py
  scripts/collector_geometry.py
  scripts/collector_execution.py
  scripts/auto_calibration_collector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
