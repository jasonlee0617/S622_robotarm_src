cmake_minimum_required(VERSION 3.5)
project(hand_eye_calibration)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(ros2_aruco_interfaces REQUIRED)

ament_python_install_package(hand_eye_calibration)

install(DIRECTORY launch config rviz docs
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install (PROGRAMS
  scripts/calibration_aruco_publisher.py
  scripts/evaluate_calibration.py
  scripts/follow_aruco_marker.py
  scripts/handeye_publisher.py
  scripts/visualize_aruco_marker.py
  scripts/capture_d435_profile.py
  hand_eye_calibration/collector/auto_calibration_collector.py
  scripts/collect_auto_calibration_diagnostics.sh
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_auto_calibration_collector test/test_auto_calibration_collector.py)
  ament_add_pytest_test(test_session_finalize test/test_session_finalize.py)
endif()

ament_package()
