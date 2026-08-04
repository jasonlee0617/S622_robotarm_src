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
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Tests import this package plus source-tree workspace dependencies.
  set(_test_pythonpath "${CMAKE_CURRENT_SOURCE_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../../manipulation_common")
  ament_add_pytest_test(test_quality test/test_quality.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_config test/test_config.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_session test/test_session.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_solver test/test_solver.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
endif()

ament_package()
