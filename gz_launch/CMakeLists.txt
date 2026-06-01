cmake_minimum_required(VERSION 3.8)
project(gz_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    scripts
    launch
    launch_utils
    config
    rviz
    worlds
    docs
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
  scripts/pick_drop_ik_node.py
  scripts/pick_drop_node.py
  scripts/robot_control_from_UI_node.py
  scripts/yolo_detector_obb_node.py
  scripts/stopmotion_node.py
  scripts/pen_box_grasping_node.py
  scripts/cube_controller_node.py
  scripts/yolo_Kalman_detector_obb_node.py
  scripts/demo_pathplanning_node.py
  scripts/ik_teset_node.py

  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
