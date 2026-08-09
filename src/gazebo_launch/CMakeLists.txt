cmake_minimum_required(VERSION 3.8)
project(gazebo_launch)

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
  find_package(ament_cmake_pytest REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(test_d435_profile test/test_d435_profile.py)
  ament_add_pytest_test(test_visual_grasping_gazebo_launch test/test_visual_grasping_gazebo_launch.py)
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
  scripts/pick_drop_node.py
  scripts/cube_controller_node.py
  scripts/trajectory_plan_node.py
  scripts/trajectory_plan_test_node.py
  scripts/ik_test_node.py

  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
  scripts/pathplanning_scene_tools.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
