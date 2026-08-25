cmake_minimum_required(VERSION 3.8)
project(trajectory_retime_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
ament_package_xml()
list(APPEND "${PROJECT_NAME}_MEMBER_OF_GROUPS" "rosidl_interface_packages")
# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/RetimeTrajectory.srv"
  DEPENDENCIES trajectory_msgs
)
ament_export_dependencies(rosidl_default_runtime)

# ---- C++ node ----
add_executable(retime_server src/retime_server.cpp)
ament_target_dependencies(retime_server
  rclcpp
  trajectory_msgs
  moveit_core
  moveit_ros_planning
  moveit_msgs
)

# Link generated service typesupport
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(retime_server "${cpp_typesupport_target}")

install(TARGETS retime_server
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()