cmake_minimum_required(VERSION 3.8)
project(llm_arm_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ControlPose.srv"
  "srv/PreviewCommand.srv"
  "action/ExecutePreview.action"
  DEPENDENCIES geometry_msgs trajectory_msgs
)

ament_python_install_package(llm_arm_control_nodes)

install(PROGRAMS llm_arm_control_nodes/robot_pose_monitor_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME robot_pose_monitor_node)
install(PROGRAMS llm_arm_control_nodes/robot_pose_control_server.py
  DESTINATION lib/${PROJECT_NAME} RENAME robot_pose_control_server)
install(PROGRAMS llm_arm_control_nodes/llm_control_task_server.py
  DESTINATION lib/${PROJECT_NAME} RENAME llm_control_task_server)
install(PROGRAMS llm_arm_control_nodes/llm_control_cli.py
  DESTINATION lib/${PROJECT_NAME} RENAME llm_control_cli)

install(
  DIRECTORY config rviz launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  set(ENV{PYTHONNOUSERSITE} "1")
  foreach(test_name
      test_deepseek_client
      test_deepseek_credentials
      test_robot_pose_control_server
      test_llm_control_cli
      test_llm_robot_control_sim_launch
      test_llm_control_state_machine
      test_llm_control_task_server_robustness
      test_shutdown
      test_task_logic)
    ament_add_pytest_test(
      ${test_name}
      test/${test_name}.py
      PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
      ENV
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
        "PYTHONNOUSERSITE=1"
    )
  endforeach()
endif()

ament_package()
