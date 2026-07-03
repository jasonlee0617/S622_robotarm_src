import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


GAZEBO_LAUNCH_ARGUMENTS = {
    "robot_profile": "s622_gripper",
    "enable_rviz": "true",
    "world": "empty",
    "use_sim_time": "true",
    "enable_camera_model": "false",
}

IK_TEST_PARAMS = {
    "fairino_move_group_namespace": "/move_group_fairino",
    "kdl_move_group_namespace": "/move_group_kdl",
    "group_name": "robot_arm",
    "base_frame_name": "base_link",
    "ee_frame_name": "grasp_frame",
    "joint_names": "j1,j2,j3,j4,j5,j6",
    "home_joints": "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0",
    "ik_timeout": 3.0,
    "execution_ik_plugin": "fairino",
    "execution_pipeline": "fairino",
    "planning_algorithm": "tube_birrt*",
}


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **GAZEBO_LAUNCH_ARGUMENTS,
            "rviz_config": os.path.join(gz_share, "rviz", "ik_test.rviz"),
        }.items(),
    )

    ik_test_node = TimerAction(
        period=3.0,
        actions=[
            LogInfo(
                msg=(
                    "[ik_test_demo] ik=fairino, pipeline=fairino, "
                    "planner=birrt*"
                )
            ),
            Node(
                package="gazebo_launch",
                executable="ik_test_node.py",
                name="ik_test_node",
                output="screen",
                emulate_tty=True,
                parameters=[IK_TEST_PARAMS],
            ),
        ],
    )

    return LaunchDescription([gazebo_launch, ik_test_node])
