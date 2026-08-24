"""Fairino 与 KDL 逆解对比的 Gazebo 演示入口."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_SCENE_DEFAULTS = {
    "robot_profile": "fairino_arm_gripper_onbase",
    "enable_rviz": "true",
    "world": "empty",
    "use_sim_time": "true",
    "enable_camera_model": "false",
}
_NODE_DEFAULTS = {
    "fairino_move_group_namespace": "/move_group_fairino",
    "kdl_move_group_namespace": "/move_group_kdl",
    "group_name": "robot_arm",
    "base_frame_name": "base_link",
    "ee_frame_name": "tool0",
    "joint_names": "j1,j2,j3,j4,j5,j6",
    "home_joints": "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0",
    "ik_timeout": "3.0",
    "execution_ik_plugin": "fairino",
    "execution_pipeline": "fairino",
    "planning_algorithm": "tube_birrt*",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in (*_SCENE_DEFAULTS, *_NODE_DEFAULTS)
}


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(name, default_value=value, description="Gazebo 场景参数。")
        for name, value in _SCENE_DEFAULTS.items()
    ] + [
        DeclareLaunchArgument(name, default_value=value, description="逆解测试节点参数。")
        for name, value in _NODE_DEFAULTS.items()
    ]


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **{name: _LAUNCH_CONFIGURATIONS[name] for name in _SCENE_DEFAULTS},
            "rviz_config": os.path.join(gz_share, "rviz", "ik_test.rviz"),
        }.items(),
    )
    ik_test_node = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg="[ik_test_demo] 启动 Fairino/KDL 逆解对比。"),
            Node(
                package="myrobot_simulation",
                executable="ik_test_node.py",
                name="ik_test_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    _NODE_DEFAULTS,
                    {name: _LAUNCH_CONFIGURATIONS[name] for name in _NODE_DEFAULTS},
                ],
            ),
        ],
    )
    return LaunchDescription([*_declare_launch_arguments(), myrobot_simulation, ik_test_node])
