#!/usr/bin/env python3
"""Demo launch orchestration for MPC avoidance.

This launch file intentionally only does orchestration:
1) include Gazebo+MoveIt,
2) spawn models from external config,
3) start obstacle simulator, MPC node, and demo node with delay.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(os.path.join(PKG_ROOT_DIR, "utils"))
from demo_models import load_spawn_config, build_spawn_actions  # noqa: E402


ROBOT_PROFILE = {
    "name": "fairino_arm_gripper",
    "moveit_config_name": "fairino_arm",
    "moveit_config_package": "fairino_arm_moveit_config",
    "group_name": "robot_arm",
    "ee_frame_name": "grasp_frame",
    "planning_frame": "base_link",
    "arm_controller": "/robot_arm_controller",
    "arm_joints": ["j1", "j2", "j3", "j4", "j5", "j6"],
}

# 本演示的场景与节点默认值集中定义；配置、RViz 与障碍物 YAML 均为包内固定资源。
_LAUNCH_ARGUMENT_SPECS = {
    "use_sim_time": ("true", "是否使用仿真时间。"),
    "world_name": ("empty", "Gazebo 世界名称。"),
    "solver_type": ("nmpc", "MPC 求解器类型。"),
    "planner_id": ("tube_birrt*", "MoveIt 规划器名称。"),
    "enable_rviz": ("true", "是否启动 RViz。"),
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_ARGUMENT_SPECS
}
_USE_SIM_TIME = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
_OBSTACLE_NODE_DEFAULTS = {"use_sim_time": True, "world_name": "empty"}
_MPC_NODE_DEFAULTS = {"use_sim_time": True, "solver_type": "nmpc"}
_DEMO_NODE_DEFAULTS = {"use_sim_time": True, "planner_id": "tube_birrt*"}


def _declare_launch_arguments():
    """声明可由命令行覆盖的场景和节点标量参数。"""
    return [
        DeclareLaunchArgument(name, default_value=value, description=description)
        for name, (value, description) in _LAUNCH_ARGUMENT_SPECS.items()
    ]


def generate_launch_description():
    fairino_mpc_dir = get_package_share_directory("fairino_mpc_avoidance")
    gz_launch_dir = get_package_share_directory("gazebo_launch")
    profile = ROBOT_PROFILE
    spawn_cfg = load_spawn_config(os.path.join(fairino_mpc_dir, "config", "obstacle_stack.yaml"))
    spawn_actions = build_spawn_actions(fairino_mpc_dir, spawn_cfg)

    moveit_config = MoveItConfigsBuilder(
        profile["moveit_config_name"],
        package_name=profile["moveit_config_package"],
    ).to_moveit_configs()

    gazebo_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_launch_dir, "launch", "gazebo.launch.py")),
        launch_arguments={
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            "rviz_config": "" + os.path.join(fairino_mpc_dir, "rviz", "mpc_avoidance.rviz"),
            "enable_rviz": _LAUNCH_CONFIGURATIONS["enable_rviz"],
            "robot_profile": profile["name"],
            "enable_camera_model": "false",
            "world": _LAUNCH_CONFIGURATIONS["world_name"],
        }.items(),
    )

    controller_topic = f'{profile["arm_controller"]}/joint_trajectory'

    obstacle_sim = TimerAction(period=3.0, actions=[
        Node(
            package="fairino_mpc_avoidance",
            executable="obstacle_simulator",
            name="obstacle_simulator",
            output="screen",
            parameters=[_OBSTACLE_NODE_DEFAULTS, {
                "use_sim_time": _USE_SIM_TIME,
                "world_name": _LAUNCH_CONFIGURATIONS["world_name"],
                "scenario_config": os.path.join(fairino_mpc_dir, "config", "obstacle_stack.yaml"),
            }],
        )
    ])

    mpc_node = TimerAction(period=5.0, actions=[
        Node(
            package="fairino_mpc_avoidance",
            executable="mpc_avoidance_node",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                _MPC_NODE_DEFAULTS,
                {
                    "use_sim_time": _USE_SIM_TIME,
                    "solver_type": _LAUNCH_CONFIGURATIONS["solver_type"],
                    "group_name": profile["group_name"],
                    "joint_names": ",".join(profile["arm_joints"]),
                    "controller_topic": controller_topic,
                    "enable_moveit_scene": True,
                    "scenario_config": os.path.join(fairino_mpc_dir, "config", "obstacle_stack.yaml"),
                },
                # YAML 最后加载：CLI > YAML > launch 覆盖 > 节点直接默认值。
                os.path.join(fairino_mpc_dir, "config", "mpc_params.yaml"),
            ],
        )
    ])

    demo_node = TimerAction(period=12.0, actions=[
        Node(
            package="fairino_mpc_avoidance",
            executable="demo_mpc_avoidance_node.py",
            name="mpc_avoidance_demo",
            output="screen",
            parameters=[
                moveit_config.robot_description_semantic,
                _DEMO_NODE_DEFAULTS,
                {
                    "use_sim_time": _USE_SIM_TIME,
                    "ik_plugin": "fairino",
                    "planning_pipeline_id": "fairino",
                    "robot_profile": profile["name"],
                    "group_name": profile["group_name"],
                    "ee_link": profile["ee_frame_name"],
                    "base_frame": profile["planning_frame"],
                    "joint_names": profile["arm_joints"],
                    "controller_topic": controller_topic,
                    "planner_id": _LAUNCH_CONFIGURATIONS["planner_id"],
                },
            ],
        )
    ])

    return LaunchDescription([
        *_declare_launch_arguments(),
        LogInfo(msg=[
            f'[mpc_planning_demo] robot_profile={profile["name"]}',
            f', moveit_pkg={profile["moveit_config_package"]}',
            f', group={profile["group_name"]}',
            f', ee_link={profile["ee_frame_name"]}',
            f', controller_topic={controller_topic}',
        ]),
        gazebo_moveit_launch,
        *spawn_actions,
        obstacle_sim,
        mpc_node,
        demo_node,
    ])
