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
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
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
            "use_sim_time": "true",
            "rviz_config": "" + os.path.join(fairino_mpc_dir, "rviz", "mpc_avoidance.rviz"),
            "enable_rviz": "true",
            "robot_profile": profile["name"],
            "enable_camera_model": "false",
            "world": "empty",
        }.items(),
    )

    controller_topic = f'{profile["arm_controller"]}/joint_trajectory'

    obstacle_sim = TimerAction(period=3.0, actions=[
        Node(
            package="fairino_mpc_avoidance",
            executable="obstacle_simulator",
            name="obstacle_simulator",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "world_name": "empty",
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
                os.path.join(fairino_mpc_dir, "config", "mpc_params.yaml"),
                {
                    "use_sim_time": True,
                    "solver_type": "nmpc",
                    "group_name": profile["group_name"],
                    "joint_names": ",".join(profile["arm_joints"]),
                    "controller_topic": controller_topic,
                    "enable_moveit_scene": True,
                    "scenario_config": os.path.join(fairino_mpc_dir, "config", "obstacle_stack.yaml"),
                },
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
                {
                    "use_sim_time": True,
                    "ik_plugin": "fairino",
                    "planning_pipeline_id": "fairino",
                    "robot_profile": profile["name"],
                    "group_name": profile["group_name"],
                    "ee_link": profile["ee_frame_name"],
                    "base_frame": profile["planning_frame"],
                    "joint_names": profile["arm_joints"],
                    "controller_topic": controller_topic,
                    "planner_id": "tube_birrt*",
                },
            ],
        )
    ])

    return LaunchDescription([
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
