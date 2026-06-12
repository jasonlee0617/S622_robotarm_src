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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(os.path.join(PKG_ROOT_DIR, "utils"))
from demo_models import load_spawn_config, build_spawn_actions  # noqa: E402
from robot_profile_loader import load_demo_robot_profile  # noqa: E402


def _build_nodes(context):
    use_sim_time = LaunchConfiguration("use_sim_time")
    planning_client = LaunchConfiguration("planning_client")
    planning_move_group_namespace = LaunchConfiguration("planning_move_group_namespace")
    planner_id = LaunchConfiguration("planner_id")
    robot_profile_name = LaunchConfiguration("robot_profile").perform(context).strip()
    fairino_mpc_dir = get_package_share_directory("fairino_mpc_avoidance")
    gz_launch_dir = get_package_share_directory("gazebo_launch")
    profile = load_demo_robot_profile(robot_profile_name)
    spawn_cfg = load_spawn_config(os.path.join(fairino_mpc_dir, "config", "obstacle_stack.yaml"))
    spawn_actions = build_spawn_actions(fairino_mpc_dir, spawn_cfg)

    moveit_config = MoveItConfigsBuilder(
        profile.moveit_config_name,
        package_name=profile.moveit_config_package,
    ).to_moveit_configs()

    gazebo_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_launch_dir, "launch", "gazebo.launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "default_planning_pipeline": "fairino",
            "rviz_config": LaunchConfiguration("rviz_config"),
            "robot_profile": TextSubstitution(text=robot_profile_name),
            "enable_camera_model": "false",
            "world": LaunchConfiguration("world"),
        }.items(),
    )

    controller_topic = f"{profile.arm_controller}/joint_trajectory"
    demo_common_params = {
        "use_sim_time": use_sim_time,
        "planning_client": planning_client,
        "move_group_namespace": planning_move_group_namespace,
        "robot_profile": profile.name,
        "group_name": profile.group_name,
        "ee_link": profile.ee_frame_name,
        "base_frame": profile.planning_frame,
        "joint_names": profile.arm_joints,
        "controller_topic": controller_topic,
        "planner_id": planner_id,
    }

    obstacle_sim = TimerAction(period=3.0, actions=[
        Node(
            package="fairino_mpc_avoidance",
            executable="obstacle_simulator",
            name="obstacle_simulator",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
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
                    "use_sim_time": use_sim_time,
                    "group_name": profile.group_name,
                    "joint_names": profile.arm_joints,
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
                demo_common_params,
            ],
        )
    ])

    return [
        LogInfo(msg=[
            "[mpc_planning_demo] robot_profile=",
            TextSubstitution(text=profile.name),
            ", moveit_pkg=",
            TextSubstitution(text=profile.moveit_config_package),
            ", group=",
            TextSubstitution(text=profile.group_name),
            ", ee_link=",
            TextSubstitution(text=profile.ee_frame_name),
            ", controller_topic=",
            TextSubstitution(text=controller_topic),
        ]),
        gazebo_moveit_launch,
        *spawn_actions,
        obstacle_sim,
        mpc_node,
        demo_node,
    ]


def generate_launch_description():
    planning_client = LaunchConfiguration("planning_client")
    planning_move_group_namespace = LaunchConfiguration("planning_move_group_namespace")
    planner_id = LaunchConfiguration("planner_id")
    fairino_mpc_avoidance_share = get_package_share_directory("fairino_mpc_avoidance")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "robot_profile",
            default_value="s622_gripper",
            description="Robot profile name from gazebo_launch/config/robots/*.yaml",
        ),
        DeclareLaunchArgument(
            "world",
            default_value="empty",
            description="Gazebo world passed through to gazebo.launch.py.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(fairino_mpc_avoidance_share, "rviz", "fairino_planning_test.rviz"),
            description="RViz config passed through to gazebo.launch.py.",
        ),
        DeclareLaunchArgument(
            "enable_rviz",
            default_value="true",
            description="Whether gazebo.launch.py should start RViz.",
        ),
        DeclareLaunchArgument(
            "planning_client",
            default_value="fairino",
            description="MoveGroup client to use: fairino or kdl.",
        ),
        DeclareLaunchArgument(
            "planning_move_group_namespace",
            default_value="",
            description="Optional namespace override, e.g. /move_group_fairino.",
        ),
        DeclareLaunchArgument(
            "planner_id",
            default_value="birrt*",
            description="Fairino planner id. Use aapf_birrt*, birrt*, or rrt*.",
        ),
        LogInfo(msg=[
            "[mpc_planning_demo] planning_client=",
            planning_client,
            ", planning_move_group_namespace=",
            planning_move_group_namespace,
            ", planner_id=",
            planner_id,
        ]),
        OpaqueFunction(function=_build_nodes),
    ])
