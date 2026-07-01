import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    EmitEvent,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.trajectory_launch import arg as _arg
from launch_utils.trajectory_launch import base_args as _shared_base_args
from launch_utils.trajectory_launch import gazebo_launch as _gazebo_launch


def _base_args(gz_share):
    return _shared_base_args(
        gz_share,
        remove_obstacle_after_demo="true",
        shutdown_on_demo_exit="false",
    ) + [
        _arg("go_home_before_demo", "false", "If true, move to HOME before accepting interactive input."),
        _arg("home_settle_timeout_s", "6.0", "HOME convergence timeout for interactive go home."),
    ]


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    trajectory_plan_node = Node(
        package="gazebo_launch",
        executable="trajectory_plan_node.py",
        name="trajectory_plan_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "planning_client": LaunchConfiguration("ik_plugin"),
            "move_group_namespace": LaunchConfiguration("planning_move_group_namespace"),
            "group_name": LaunchConfiguration("group_name"),
            "base_frame_name": LaunchConfiguration("base_frame_name"),
            "ee_frame_name": LaunchConfiguration("ee_frame_name"),
            "joint_names": LaunchConfiguration("joint_names"),
            "home_joints": LaunchConfiguration("home_joints"),
            "home_settle_timeout_s": LaunchConfiguration("home_settle_timeout_s"),
            "default_pipeline_id": LaunchConfiguration("planning_pipeline"),
            "default_planner_id": LaunchConfiguration("planning_algorithm"),
            "target_rpy_deg": LaunchConfiguration("target_rpy_deg"),
            "go_home_before_demo": LaunchConfiguration("go_home_before_demo"),
            "auto_add_obstacle": LaunchConfiguration("auto_add_obstacle"),
            "remove_obstacle_after_demo": LaunchConfiguration("remove_obstacle_after_demo"),
            "obstacle_name": LaunchConfiguration("obstacle_name"),
            "obstacle_position": LaunchConfiguration("obstacle_position"),
            "obstacle_size": LaunchConfiguration("obstacle_size"),
            "obstacle_boxes": LaunchConfiguration("obstacle_boxes"),
            "scene_assets_dir": LaunchConfiguration("scene_assets_dir"),
            "scene_config_file": LaunchConfiguration("scene_config_file"),
            "scene_name": LaunchConfiguration("scene_name"),
            "spawn_gazebo_scene_models": LaunchConfiguration("spawn_gazebo_scene_models"),
            "gazebo_world": LaunchConfiguration("world"),
            "publish_planning_scene": LaunchConfiguration("publish_planning_scene"),
            "publish_obstacle_markers": LaunchConfiguration("publish_obstacle_markers"),
            "obstacle_marker_topic": LaunchConfiguration("obstacle_marker_topic"),
            "planning_scene_obstacle_padding_m": LaunchConfiguration("planning_scene_obstacle_padding_m"),
        }],
    )

    delayed_node = TimerAction(
        period=LaunchConfiguration("demo_start_delay_s"),
        actions=[
            LogInfo(msg=[
                "[trajectory_plan_demo] client=", LaunchConfiguration("ik_plugin"),
                ", namespace_override=", LaunchConfiguration("planning_move_group_namespace"),
                ", pipeline=", LaunchConfiguration("planning_pipeline"),
                ", planner=", LaunchConfiguration("planning_algorithm"),
            ]),
            trajectory_plan_node,
        ],
    )
    shutdown_when_node_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=trajectory_plan_node,
            on_exit=[
                LogInfo(msg="[trajectory_plan_demo] trajectory_plan_node exited; shutting down launch."),
                EmitEvent(event=Shutdown(reason="trajectory plan demo completed")),
            ],
        ),
        condition=IfCondition(LaunchConfiguration("shutdown_on_demo_exit")),
    )

    return LaunchDescription(_base_args(gz_share) + [
        _gazebo_launch(gz_share),
        delayed_node,
        shutdown_when_node_exits,
    ])
