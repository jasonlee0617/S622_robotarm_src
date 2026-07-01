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
        remove_obstacle_after_demo="false",
        shutdown_on_demo_exit="true",
    )


def _benchmark_args():
    """Single-algorithm benchmark parameters only."""
    return [
        _arg("benchmark_repetitions", "20", "Number of repetitions for the single planner under test."),
        _arg("benchmark_start_pose", "", "Reference start pose for goal sampling/separation as x,y,z[,rx,ry,rz]."),
        _arg("benchmark_goal_pose", "", "Benchmark goal pose as x,y,z[,rx,ry,rz]."),
        _arg("benchmark_result_csv", "", "Optional CSV output path for benchmark results."),
        _arg("benchmark_case_label", "", "Optional label written into benchmark results."),
        _arg("benchmark_goal_mode", "random_obstacle_envelope", "fixed, random_obstacle_envelope, or random_pose_goal_region."),
        _arg("benchmark_goal_seed", "17", "Random goal seed."),
        _arg("benchmark_goal_clearance_min_m", "0.06", "Minimum goal obstacle clearance."),
        _arg("benchmark_goal_clearance_max_m", "0.14", "Maximum goal obstacle clearance."),
        _arg("benchmark_goal_min_separation_m", "0.04", "Minimum start/goal and goal/goal separation."),
        _arg("benchmark_goal_max_attempts_per_sample", "200", "Max random sampling attempts per goal candidate."),
        _arg("benchmark_goal_region_min", "", "x,y,z lower bound for random_pose_goal_region."),
        _arg("benchmark_goal_region_max", "", "x,y,z upper bound for random_pose_goal_region."),
        _arg("benchmark_goal_state_validity_timeout_s", "2.0", "Timeout for each sampled goal's MoveIt state-validity check."),
        _arg("benchmark_startup_joint_state_timeout_s", "90.0", "Maximum wait for initial joint state before starting benchmark."),
        _arg("execute_planned_trajectory", "false", "If true, execute each successfully planned trajectory on the controller."),
        _arg("go_home_before_benchmark", "false", "If true, move to HOME once before pure-planning benchmark runs."),
    ]


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    trajectory_plan_test_node = Node(
        package="gazebo_launch",
        executable="trajectory_plan_test_node.py",
        name="trajectory_plan_test_node",
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
            "default_pipeline_id": LaunchConfiguration("planning_pipeline"),
            "default_planner_id": LaunchConfiguration("planning_algorithm"),
            "target_rpy_deg": LaunchConfiguration("target_rpy_deg"),
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
            "benchmark_repetitions": LaunchConfiguration("benchmark_repetitions"),
            "benchmark_start_pose": LaunchConfiguration("benchmark_start_pose"),
            "benchmark_goal_pose": LaunchConfiguration("benchmark_goal_pose"),
            "benchmark_result_csv": LaunchConfiguration("benchmark_result_csv"),
            "benchmark_case_label": LaunchConfiguration("benchmark_case_label"),
            "benchmark_goal_mode": LaunchConfiguration("benchmark_goal_mode"),
            "benchmark_goal_seed": LaunchConfiguration("benchmark_goal_seed"),
            "benchmark_goal_clearance_min_m": LaunchConfiguration("benchmark_goal_clearance_min_m"),
            "benchmark_goal_clearance_max_m": LaunchConfiguration("benchmark_goal_clearance_max_m"),
            "benchmark_goal_min_separation_m": LaunchConfiguration("benchmark_goal_min_separation_m"),
            "benchmark_goal_max_attempts_per_sample": LaunchConfiguration("benchmark_goal_max_attempts_per_sample"),
            "benchmark_goal_region_min": LaunchConfiguration("benchmark_goal_region_min"),
            "benchmark_goal_region_max": LaunchConfiguration("benchmark_goal_region_max"),
            "benchmark_goal_state_validity_timeout_s": LaunchConfiguration("benchmark_goal_state_validity_timeout_s"),
            "benchmark_startup_joint_state_timeout_s": LaunchConfiguration("benchmark_startup_joint_state_timeout_s"),
            "execute_planned_trajectory": LaunchConfiguration("execute_planned_trajectory"),
            "go_home_before_benchmark": LaunchConfiguration("go_home_before_benchmark"),
        }],
    )

    delayed_node = TimerAction(
        period=LaunchConfiguration("demo_start_delay_s"),
        actions=[
            LogInfo(msg=[
                "[trajectory_plan_test] client=", LaunchConfiguration("ik_plugin"),
                ", namespace_override=", LaunchConfiguration("planning_move_group_namespace"),
                ", pipeline=", LaunchConfiguration("planning_pipeline"),
                ", planner=", LaunchConfiguration("planning_algorithm"),
                ", pre_home=", LaunchConfiguration("go_home_before_benchmark"),
            ]),
            trajectory_plan_test_node,
        ],
    )
    shutdown_when_node_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=trajectory_plan_test_node,
            on_exit=[
                LogInfo(msg="[trajectory_plan_test] trajectory_plan_test_node exited; shutting down launch."),
                EmitEvent(event=Shutdown(reason="trajectory plan test completed")),
            ],
        ),
        condition=IfCondition(LaunchConfiguration("shutdown_on_demo_exit")),
    )

    return LaunchDescription(_base_args(gz_share) + _benchmark_args() + [
        _gazebo_launch(gz_share),
        delayed_node,
        shutdown_when_node_exits,
    ])
