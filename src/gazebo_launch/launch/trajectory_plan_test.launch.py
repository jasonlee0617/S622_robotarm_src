import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _arg(name, default_value, description=""):
    return DeclareLaunchArgument(name, default_value=default_value, description=description)


def _base_args(gz_share):
    return [
        _arg("robot_profile", "s622_gripper", "Robot profile used by gazebo.launch.py."),
        _arg("rviz_config", os.path.join(gz_share, "rviz", "fairino_planning_test.rviz"), "RViz config for Fairino trajectory planning."),
        _arg("enable_rviz", "true"),
        _arg("world", "empty"),
        _arg("use_sim_time", "true"),
        _arg("demo_start_delay_s", "12.0", "Delay node start until Gazebo and controllers are ready."),
        _arg("initial_positions_file", "", "Optional Gazebo/ros2_control initial_positions YAML override."),
        _arg("ik_plugin", "fairino", "Planning client in dual move_group setup: fairino or kdl."),
        _arg("planning_move_group_namespace", "", "Optional explicit move_group namespace override."),
        _arg("planning_pipeline", "fairino", "Planning pipeline: fairino or ompl."),
        _arg("planning_algorithm", "aapf_birrt*", "Planner id: birrt*, rrt*, aapf_birrt*, RRTConnect, etc."),
        _arg("group_name", "robot_arm"),
        _arg("base_frame_name", "base_link"),
        _arg("ee_frame_name", "grasp_frame"),
        _arg("joint_names", "j1,j2,j3,j4,j5,j6"),
        _arg("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0"),
        _arg("target_rpy_deg", "0,-180,0", "Fixed end-effector orientation as roll,pitch,yaw in degrees."),
        _arg("auto_add_obstacle", "true", "Whether to add the default PlanningScene obstacle at start."),
        _arg("remove_obstacle_after_demo", "true", "Whether to remove demo-created PlanningScene obstacles on exit."),
        _arg("obstacle_name", "birrt_test_obstacle"),
        _arg("obstacle_position", "0.35,0.05,0.28", "Default obstacle center xyz in base frame."),
        _arg("obstacle_size", "0.18,0.45,0.35", "Default obstacle size sx,sy,sz."),
        _arg("obstacle_boxes", "", "Optional boxes: name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz."),
        _arg("scene_assets_dir", os.path.join(gz_share, "config", "scenes"), "Directory containing scene URDF/SDF assets."),
        _arg("scene_config_file", os.path.join(gz_share, "config", "scenes", "pathplanning_scenes.yaml"), "YAML file containing named path-planning scenes."),
        _arg("scene_name", "paper_simple_3d_avoidance", "Scene key in pathplanning_scenes.yaml."),
        _arg("spawn_gazebo_scene_models", "true", "If true, also spawn scene assets into Gazebo."),
        _arg("publish_planning_scene", "true", "If true, publish scene obstacles into MoveIt PlanningScene."),
        _arg("publish_obstacle_markers", "true", "If true, publish RViz obstacle markers for the selected scene."),
        _arg("obstacle_marker_topic", "/demo_pathplanning/obstacle_markers", "MarkerArray topic for planning scene obstacles."),
        _arg("planning_scene_obstacle_padding_m", "0.03", "Padding applied only to MoveIt collision objects."),
        _arg("shutdown_on_demo_exit", "false", "If true, shut down launch when the trajectory plan node exits."),
    ]


def _gazebo_launch(gz_share):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            "robot_profile": LaunchConfiguration("robot_profile"),
            "rviz_config": LaunchConfiguration("rviz_config"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "world": LaunchConfiguration("world"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "initial_positions_file": LaunchConfiguration("initial_positions_file"),
            "default_planning_pipeline": LaunchConfiguration("planning_pipeline"),
            "enable_camera_model": "false",
            "scene_assets_dir": LaunchConfiguration("scene_assets_dir"),
            "scene_config_file": LaunchConfiguration("scene_config_file"),
            "scene_name": LaunchConfiguration("scene_name"),
            "spawn_gazebo_scene_models": LaunchConfiguration("spawn_gazebo_scene_models"),
            "publish_planning_scene": LaunchConfiguration("publish_planning_scene"),
            "publish_obstacle_markers": LaunchConfiguration("publish_obstacle_markers"),
            "obstacle_marker_topic": LaunchConfiguration("obstacle_marker_topic"),
        }.items(),
    )


def _benchmark_args():
    return [
        _arg("benchmark_planners", "", "Comma-separated planner ids, e.g. aapf_birrt*,birrt*."),
        _arg("benchmark_repetitions", "1", "Number of repetitions per planner."),
        _arg("benchmark_start_pose", "", "Benchmark start pose as x,y,z[,rx,ry,rz]."),
        _arg("benchmark_goal_pose", "", "Benchmark goal pose as x,y,z[,rx,ry,rz]."),
        _arg("benchmark_result_csv", "", "Optional CSV output path for benchmark results."),
        _arg("benchmark_case_label", "", "Optional label written into benchmark results."),
        _arg("benchmark_notes", "", "Optional notes written into benchmark results."),
        _arg("benchmark_go_home_each_run", "true", "If true, benchmark runs return HOME before each repetition."),
        _arg("benchmark_reset_scene_each_run", "true", "If true, benchmark runs reload the PlanningScene before each repetition."),
        _arg("benchmark_move_to_start_each_run", "true", "If true, benchmark runs execute HOME -> start_pose before planning to goal."),
        _arg("benchmark_setup_planner_id", "birrt*", "Planner used for HOME->start leg."),
        _arg("benchmark_use_controller_reset_for_home", "false", "Legacy HOME reset switch."),
        _arg("benchmark_home_reset_mode", "planner", "HOME reset mode: planner or controller_trajectory."),
        _arg("benchmark_home_planner_id", "birrt*", "Planner used for collision-aware HOME reset."),
        _arg("benchmark_home_fallback_planner_id", "none", "Optional planner tried after primary HOME planner retries."),
        _arg("benchmark_home_settle_timeout_s", "6.0", "Maximum post-execution HOME convergence timeout."),
        _arg("benchmark_home_retry_count", "2", "Additional HOME attempts before a benchmark failure."),
        _arg("benchmark_abort_on_home_reset_failure", "true", "Abort after HOME reset failure."),
        _arg("benchmark_record_phase_times", "true", "If true, record per-phase timing."),
        _arg("benchmark_action_delay_s", "0.0", "Post-action sleep during benchmark."),
        _arg("benchmark_pair_planners_by_goal", "true", "Run all planners for each generated goal before moving on."),
        _arg("benchmark_goal_mode", "random_obstacle_envelope", "fixed, random_obstacle_envelope, or random_pose_goal_region."),
        _arg("benchmark_goal_seed", "17", "Random goal seed."),
        _arg("benchmark_goal_clearance_min_m", "0.06", "Minimum goal obstacle clearance."),
        _arg("benchmark_goal_clearance_max_m", "0.14", "Maximum goal obstacle clearance."),
        _arg("benchmark_goal_min_separation_m", "0.04", "Minimum start/goal and goal/goal separation."),
        _arg("benchmark_goal_max_attempts_per_sample", "200", "Max random sampling attempts per goal candidate."),
        _arg("benchmark_goal_region_min", "", "x,y,z lower bound for random_pose_goal_region."),
        _arg("benchmark_goal_region_max", "", "x,y,z upper bound for random_pose_goal_region."),
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
            "benchmark_planners": LaunchConfiguration("benchmark_planners"),
            "benchmark_repetitions": LaunchConfiguration("benchmark_repetitions"),
            "benchmark_start_pose": LaunchConfiguration("benchmark_start_pose"),
            "benchmark_goal_pose": LaunchConfiguration("benchmark_goal_pose"),
            "benchmark_result_csv": LaunchConfiguration("benchmark_result_csv"),
            "benchmark_case_label": LaunchConfiguration("benchmark_case_label"),
            "benchmark_notes": LaunchConfiguration("benchmark_notes"),
            "benchmark_go_home_each_run": LaunchConfiguration("benchmark_go_home_each_run"),
            "benchmark_reset_scene_each_run": LaunchConfiguration("benchmark_reset_scene_each_run"),
            "benchmark_move_to_start_each_run": LaunchConfiguration("benchmark_move_to_start_each_run"),
            "benchmark_setup_planner_id": LaunchConfiguration("benchmark_setup_planner_id"),
            "benchmark_use_controller_reset_for_home": LaunchConfiguration("benchmark_use_controller_reset_for_home"),
            "benchmark_home_reset_mode": LaunchConfiguration("benchmark_home_reset_mode"),
            "benchmark_home_planner_id": LaunchConfiguration("benchmark_home_planner_id"),
            "benchmark_home_fallback_planner_id": LaunchConfiguration("benchmark_home_fallback_planner_id"),
            "benchmark_home_settle_timeout_s": LaunchConfiguration("benchmark_home_settle_timeout_s"),
            "benchmark_home_retry_count": LaunchConfiguration("benchmark_home_retry_count"),
            "benchmark_abort_on_home_reset_failure": LaunchConfiguration("benchmark_abort_on_home_reset_failure"),
            "benchmark_record_phase_times": LaunchConfiguration("benchmark_record_phase_times"),
            "benchmark_action_delay_s": LaunchConfiguration("benchmark_action_delay_s"),
            "benchmark_pair_planners_by_goal": LaunchConfiguration("benchmark_pair_planners_by_goal"),
            "benchmark_goal_mode": LaunchConfiguration("benchmark_goal_mode"),
            "benchmark_goal_seed": LaunchConfiguration("benchmark_goal_seed"),
            "benchmark_goal_clearance_min_m": LaunchConfiguration("benchmark_goal_clearance_min_m"),
            "benchmark_goal_clearance_max_m": LaunchConfiguration("benchmark_goal_clearance_max_m"),
            "benchmark_goal_min_separation_m": LaunchConfiguration("benchmark_goal_min_separation_m"),
            "benchmark_goal_max_attempts_per_sample": LaunchConfiguration("benchmark_goal_max_attempts_per_sample"),
            "benchmark_goal_region_min": LaunchConfiguration("benchmark_goal_region_min"),
            "benchmark_goal_region_max": LaunchConfiguration("benchmark_goal_region_max"),
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
