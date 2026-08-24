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
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


RESULT_CSV = "/tmp/trajectory_plan_test_node_results.csv"

GAZEBO_LAUNCH_ARGUMENTS = {
    "robot_profile": "fairino_arm_gripper_onbase",
    "enable_rviz": "false",
    "world": "empty",
    "use_sim_time": "true",
    "initial_positions_file": "",
    "enable_camera_model": "false",
    "scene_name": "multi_obstacle_3d_avoidance",
    "spawn_gazebo_scene_models": "true",
    "publish_planning_scene": "true",
    "publish_obstacle_markers": "true",
    "obstacle_marker_topic": "/demo_pathplanning/obstacle_markers",
}

NODE_PARAMS = {
    "planning_client": "fairino",
    "move_group_namespace": "",
    "group_name": "robot_arm",
    "base_frame_name": "base_link",
    "ee_frame_name": "tool0",
    "joint_names": "j1,j2,j3,j4,j5,j6",
    "home_joints": "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0",
    "default_pipeline_id": "fairino",
    "default_planner_id": "aapf_birrt*",
    "target_rpy_deg": "0,-180,0",
    "auto_add_obstacle": True,
    "remove_obstacle_after_demo": False,
    "obstacle_name": "birrt_test_obstacle",
    "obstacle_position": "0.35,0.05,0.28",
    "obstacle_size": "0.18,0.45,0.35",
    "obstacle_boxes": "",
    "scene_name": "multi_obstacle_3d_avoidance",
    "spawn_gazebo_scene_models": True,
    "gazebo_world": "empty",
    "publish_planning_scene": True,
    "publish_obstacle_markers": True,
    "obstacle_marker_topic": "/demo_pathplanning/obstacle_markers",
    "planning_scene_obstacle_padding_m": 0.03,
    "benchmark_repetitions": 20,
    "benchmark_start_pose": "",
    "benchmark_result_csv": RESULT_CSV,
    "benchmark_case_label": "",
    "benchmark_goal_mode": "adaptive_obstacle_challenge_region",
    "benchmark_goal_seed": 17,
    "benchmark_goal_file": "",
    "benchmark_goal_clearance_min_m": 0.06,
    "benchmark_goal_clearance_max_m": 0.14,
    "benchmark_goal_corridor_clearance_max_m": 0.10,
    "benchmark_goal_min_separation_m": 0.04,
    "benchmark_goal_max_attempts_per_sample": 2000,
    "benchmark_goal_state_validity_timeout_s": 2.0,
    "planner_random_seed": 7,
    "benchmark_startup_joint_state_timeout_s": 90.0,
    "execute_planned_trajectory": False,
    "go_home_before_benchmark": True,
}


def _scene_paths(gz_share):
    scene_assets_dir = os.path.join(gz_share, "config", "scenes")
    return {
        "scene_assets_dir": scene_assets_dir,
        "scene_config_file": os.path.join(scene_assets_dir, "pathplanning_scenes.yaml"),
    }


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    scene_paths = _scene_paths(gz_share)

    launch_arguments = [
        DeclareLaunchArgument(
            "benchmark_scene_assets_dir", default_value=scene_paths["scene_assets_dir"]
        ),
        DeclareLaunchArgument(
            "benchmark_scene_config_file", default_value=scene_paths["scene_config_file"]
        ),
        DeclareLaunchArgument(
            "default_planner_id", default_value=NODE_PARAMS["default_planner_id"]
        ),
        DeclareLaunchArgument("scene_name", default_value=NODE_PARAMS["scene_name"]),
        DeclareLaunchArgument(
            "benchmark_repetitions", default_value=str(NODE_PARAMS["benchmark_repetitions"])
        ),
        DeclareLaunchArgument(
            "benchmark_goal_mode", default_value=NODE_PARAMS["benchmark_goal_mode"]
        ),
        DeclareLaunchArgument(
            "benchmark_goal_seed", default_value=str(NODE_PARAMS["benchmark_goal_seed"])
        ),
        DeclareLaunchArgument(
            "benchmark_goal_file", default_value=NODE_PARAMS["benchmark_goal_file"]
        ),
        DeclareLaunchArgument(
            "planner_random_seed", default_value=str(NODE_PARAMS["planner_random_seed"])
        ),
        DeclareLaunchArgument("enable_rviz", default_value=GAZEBO_LAUNCH_ARGUMENTS["enable_rviz"]),
        DeclareLaunchArgument("target_rpy_deg", default_value=NODE_PARAMS["target_rpy_deg"]),
        DeclareLaunchArgument(
            "planning_scene_obstacle_padding_m",
            default_value=str(NODE_PARAMS["planning_scene_obstacle_padding_m"]),
        ),
        DeclareLaunchArgument(
            "benchmark_goal_clearance_min_m",
            default_value=str(NODE_PARAMS["benchmark_goal_clearance_min_m"]),
        ),
        DeclareLaunchArgument(
            "benchmark_goal_clearance_max_m",
            default_value=str(NODE_PARAMS["benchmark_goal_clearance_max_m"]),
        ),
        DeclareLaunchArgument(
            "benchmark_goal_corridor_clearance_max_m",
            default_value=str(NODE_PARAMS["benchmark_goal_corridor_clearance_max_m"]),
        ),
        DeclareLaunchArgument(
            "benchmark_goal_min_separation_m",
            default_value=str(NODE_PARAMS["benchmark_goal_min_separation_m"]),
        ),
        DeclareLaunchArgument(
            "benchmark_goal_max_attempts_per_sample",
            default_value=str(NODE_PARAMS["benchmark_goal_max_attempts_per_sample"]),
        ),
        DeclareLaunchArgument(
            "execute_planned_trajectory",
            default_value=str(NODE_PARAMS["execute_planned_trajectory"]).lower(),
        ),
        DeclareLaunchArgument(
            "go_home_before_benchmark",
            default_value=str(NODE_PARAMS["go_home_before_benchmark"]).lower(),
        ),
    ]

    launch_scene_name = LaunchConfiguration("scene_name")
    launch_scene_paths = {
        "scene_assets_dir": LaunchConfiguration("benchmark_scene_assets_dir"),
        "scene_config_file": LaunchConfiguration("benchmark_scene_config_file"),
    }
    node_params = {
        **NODE_PARAMS,
        "default_planner_id": LaunchConfiguration("default_planner_id"),
        "scene_name": launch_scene_name,
        "benchmark_repetitions": ParameterValue(
            LaunchConfiguration("benchmark_repetitions"), value_type=int
        ),
        "benchmark_goal_mode": LaunchConfiguration("benchmark_goal_mode"),
        "benchmark_goal_seed": ParameterValue(
            LaunchConfiguration("benchmark_goal_seed"), value_type=int
        ),
        "benchmark_goal_file": LaunchConfiguration("benchmark_goal_file"),
        "planner_random_seed": ParameterValue(
            LaunchConfiguration("planner_random_seed"), value_type=int
        ),
        "target_rpy_deg": LaunchConfiguration("target_rpy_deg"),
        "planning_scene_obstacle_padding_m": ParameterValue(
            LaunchConfiguration("planning_scene_obstacle_padding_m"), value_type=float
        ),
        "benchmark_goal_clearance_min_m": ParameterValue(
            LaunchConfiguration("benchmark_goal_clearance_min_m"), value_type=float
        ),
        "benchmark_goal_clearance_max_m": ParameterValue(
            LaunchConfiguration("benchmark_goal_clearance_max_m"), value_type=float
        ),
        "benchmark_goal_corridor_clearance_max_m": ParameterValue(
            LaunchConfiguration("benchmark_goal_corridor_clearance_max_m"), value_type=float
        ),
        "benchmark_goal_min_separation_m": ParameterValue(
            LaunchConfiguration("benchmark_goal_min_separation_m"), value_type=float
        ),
        "benchmark_goal_max_attempts_per_sample": ParameterValue(
            LaunchConfiguration("benchmark_goal_max_attempts_per_sample"), value_type=int
        ),
        "execute_planned_trajectory": ParameterValue(
            LaunchConfiguration("execute_planned_trajectory"), value_type=bool
        ),
        "go_home_before_benchmark": ParameterValue(
            LaunchConfiguration("go_home_before_benchmark"), value_type=bool
        ),
    }

    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **GAZEBO_LAUNCH_ARGUMENTS,
            "scene_name": launch_scene_name,
            "planner_random_seed": LaunchConfiguration("planner_random_seed"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            **launch_scene_paths,
            "rviz_config": os.path.join(gz_share, "rviz", "fairino_planning_test.rviz"),
        }.items(),
    )

    trajectory_plan_test_node = Node(
        package="myrobot_simulation",
        executable="trajectory_plan_test_node.py",
        name="trajectory_plan_test_node",
        output="screen",
        emulate_tty=True,
        parameters=[{**node_params, **launch_scene_paths}],
    )

    delayed_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(
                msg=[
                    "[trajectory_plan_test] client=fairino, namespace_override=, "
                    "pipeline=fairino, planner=",
                    LaunchConfiguration("default_planner_id"),
                    ", scene=",
                    LaunchConfiguration("scene_name"),
                    ", goal_mode=",
                    LaunchConfiguration("benchmark_goal_mode"),
                    ", planner_seed=",
                    LaunchConfiguration("planner_random_seed"),
                    ", execute=",
                    LaunchConfiguration("execute_planned_trajectory"),
                    ", pre_home=",
                    LaunchConfiguration("go_home_before_benchmark"),
                ]
            ),
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
        )
    )

    return LaunchDescription([*launch_arguments, myrobot_simulation, delayed_node, shutdown_when_node_exits])
