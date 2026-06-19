import os

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
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")

    robot_profile_arg = DeclareLaunchArgument(
        "robot_profile",
        default_value="s622_gripper",
        description="Robot profile used by gazebo.launch.py.",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(gz_share, "rviz", "fairino_planning_test.rviz"),
        description="RViz config for Fairino planning test.",
    )

    enable_rviz_arg = DeclareLaunchArgument("enable_rviz", default_value="true")
    world_arg = DeclareLaunchArgument("world", default_value="empty")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")

    ik_plugin_arg = DeclareLaunchArgument(
        "ik_plugin",
        default_value="fairino",
        description="Planning client in dual move_group setup: fairino or kdl.",
    )

    planning_move_group_namespace_arg = DeclareLaunchArgument(
        "planning_move_group_namespace",
        default_value="",
        description="Optional explicit move_group namespace override, e.g. /move_group_kdl.",
    )

    planning_pipeline_arg = DeclareLaunchArgument(
        "planning_pipeline",
        default_value="fairino",
        description="Planning pipeline: fairino or ompl.",
    )

    planning_algorithm_arg = DeclareLaunchArgument(
        "planning_algorithm",
        # default_value="birrt*",
        default_value="aapf_birrt*",

        description="Planner id: birrt*, rrt*, RRTConnect, etc. Fairino planner ids are lowercase only.",
    )

    group_name_arg = DeclareLaunchArgument("group_name", default_value="robot_arm")
    base_frame_arg = DeclareLaunchArgument("base_frame_name", default_value="base_link")
    ee_frame_arg = DeclareLaunchArgument("ee_frame_name", default_value="grasp_frame")
    joint_names_arg = DeclareLaunchArgument("joint_names", default_value="j1,j2,j3,j4,j5,j6")

    home_joints_arg = DeclareLaunchArgument(
        "home_joints",
        default_value="-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0",
    )

    target_rpy_deg_arg = DeclareLaunchArgument(
        "target_rpy_deg",
        default_value="0,-180,0",
        description="Fixed end-effector orientation as roll,pitch,yaw in degrees.",
    )

    go_home_before_demo_arg = DeclareLaunchArgument(
        "go_home_before_demo",
        default_value="false",
        description="If true, move to HOME before accepting start/goal input.",
    )

    auto_add_obstacle_arg = DeclareLaunchArgument(
        "auto_add_obstacle",
        default_value="true",
        description="Whether to add the default PlanningScene obstacle at demo start and recover.",
    )

    remove_obstacle_after_demo_arg = DeclareLaunchArgument(
        "remove_obstacle_after_demo",
        default_value="true",
        description="Whether to remove demo-created PlanningScene obstacles on exit.",
    )

    obstacle_name_arg = DeclareLaunchArgument(
        "obstacle_name",
        default_value="birrt_test_obstacle",
        # default_value="simple_left_cylinder",

    )

    obstacle_position_arg = DeclareLaunchArgument(
        "obstacle_position",
        default_value="0.35,0.05,0.28",
        description="Default obstacle center xyz in base frame.",
    )

    obstacle_size_arg = DeclareLaunchArgument(
        "obstacle_size",
        default_value="0.18,0.45,0.35",
        description="Default obstacle size sx,sy,sz.",
    )

    obstacle_boxes_arg = DeclareLaunchArgument(
        "obstacle_boxes",
        default_value="",
        description="Optional boxes: name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz.",
    )

    scene_assets_dir_arg = DeclareLaunchArgument(
        "scene_assets_dir",
        default_value=os.path.join(gz_share, "config", "scenes"),
        description="Directory containing scene URDF/SDF assets.",
    )

    scene_config_file_arg = DeclareLaunchArgument(
        "scene_config_file",
        default_value=os.path.join(gz_share, "config", "scenes", "pathplanning_scenes.yaml"),
        description="YAML file containing named path-planning scenes.",
    )

    scene_name_arg = DeclareLaunchArgument(
        "scene_name",
        # default_value="single_obstacle",
        default_value="paper_simple_3d_avoidance",
        description="Scene key in pathplanning_scenes.yaml.",
    )

    spawn_gazebo_scene_models_arg = DeclareLaunchArgument(
        "spawn_gazebo_scene_models",
        default_value="true",
        description="If true, also spawn scene assets into Gazebo using ros_gz_sim create.",
    )

    publish_planning_scene_arg = DeclareLaunchArgument(
        "publish_planning_scene",
        default_value="true",
        description="If true, publish scene obstacles into MoveIt PlanningScene.",
    )

    publish_obstacle_markers_arg = DeclareLaunchArgument(
        "publish_obstacle_markers",
        default_value="true",
        description="If true, publish RViz obstacle markers for the selected scene.",
    )

    obstacle_marker_topic_arg = DeclareLaunchArgument(
        "obstacle_marker_topic",
        default_value="/demo_pathplanning/obstacle_markers",
        description="MarkerArray topic for visualizing planning scene obstacles.",
    )

    benchmark_setup_planner_id_arg = DeclareLaunchArgument(
        "benchmark_setup_planner_id",
        default_value="birrt*",
        description="Planner used for HOME→start leg in benchmark mode (decoupled from tested planner).",
    )
    benchmark_use_controller_reset_for_home_arg = DeclareLaunchArgument(
        "benchmark_use_controller_reset_for_home",
        default_value="false",
        description="Legacy switch for controller HOME reset; benchmark_home_reset_mode controls new cases.",
    )
    benchmark_home_reset_mode_arg = DeclareLaunchArgument(
        "benchmark_home_reset_mode",
        default_value="planner",
        description="HOME reset mode in benchmark: planner or controller_trajectory.",
    )
    benchmark_home_planner_id_arg = DeclareLaunchArgument(
        "benchmark_home_planner_id",
        default_value="birrt*",
        description="Planner used for collision-aware HOME reset in benchmark mode.",
    )
    benchmark_abort_on_home_reset_failure_arg = DeclareLaunchArgument(
        "benchmark_abort_on_home_reset_failure",
        default_value="true",
        description="Abort the benchmark after a HOME reset failure to avoid stale repeated failures.",
    )
    benchmark_record_phase_times_arg = DeclareLaunchArgument(
        "benchmark_record_phase_times",
        default_value="true",
        description="If true, record per-phase timing (home reset, setup start, goal plan).",
    )
    benchmark_action_delay_s_arg = DeclareLaunchArgument(
        "benchmark_action_delay_s",
        default_value="0.0",
        description="Post-action sleep in benchmark mode; interactive demo keeps its internal default.",
    )
    benchmark_pair_planners_by_goal_arg = DeclareLaunchArgument(
        "benchmark_pair_planners_by_goal",
        default_value="true",
        description="If true, run all planners for each generated goal before moving to the next goal.",
    )
    benchmark_goal_mode_arg = DeclareLaunchArgument(
        "benchmark_goal_mode",
        default_value="random_obstacle_envelope",
        description="Goal generation mode: fixed, random_obstacle_envelope, or random_pose_goal_region.",
    )
    benchmark_goal_seed_arg = DeclareLaunchArgument(
        "benchmark_goal_seed",
        default_value="17",
        description="Random seed for goal generation when benchmark_goal_mode is random.",
    )
    benchmark_goal_clearance_min_m_arg = DeclareLaunchArgument(
        "benchmark_goal_clearance_min_m",
        default_value="0.06",
        description="Minimum clearance from goal TCP to nearest obstacle surface (m).",
    )
    benchmark_goal_clearance_max_m_arg = DeclareLaunchArgument(
        "benchmark_goal_clearance_max_m",
        default_value="0.14",
        description="Maximum clearance from goal TCP to nearest obstacle surface (m).",
    )
    benchmark_goal_min_separation_m_arg = DeclareLaunchArgument(
        "benchmark_goal_min_separation_m",
        default_value="0.04",
        description="Minimum separation between generated goals and start pose (m).",
    )
    benchmark_goal_max_attempts_per_sample_arg = DeclareLaunchArgument(
        "benchmark_goal_max_attempts_per_sample",
        default_value="200",
        description="Max random sampling attempts per goal candidate.",
    )
    benchmark_goal_region_min_arg = DeclareLaunchArgument(
        "benchmark_goal_region_min",
        default_value="",
        description="Optional x,y,z lower bound for benchmark_goal_mode=random_pose_goal_region.",
    )
    benchmark_goal_region_max_arg = DeclareLaunchArgument(
        "benchmark_goal_region_max",
        default_value="",
        description="Optional x,y,z upper bound for benchmark_goal_mode=random_pose_goal_region.",
    )
    planning_scene_obstacle_padding_m_arg = DeclareLaunchArgument(
        "planning_scene_obstacle_padding_m",
        default_value="0.03",
        description="Padding applied only to MoveIt collision objects, not Gazebo visual models.",
    )
    benchmark_mode_arg = DeclareLaunchArgument(
        "benchmark_mode",
        default_value="false",
        description="If true, auto-run benchmark planners on fixed or reproducible random goals.",
    )

    benchmark_planners_arg = DeclareLaunchArgument(
        "benchmark_planners",
        default_value="",
        description="Comma-separated planner ids for benchmark mode, e.g. aapf_birrt*,birrt*.",
    )

    benchmark_repetitions_arg = DeclareLaunchArgument(
        "benchmark_repetitions",
        default_value="1",
        description="Number of repetitions per planner in benchmark mode.",
    )

    benchmark_start_pose_arg = DeclareLaunchArgument(
        "benchmark_start_pose",
        default_value="",
        description="Benchmark start pose as x,y,z[,rx,ry,rz].",
    )

    benchmark_goal_pose_arg = DeclareLaunchArgument(
        "benchmark_goal_pose",
        default_value="",
        description="Benchmark goal pose as x,y,z[,rx,ry,rz].",
    )

    benchmark_result_csv_arg = DeclareLaunchArgument(
        "benchmark_result_csv",
        default_value="",
        description="Optional CSV output path for benchmark results.",
    )

    benchmark_case_label_arg = DeclareLaunchArgument(
        "benchmark_case_label",
        default_value="",
        description="Optional label written into benchmark results.",
    )

    benchmark_notes_arg = DeclareLaunchArgument(
        "benchmark_notes",
        default_value="",
        description="Optional notes written into benchmark results.",
    )

    benchmark_go_home_each_run_arg = DeclareLaunchArgument(
        "benchmark_go_home_each_run",
        default_value="true",
        description="If true, benchmark runs return HOME before each repetition.",
    )

    benchmark_reset_scene_each_run_arg = DeclareLaunchArgument(
        "benchmark_reset_scene_each_run",
        default_value="true",
        description="If true, benchmark runs reload the PlanningScene before each repetition.",
    )

    benchmark_move_to_start_each_run_arg = DeclareLaunchArgument(
        "benchmark_move_to_start_each_run",
        default_value="true",
        description="If true, benchmark runs execute HOME -> start_pose before planning to goal.",
    )

    shutdown_on_demo_exit_arg = DeclareLaunchArgument(
        "shutdown_on_demo_exit",
        default_value="false",
        description="If true, shut down the launch when demo_pathplanning_node exits.",
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "robot_profile": LaunchConfiguration("robot_profile"),
            "rviz_config": LaunchConfiguration("rviz_config"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "world": LaunchConfiguration("world"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
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

    demo_pathplanning_node = Node(
        package="gazebo_launch",
        executable="demo_pathplanning_node.py",
        name="demo_pathplanning_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
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
                "benchmark_mode": LaunchConfiguration("benchmark_mode"),
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
                "planning_scene_obstacle_padding_m": LaunchConfiguration("planning_scene_obstacle_padding_m"),
            }
        ],
    )

    path_planning_demo_node = TimerAction(
        period=3.0,
        actions=[
            LogInfo(
                msg=[
                    "[planning_demo] client=",
                    LaunchConfiguration("ik_plugin"),
                    ", namespace_override=",
                    LaunchConfiguration("planning_move_group_namespace"),
                    ", pipeline=",
                    LaunchConfiguration("planning_pipeline"),
                    ", planner=",
                    LaunchConfiguration("planning_algorithm"),
                    ", rviz_config=",
                    LaunchConfiguration("rviz_config"),
                ]
            ),
            demo_pathplanning_node,
        ],
    )

    shutdown_when_demo_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=demo_pathplanning_node,
            on_exit=[
                LogInfo(msg="[planning_demo] demo_pathplanning_node exited; shutting down launch."),
                EmitEvent(event=Shutdown(reason="planning demo completed")),
            ],
        ),
        condition=IfCondition(LaunchConfiguration("shutdown_on_demo_exit")),
    )

    return LaunchDescription(
        [
            robot_profile_arg,
            rviz_config_arg,
            enable_rviz_arg,
            world_arg,
            use_sim_time_arg,
            ik_plugin_arg,
            planning_move_group_namespace_arg,
            planning_pipeline_arg,
            planning_algorithm_arg,
            group_name_arg,
            base_frame_arg,
            ee_frame_arg,
            joint_names_arg,
            home_joints_arg,
            target_rpy_deg_arg,

            go_home_before_demo_arg,

            auto_add_obstacle_arg,
            remove_obstacle_after_demo_arg,
            obstacle_name_arg,
            obstacle_position_arg,
            obstacle_size_arg,
            obstacle_boxes_arg,
            scene_assets_dir_arg,
            scene_config_file_arg,
            scene_name_arg,
            spawn_gazebo_scene_models_arg,
            publish_planning_scene_arg,
            publish_obstacle_markers_arg,
            obstacle_marker_topic_arg,
            benchmark_setup_planner_id_arg,
            benchmark_use_controller_reset_for_home_arg,
            benchmark_home_reset_mode_arg,
            benchmark_home_planner_id_arg,
            benchmark_abort_on_home_reset_failure_arg,
            benchmark_record_phase_times_arg,
            benchmark_action_delay_s_arg,
            benchmark_pair_planners_by_goal_arg,
            benchmark_goal_mode_arg,
            benchmark_goal_seed_arg,
            benchmark_goal_clearance_min_m_arg,
            benchmark_goal_clearance_max_m_arg,
            benchmark_goal_min_separation_m_arg,
            benchmark_goal_max_attempts_per_sample_arg,
            benchmark_goal_region_min_arg,
            benchmark_goal_region_max_arg,
            planning_scene_obstacle_padding_m_arg,
            benchmark_mode_arg,
            benchmark_planners_arg,
            benchmark_repetitions_arg,
            benchmark_start_pose_arg,
            benchmark_goal_pose_arg,
            benchmark_result_csv_arg,
            benchmark_case_label_arg,
            benchmark_notes_arg,
            benchmark_go_home_each_run_arg,
            benchmark_reset_scene_each_run_arg,
            benchmark_move_to_start_each_run_arg,
            shutdown_on_demo_exit_arg,
            gazebo_launch,
            path_planning_demo_node,
            shutdown_when_demo_exits,
        ]
    )
