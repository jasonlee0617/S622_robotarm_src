import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
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
            Node(
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
                    }
                ],
            ),
        ],
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
            gazebo_launch,
            path_planning_demo_node,
        ]
    )
