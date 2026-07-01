import os

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def arg(name, default_value, description=""):
    return DeclareLaunchArgument(name, default_value=default_value, description=description)


def base_args(gz_share, *, remove_obstacle_after_demo, shutdown_on_demo_exit):
    return [
        arg("robot_profile", "s622_gripper", "Robot profile used by gazebo.launch.py."),
        arg("rviz_config", os.path.join(gz_share, "rviz", "fairino_planning_test.rviz"), "RViz config for Fairino trajectory planning."),
        arg("enable_rviz", "true"),
        arg("world", "empty"),
        arg("use_sim_time", "true"),
        arg("demo_start_delay_s", "5.0", "Delay node start until Gazebo and controllers are ready."),
        arg("initial_positions_file", "", "Optional Gazebo/ros2_control initial_positions YAML override."),
        arg("ik_plugin", "fairino", "Planning client in dual move_group setup: fairino or kdl."),
        arg("planning_move_group_namespace", "", "Optional explicit move_group namespace override."),
        arg("planning_pipeline", "fairino", "Planning pipeline: fairino or ompl."),
        arg("planning_algorithm", "aapf_birrt*", "Planner id: tube_birrt*, birrt*, rrt*, aapf_birrt*, RRTConnect, etc."),
        arg("group_name", "robot_arm"),
        arg("base_frame_name", "base_link"),
        arg("ee_frame_name", "grasp_frame"),
        arg("joint_names", "j1,j2,j3,j4,j5,j6"),
        arg("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0"),
        arg("target_rpy_deg", "0,-180,0", "Fixed end-effector orientation as roll,pitch,yaw in degrees."),
        arg("auto_add_obstacle", "true", "Whether to add the default PlanningScene obstacle at start."),
        arg("remove_obstacle_after_demo", remove_obstacle_after_demo, "Whether to remove scene obstacles on exit."),
        arg("obstacle_name", "birrt_test_obstacle"),
        arg("obstacle_position", "0.35,0.05,0.28", "Default obstacle center xyz in base frame."),
        arg("obstacle_size", "0.18,0.45,0.35", "Default obstacle size sx,sy,sz."),
        arg("obstacle_boxes", "", "Optional boxes: name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz."),
        arg("scene_assets_dir", os.path.join(gz_share, "config", "scenes"), "Directory containing scene URDF/SDF assets."),
        arg("scene_config_file", os.path.join(gz_share, "config", "scenes", "pathplanning_scenes.yaml"), "YAML file containing named path-planning scenes."),
        arg("scene_name", "paper_simple_3d_avoidance", "Scene key in pathplanning_scenes.yaml."),
        arg("spawn_gazebo_scene_models", "true", "If true, also spawn scene assets into Gazebo."),
        arg("publish_planning_scene", "true", "If true, publish scene obstacles into MoveIt PlanningScene."),
        arg("publish_obstacle_markers", "true", "If true, publish RViz obstacle markers for the selected scene."),
        arg("obstacle_marker_topic", "/demo_pathplanning/obstacle_markers", "MarkerArray topic for planning scene obstacles."),
        arg("planning_scene_obstacle_padding_m", "0.03", "Padding applied only to MoveIt collision objects."),
        arg("shutdown_on_demo_exit", shutdown_on_demo_exit, "If true, shut down launch when the trajectory plan node exits."),
    ]


def gazebo_launch(gz_share):
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
