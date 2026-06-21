import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.gazebo_stack import base_simulation_actions
from launch_utils.launch_parsing import as_bool, spawn_pose_from_context
from launch_utils.robot_profiles import load_robot_profile


def _launch_setup(context, *args, **kwargs):
    profile = load_robot_profile(LaunchConfiguration("robot_profile").perform(context))

    spawn_name = LaunchConfiguration("spawn_name").perform(context) or profile.spawn_name
    spawn_xyz, spawn_rpy = spawn_pose_from_context(context)
    initial_positions_file = LaunchConfiguration("initial_positions_file").perform(context)
    extra_mappings = {}
    if initial_positions_file:
        extra_mappings["initial_positions_file"] = initial_positions_file

    actions, _ = base_simulation_actions(
        profile,
        world=LaunchConfiguration("world").perform(context),
        rviz_config=LaunchConfiguration("rviz_config").perform(context),
        spawn_xyz=spawn_xyz,
        spawn_rpy=spawn_rpy,
        spawn_name=spawn_name,
        use_sim_time=as_bool(LaunchConfiguration("use_sim_time").perform(context)),
        enable_rviz=as_bool(LaunchConfiguration("enable_rviz").perform(context)),
        publish_frequency=float(LaunchConfiguration("publish_frequency").perform(context)),
        default_planning_pipeline=LaunchConfiguration("default_planning_pipeline").perform(context),
        enable_camera_model=as_bool(LaunchConfiguration("enable_camera_model").perform(context)),
        extra_mappings=extra_mappings,
    )
    return actions


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_profile", default_value="s622_gripper"),
            DeclareLaunchArgument("world", default_value="empty"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(gz_share, "rviz", "ik_test.rviz"),
            ),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("publish_frequency", default_value="100.0"),
            DeclareLaunchArgument("default_planning_pipeline", default_value="fairino"),
            DeclareLaunchArgument(
                "initial_positions_file",
                default_value="",
                description="Optional xacro initial_positions YAML override.",
            ),
            DeclareLaunchArgument(
                "enable_camera_model",
                default_value="false",
                description="Enable camera model/sensor plugins in robot_description xacro.",
            ),
            DeclareLaunchArgument("spawn_name", default_value=""),
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="0.0"),
            DeclareLaunchArgument("spawn_z", default_value="0.0"),
            DeclareLaunchArgument("spawn_roll", default_value="0.0"),
            DeclareLaunchArgument("spawn_pitch", default_value="0.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "scene_assets_dir",
                default_value=os.path.join(gz_share, "config", "scenes"),
                description="Fallback scene asset directory for parent launches.",
            ),
            DeclareLaunchArgument(
                "scene_config_file",
                default_value=os.path.join(gz_share, "config", "scenes", "pathplanning_scenes.yaml"),
                description="Fallback path-planning scene config for parent launches.",
            ),
            DeclareLaunchArgument("scene_name", default_value="single_obstacle"),
            DeclareLaunchArgument("spawn_gazebo_scene_models", default_value="false"),
            DeclareLaunchArgument("publish_planning_scene", default_value="true"),
            DeclareLaunchArgument("publish_obstacle_markers", default_value="true"),
            DeclareLaunchArgument(
                "obstacle_marker_topic",
                default_value="/demo_pathplanning/obstacle_markers",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
