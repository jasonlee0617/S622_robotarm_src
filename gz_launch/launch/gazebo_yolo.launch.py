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
from launch_utils.perception_stack import camera_bridge_nodes, servo_node
from launch_utils.robot_profiles import load_robot_profile
from launch_utils.yaml_loader import load_yaml


def _launch_setup(context, *args, **kwargs):
    profile = load_robot_profile(LaunchConfiguration("robot_profile").perform(context))
    spawn_name = LaunchConfiguration("spawn_name").perform(context) or profile.spawn_name
    spawn_xyz, spawn_rpy = spawn_pose_from_context(context)
    use_sim_time = as_bool(LaunchConfiguration("use_sim_time").perform(context))

    actions, moveit_config = base_simulation_actions(
        profile,
        world=LaunchConfiguration("world").perform(context),
        rviz_config=LaunchConfiguration("rviz_config").perform(context),
        spawn_xyz=spawn_xyz,
        spawn_rpy=spawn_rpy,
        spawn_name=spawn_name,
        use_sim_time=use_sim_time,
        enable_rviz=as_bool(LaunchConfiguration("enable_rviz").perform(context)),
        publish_frequency=float(LaunchConfiguration("publish_frequency").perform(context)),
        default_planning_pipeline=LaunchConfiguration("default_planning_pipeline").perform(context),
        enable_camera_model=as_bool(LaunchConfiguration("enable_camera_model").perform(context)),
    )

    if as_bool(LaunchConfiguration("enable_camera_bridge").perform(context)):
        actions.extend(camera_bridge_nodes(use_sim_time))

    if as_bool(LaunchConfiguration("enable_servo").perform(context)):
        kinematics_kdl_config = load_yaml(profile.moveit_config_package, profile.kinematics_kdl_file)
        actions.append(servo_node(moveit_config, profile, kinematics_kdl_config, use_sim_time))

    return actions


def generate_launch_description():
    gz_share = get_package_share_directory("gz_launch")
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_profile", default_value="s622_gripper"),
            DeclareLaunchArgument("world", default_value="arm_on_the_table"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(gz_share, "rviz", "gazebo_yolo.rviz"),
            ),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("publish_frequency", default_value="30.0"),
            DeclareLaunchArgument("default_planning_pipeline", default_value="fairino"),
            DeclareLaunchArgument(
                "enable_camera_model",
                default_value="true",
                description="Enable camera model/sensor plugins in robot_description xacro.",
            ),
            DeclareLaunchArgument("enable_camera_bridge", default_value="true"),
            DeclareLaunchArgument("enable_servo", default_value="true"),
            DeclareLaunchArgument("spawn_name", default_value=""),
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="0.0"),
            DeclareLaunchArgument("spawn_z", default_value="1.02"),
            DeclareLaunchArgument("spawn_roll", default_value="0.0"),
            DeclareLaunchArgument("spawn_pitch", default_value="0.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
