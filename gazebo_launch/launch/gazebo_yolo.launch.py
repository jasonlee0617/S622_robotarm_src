import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.gazebo_stack import base_simulation_actions
from launch_utils.launch_parsing import as_bool, resolve_launch_args, spawn_pose_from_context
from launch_utils.perception_stack import camera_bridge_nodes, servo_node
from launch_utils.robot_profiles import load_robot_profile
from manipulation_common.launch_utils.yaml_loader import load_yaml


_GAZEBO_SHARE = get_package_share_directory("gazebo_launch")
_GAZEBO_YOLO_DEFAULTS = {
    "robot_profile": "s622_gripper",
    "world": "arm_on_the_table",
    "rviz_config": os.path.join(_GAZEBO_SHARE, "rviz", "gazebo_yolo.rviz"),
    "enable_rviz": "true",
    "use_sim_time": "true",
    "publish_frequency": "30.0",
    "enable_camera_model": "true",
    "enable_camera_bridge": "true",
    "enable_servo": "true",
    "camera_info_remap": "/camera/camera/aligned_depth_to_color/camera_info",
    "camera_fps": "60",
    "camera_image_width": "640",
    "camera_image_height": "480",
    "spawn_name": "",
    "spawn_x": "0.0",
    "spawn_y": "0.0",
    "spawn_z": "1.02",
    "spawn_roll": "0.0",
    "spawn_pitch": "0.0",
    "spawn_yaw": "0.0",
    "robot_spawn_delay": "5.0",
    "controller_spawn_delay": "5.0",
    "scene_assets_dir": os.path.join(_GAZEBO_SHARE, "config", "scenes"),
    "scene_config_file": os.path.join(_GAZEBO_SHARE, "config", "scenes", "pathplanning_scenes.yaml"),
    "scene_name": "single_obstacle",
    "spawn_gazebo_scene_models": "false",
    "publish_planning_scene": "true",
    "publish_obstacle_markers": "true",
    "obstacle_marker_topic": "/demo_pathplanning/obstacle_markers",
}
_GAZEBO_YOLO_ARG_DESCRIPTIONS = {
    "enable_camera_model": "Enable camera model/sensor plugins in robot_description xacro.",
    "scene_assets_dir": "Fallback scene asset directory for parent launches.",
    "scene_config_file": "Fallback path-planning scene config for parent launches.",
}


def _declare_launch_arguments(defaults: dict):
    arguments = []
    for name, value in defaults.items():
        kwargs = {"default_value": str(value)}
        if name in _GAZEBO_YOLO_ARG_DESCRIPTIONS:
            kwargs["description"] = _GAZEBO_YOLO_ARG_DESCRIPTIONS[name]
        arguments.append(DeclareLaunchArgument(name, **kwargs))
    return arguments


def _launch_setup(context, *args, **kwargs):
    p = resolve_launch_args(context, _GAZEBO_YOLO_DEFAULTS)

    profile = load_robot_profile(p["robot_profile"])
    spawn_xyz, spawn_rpy = spawn_pose_from_context(context)

    actions, moveit_config = base_simulation_actions(
        profile,
        world=p["world"],
        rviz_config=p["rviz_config"],
        spawn_xyz=spawn_xyz,
        spawn_rpy=spawn_rpy,
        spawn_name=p["spawn_name"] or profile.spawn_name,
        use_sim_time=as_bool(p["use_sim_time"]),
        enable_rviz=as_bool(p["enable_rviz"]),
        publish_frequency=float(p["publish_frequency"]),
        enable_camera_model=as_bool(p["enable_camera_model"]),
        robot_spawn_delay=float(p["robot_spawn_delay"]),
        controller_spawn_delay=float(p["controller_spawn_delay"]),
        extra_mappings={
            "camera_fps": p["camera_fps"],
            "camera_image_width": p["camera_image_width"],
            "camera_image_height": p["camera_image_height"],
        },
    )

    if as_bool(p["enable_camera_bridge"]):
        actions.extend(camera_bridge_nodes(as_bool(p["use_sim_time"]), p["camera_info_remap"]))

    if as_bool(p["enable_servo"]):
        kinematics_kdl_config = load_yaml(profile.moveit_config_package, profile.kinematics_kdl_file)
        actions.append(servo_node(moveit_config, profile, kinematics_kdl_config, as_bool(p["use_sim_time"])))

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            *_declare_launch_arguments(_GAZEBO_YOLO_DEFAULTS),
            OpaqueFunction(function=_launch_setup),
        ]
    )
