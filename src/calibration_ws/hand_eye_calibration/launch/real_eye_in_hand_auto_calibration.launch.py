"""Real eye-in-hand automatic calibration without ros2_aruco/easy_handeye2 sampling."""

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)
from handeye_launch_utils import camera_launch


def _camera_action(context, *_args, **_kwargs):
    return [camera_launch(
        LaunchConfiguration("camera_type").perform(context),
        realsense_args={
            "serial_no": LaunchConfiguration("camera_serial_no").perform(context),
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.profile": LaunchConfiguration("color_profile").perform(context),
            "depth_module.profile": LaunchConfiguration("depth_profile").perform(context),
            "align_depth.enable": "true",
        },
    )]


def generate_launch_description():
    moveit_share = get_package_share_directory("fairino_arm_moveit_config")
    storage = str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real")
    return LaunchDescription([
        DeclareLaunchArgument("camera_type", default_value="realsense", choices=["realsense", "oak"]),
        DeclareLaunchArgument("camera_serial_no", default_value="", description="D435 serial number; empty lets the driver select one camera."),
        DeclareLaunchArgument("color_profile", default_value="1280x720x30"),
        DeclareLaunchArgument("depth_profile", default_value="848x480x30"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "start_demo_moveit", default_value="false",
            description="Only for dry-run development. A real robot must use its already-running hardware MoveIt stack.",
        ),
        DeclareLaunchArgument("auto_start", default_value="false"),
        DeclareLaunchArgument("storage_directory", default_value=storage),
        OpaqueFunction(function=_camera_action),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(moveit_share, "launch", "demo.launch.py")),
            launch_arguments={"use_rviz": LaunchConfiguration("use_rviz")}.items(),
            condition=IfCondition(LaunchConfiguration("start_demo_moveit")),
        ),
        TimerAction(
            period=15.0,
            actions=[Node(
                package="hand_eye_calibration",
                executable="auto_calibration_collector.py",
                name="auto_calibration_collector",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "use_sim_time": False,
                        "auto_start": LaunchConfiguration("auto_start"),
                        "move_group_ns_fairino": "",
                        "move_group_ns_kdl": "",
                        "ik_plugin": "kdl",
                        "planning_pipeline_id": "ompl",
                        "planner_id": "RRTConnectFast",
                        "calibration_output_directory": LaunchConfiguration("storage_directory"),
                    },
                ],
            )],
        ),
    ])
