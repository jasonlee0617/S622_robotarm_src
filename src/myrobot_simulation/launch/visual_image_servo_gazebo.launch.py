#!/usr/bin/env python3
"""Gazebo entry point for ArUco four-corner IBVS."""

import os
from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from visual_servo_bringup.image_servo_config import NODE_PARAMETER_DESCRIPTIONS, image_servo_parameters


_IMAGE_SERVO_DEFAULTS = image_servo_parameters()
_REFERENCE_PATH = str(
    Path.home()
    / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim"
    / "image_servo_aruco_id1.yaml"
)


def _launch_default(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    return yaml.safe_dump(value, default_flow_style=True).strip()

DEFAULTS = {
    "use_sim_time": "true",
    "enable_rviz": "true",
    "camera_profile": "d435_color_640x480x60_depth_640x480x60",
    "camera_fps": "60",
    "calibration_name": "robot_calibration",
    "reference_path": _launch_default(_IMAGE_SERVO_DEFAULTS["reference_path"] or _REFERENCE_PATH),
    **{
        name: _launch_default(default)
        for name, default in _IMAGE_SERVO_DEFAULTS.items()
        if name != "reference_path"
    },
}

DESCRIPTIONS = {
    "use_sim_time": "是否使用 Gazebo 的 /clock。",
    "enable_rviz": "是否启动 Gazebo RViz。",
    "camera_profile": "仿真 D435 配置；图像伺服默认 640x480。",
    "camera_fps": "仿真相机帧率。",
    "calibration_name": "读取的仿真手眼标定名称。",
    "reference_path": "记录仿真目标图像角点的 YAML 文件。",
    **NODE_PARAMETER_DESCRIPTIONS,
}


def _node_parameter(context, name):
    raw = LaunchConfiguration(name).perform(context)
    return raw if isinstance(_IMAGE_SERVO_DEFAULTS[name], str) else yaml.safe_load(raw)


def _argument(name, default):
    return DeclareLaunchArgument(name, default_value=default, description=DESCRIPTIONS[name])


def _launch_setup(context):
    simulation_share = get_package_share_directory("myrobot_simulation")
    visual_servo_share = get_package_share_directory("visual_servo_bringup")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(simulation_share, "launch", "calibration_gazebo.launch.py")
        ),
        launch_arguments={
            "robot_profile": "fairino_arm_gripper_inhand",
            "camera_profile": LaunchConfiguration("camera_profile"),
            "camera_fps": LaunchConfiguration("camera_fps"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "enable_servo": "true",
            "rviz_config": os.path.join(
                visual_servo_share, "rviz", "visual_image_servo.rviz"
            ),
            # calibration_gazebo.launch.py spawns this exact board model.
            "spawn_fixed_board": "true",
        }.items(),
    )

    # 仿真使用现有 validate.launch.py 同样的手眼 TF 发布节点。
    handeye = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "calibration_name": LaunchConfiguration("calibration_name"),
                "storage_directory": str(
                    Path.home()
                    / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim"
                ),
            }
        ],
    )

    image_params = {name: _node_parameter(context, name) for name in _IMAGE_SERVO_DEFAULTS}
    ibvs = TimerAction(
        period=12.0,
        actions=[
            Node(
                package="visual_servo_bringup",
                executable="visual_image_servo",
                name="visual_image_servo",
                output="screen",
                parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}, image_params],
            )
        ],
    )

    return [gazebo, handeye, ibvs]


def generate_launch_description():
    return LaunchDescription(
        [
            *(_argument(name, default) for name, default in DEFAULTS.items()),
            OpaqueFunction(function=_launch_setup),
        ]
    )
