"""Gazebo YOLO-OBB 检测入口."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# 模型文件是包内固定资源；其余运行时标量可由 launch 命令覆盖。
_NODE_DEFAULTS = {
    "device": "auto",
    "conf": "0.6",
    "imgsz": "640",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _NODE_DEFAULTS
}


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(
            "device", default_value=_NODE_DEFAULTS["device"], description="YOLO 推理设备。"
        ),
        DeclareLaunchArgument(
            "conf", default_value=_NODE_DEFAULTS["conf"], description="检测置信度阈值。"
        ),
        DeclareLaunchArgument(
            "imgsz", default_value=_NODE_DEFAULTS["imgsz"], description="YOLO 输入图像尺寸。"
        ),
    ]


def generate_launch_description():
    model_path = os.path.join(
        get_package_share_directory("yolo_perception"), "models", "yolo-obb-gazebo.pt"
    )
    yolo_detector_node_obb = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="yolo_perception",
                executable="yolo_detector_obb.py",
                name="yolov8_detector_obb",
                output="screen",
                parameters=[
                    {
                        "model_path": model_path,
                        "device": _LAUNCH_CONFIGURATIONS["device"],
                        "conf": _LAUNCH_CONFIGURATIONS["conf"],
                        "imgsz": _LAUNCH_CONFIGURATIONS["imgsz"],
                    }
                ],
            )
        ],
    )
    return LaunchDescription([*_declare_launch_arguments(), yolo_detector_node_obb])
