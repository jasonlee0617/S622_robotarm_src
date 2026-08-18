"""真实 D435 的 YOLO-OBB 检测入口."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_NODE_DEFAULTS = {"device": "auto", "conf": "0.6", "imgsz": "1280"}
_CAMERA_DEFAULTS = {
    "depth_profile": "1280x720x30",
    "color_profile": "848x480x30",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in (*_NODE_DEFAULTS, *_CAMERA_DEFAULTS)
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
        DeclareLaunchArgument(
            "depth_profile",
            default_value=_CAMERA_DEFAULTS["depth_profile"],
            description="D435 深度流配置。",
        ),
        DeclareLaunchArgument(
            "color_profile",
            default_value=_CAMERA_DEFAULTS["color_profile"],
            description="D435 彩色流配置。",
        ),
    ]


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                )
            ]
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "depth_module.depth_profile": _LAUNCH_CONFIGURATIONS["depth_profile"],
            "rgb_camera.color_profile": _LAUNCH_CONFIGURATIONS["color_profile"],
            "pointcloud.enable": "false",
            "align_depth.enable": "true",
            "enable_sync": "true",
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
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
                        "model_path": os.path.join(
                            get_package_share_directory("yolo_perception"),
                            "models",
                            "yolo-obb-1280.pt",
                        ),
                        "device": _LAUNCH_CONFIGURATIONS["device"],
                        "conf": _LAUNCH_CONFIGURATIONS["conf"],
                        "imgsz": _LAUNCH_CONFIGURATIONS["imgsz"],
                    }
                ],
            )
        ],
    )
    return LaunchDescription(
        [*_declare_launch_arguments(), realsense_launch, yolo_detector_node_obb]
    )
