import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.join(
            get_package_share_directory("yolo_perception"),
            "models",
            "best_stone.pt",
        ),
        description="Path to YOLOv8 model file.",
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="auto",
        description="Device for YOLOv8 inference (cpu or cuda:0).",
    )
    conf_threshold_arg = DeclareLaunchArgument(
        "conf",
        default_value="0.6",
        description="Confidence threshold for detections.",
    )
    imgsz_arg = DeclareLaunchArgument(
        "imgsz",
        default_value="640",
        description="Input image size for YOLOv8.",
    )

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
            "depth_module.profile": "640x480x30",
            "rgb_camera.profile": "640x480x30",
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
                        "model_path": LaunchConfiguration("model_path"),
                        "device": LaunchConfiguration("device"),
                        "conf": LaunchConfiguration("conf"),
                        "imgsz": LaunchConfiguration("imgsz"),
                    }
                ],
            )
        ],
    )

    return LaunchDescription(
        [
            model_path_arg,
            device_arg,
            conf_threshold_arg,
            imgsz_arg,
            realsense_launch,
            yolo_detector_node_obb,
        ]
    )
