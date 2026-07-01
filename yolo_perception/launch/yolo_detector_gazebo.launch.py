import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.join(
            get_package_share_directory("yolo_perception"),
            "models",
            "yolo-obb-gazebo.pt",
        ),
        description=(
            "Path to YOLOv8 model file. Defaults to yolo_perception package share; "
            "relative values are resolved by the node."
        ),
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
            yolo_detector_node_obb,
        ]
    )
