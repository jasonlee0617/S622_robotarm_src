from launch import LaunchDescription
from launch_ros.actions import Node


# 此节点没有可变运行时标量；相机话题固定映射到 Gazebo 数据流。

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="yolo_perception",
            executable="yolov8_obb_publisher.py",
            remappings=[("/image_raw", "/camera/camera/color/image_raw")],
        ),
    ])
