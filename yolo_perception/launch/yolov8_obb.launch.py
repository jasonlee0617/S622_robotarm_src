from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(package='yolo_perception',
             executable='yolov8_obb_publisher.py',
            #  output='screen'
            ),
    ])
