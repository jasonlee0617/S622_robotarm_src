from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='manipulation_common',
            executable='motion_control',
            name='motion_control',
            output='screen',
        ),
    ])
