from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("use_continuous_yolo", default_value="true"),
        Node(
            package="yolo_perception",
            executable="llm_yolo_perception.py",
            parameters=[{
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "use_continuous_yolo": LaunchConfiguration("use_continuous_yolo"),
            }],
        ),
    ])
