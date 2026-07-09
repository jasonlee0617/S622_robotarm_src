import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_share = get_package_share_directory("gazebo_launch")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            "world": "arm_on_the_table",
            "rviz_config": os.path.join(gazebo_share, "rviz", "gazebo_launch.rviz"),
            "publish_frequency": "30.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "true",
            "spawn_z": "1.02",
            "controller_spawn_delay": "5.0",
        }.items(),
    )

    pose_monitor = Node(
        package="llm_arm_control",
        executable="fairino_pose_monitor",
        name="fairino_pose_monitor",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    pose_control = Node(
        package="llm_arm_control",
        executable="fairino_pose_control_server",
        name="fairino_pose_control_server",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([gazebo, pose_monitor, pose_control])
