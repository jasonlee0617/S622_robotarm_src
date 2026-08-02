import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
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
            "camera_fps": "60",
            "camera_image_width": "640",
            "camera_image_height": "480",
            "camera_profile": LaunchConfiguration("camera_profile"),
            "camera_profile_file": LaunchConfiguration("camera_profile_file"),
            "camera_noise_mode": LaunchConfiguration("camera_noise_mode"),
            "camera_depth_far_m": LaunchConfiguration("camera_depth_far_m"),
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_profile",
            default_value="d435_color_640x480x30_depth_640x480x30",
            description="Named D435 profile for the LLM control camera simulation.",
        ),
        DeclareLaunchArgument(
            "camera_profile_file",
            default_value="",
            description="External D435 profile YAML; set camera_profile:='' when using it.",
        ),
        DeclareLaunchArgument(
            "camera_noise_mode",
            default_value="off",
            choices=["off", "d435_empirical"],
        ),
        DeclareLaunchArgument(
            "camera_depth_far_m",
            default_value="3.0",
            description="D435 depth far clip in metres; valid up to 10.0.",
        ),
        gazebo,
        pose_monitor,
        pose_control,
    ])
