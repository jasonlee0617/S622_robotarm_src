import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            "world": "arm_on_the_table",
            "rviz_config": os.path.join(gz_share, "rviz", "gazebo_launch.rviz"),
            "publish_frequency": "100.0",
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

    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    yolo_obb = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("yolo_perception"),
                        "launch",
                        "yolov8_obb.launch.py",
                    )
                )
            )
        ],
    )
    pose_monitor_node = Node(
        package="llm_arm_control",
        executable="fairino_pose_monitor",
        name="llm_yolo_pose_monitor",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    task_server_node = Node(
        package="llm_arm_control",
        executable="llm_yolo_task_server",
        name="llm_yolo_task_server",
        output="screen",
        parameters=[
            os.path.join(
                get_package_share_directory("llm_arm_control"),
                "config",
                "llm_yolo_task_sim.yaml",
            )
        ],
    )
    motion_control_node = Node(
        package="manipulation_common",
        executable="motion_control",
        name="motion_control",
        output="screen",
        parameters=[{"command_burst_count": 1}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_profile",
                default_value="d435_color_640x480x30_depth_640x480x30",
                description="Named D435 profile for the LLM YOLO camera simulation.",
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
            gazebo_launch,
            retime_server_launch,
            yolo_obb,
            pose_monitor_node,
            task_server_node,
            motion_control_node,
        ]
    )
