import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
            gazebo_launch,
            retime_server_launch,
            yolo_obb,
            pose_monitor_node,
            task_server_node,
            motion_control_node,
        ]
    )
