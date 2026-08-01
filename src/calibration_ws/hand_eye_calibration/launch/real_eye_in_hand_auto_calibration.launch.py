import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    handeye_share = get_package_share_directory("hand_eye_calibration")
    default_storage_directory = str(
        Path.home()
        / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"
    )

    calibrate = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(handeye_share, "launch", "calibrate.launch.py")
        ),
        launch_arguments={
            "calibration_type": "eye_in_hand",
            "camera_type": LaunchConfiguration("camera_type"),
            "calibration_name": LaunchConfiguration("calibration_name"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_sim_time": "false",
            "storage_directory": LaunchConfiguration("storage_directory"),
        }.items(),
    )

    collector = TimerAction(
        period=15.0,
        actions=[
            Node(
                package="hand_eye_calibration",
                executable="auto_calibration_collector.py",
                name="auto_calibration_collector",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    os.path.join(handeye_share, "config", "auto_calibration_collector.yaml"),
                    {
                        "use_sim_time": False,
                        "auto_start": LaunchConfiguration("auto_start"),
                        "move_group_ns_fairino": "",
                        "move_group_ns_kdl": "",
                        "ik_plugin": "kdl",
                        "planning_pipeline_id": "ompl",
                        "planner_id": "RRTConnectFast",
                    },
                ],
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_type",
                default_value="realsense",
                choices=["realsense", "oak"],
            ),
            DeclareLaunchArgument("calibration_name", default_value="robot_calibration"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "storage_directory",
                default_value=default_storage_directory,
            ),
            DeclareLaunchArgument(
                "auto_start",
                default_value="false",
                description="Move only after explicit authorization; default is standby.",
            ),
            calibrate,
            collector,
        ]
    )
import os
from pathlib import Path
