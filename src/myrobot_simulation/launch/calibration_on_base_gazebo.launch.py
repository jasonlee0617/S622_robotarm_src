"""Eye-on-base Gazebo calibration with a flange-mounted ArUco board."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


CALIBRATION_BOARD_MOUNT_DEFAULTS = {
    "calibration_board_x": "0.00",
    "calibration_board_y": "-0.05",
    "calibration_board_z": "-0.03",
    "calibration_board_roll": "1.5708",
    "calibration_board_pitch": "0",
    "calibration_board_yaw": "0",
}

# Eye-on-Base 场景专用的可覆盖参数；标定板安装偏置仍转发给通用场景入口。
_LAUNCH_ARGUMENT_SPECS = (
    ("camera_profile", "d435_color_640x480x30_depth_640x480x30", "Eye-on-Base 场景使用的 D435 配置。"),
    ("camera_fps", "30", "仿真相机帧率。"),
)


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(name, default_value=default_value, description=description)
        for name, default_value, description in _LAUNCH_ARGUMENT_SPECS
    ]


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    calibration = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "calibration_gazebo.launch.py")
        ),
        launch_arguments={
            "robot_profile": "fairino_arm_gripper_eye_on_base",
            "spawn_fixed_board": "false",
            "camera_profile": LaunchConfiguration("camera_profile"),
            "camera_fps": LaunchConfiguration("camera_fps"),
            **{
                name: LaunchConfiguration(name)
                for name in CALIBRATION_BOARD_MOUNT_DEFAULTS
            },
        }.items(),
    )

    return LaunchDescription([
        *_declare_launch_arguments(),
        *[
            DeclareLaunchArgument(name, default_value=default)
            for name, default in CALIBRATION_BOARD_MOUNT_DEFAULTS.items()
        ],
        calibration,
    ])
