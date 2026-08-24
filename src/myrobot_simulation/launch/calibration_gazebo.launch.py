"""Eye-in-hand Gazebo scene, vision pipeline, marker TF, and visualization."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}
ROBOT_BASE_FRAME = "base_link"
TRACKING_BASE_FRAME = "camera_color_optical_frame"
TRACKING_MARKER_FRAME = "calibration_aruco"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
ARUCO_TOPIC = "/aruco_markers"
ARUCO_DICTIONARY_ID = "DICT_5X5_250"
MARKER_ID = 1
MARKER_SIZE_M = 0.07

# 可由 CLI 覆盖的场景和视觉运行参数集中在此；固定资源路径保持包内引用。
_LAUNCH_ARGUMENT_SPECS = (
    ("enable_rviz", "true", "是否启动 RViz。", None),
    (
        "rviz_config",
        os.path.join(
            get_package_share_directory("myrobot_simulation"),
            "rviz",
            "calibration_gazebo.rviz",
        ),
        "RViz 配置文件。",
        None,
    ),
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock 仿真时间。", None),
    ("camera_profile", "d435_color_1280x720x30_depth_848x480x30", "仿真 D435 命名相机配置。", None),
    ("camera_profile_file", "", "外部 D435 配置 YAML。", None),
    ("camera_noise_mode", "off", "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "3.0", "D435 深度远裁剪距离，单位米。", None),
    ("camera_fps", "30", "仿真相机帧率。", None),
    ("robot_profile", "fairino_arm_gripper_inhand", "Gazebo 机器人配置。", None),
    ("enable_servo", "false", "是否启动 MoveIt Servo。", None),
    ("spawn_fixed_board", "true", "是否生成世界固定标定板。", None),
)


def _declare_launch_arguments():
    """创建场景和视觉参数的中文 launch 声明."""
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations

CALIBRATION_BOARD_MOUNT_DEFAULTS = {
    "calibration_board_x": "0.055",
    "calibration_board_y": "-0.050",
    "calibration_board_z": "0.2168",
    "calibration_board_roll": "0.0",
    "calibration_board_pitch": "1.5707963267948966",
    "calibration_board_yaw": "0.0",
}

GAZEBO_LAUNCH_ARGUMENTS = {
    "robot_profile": "fairino_arm_gripper_inhand",
    "world": "calibration_table",
    "enable_rviz": "true",
    "use_sim_time": "true",
    "publish_frequency": "30.0",
    "enable_camera_model": "true",
    "enable_camera_bridge": "true",
    "camera_fps": "30",
    "camera_image_width": "1280",
    "camera_image_height": "720",
    "enable_servo": "false",
    "spawn_name": "",
    "spawn_x": "0.0",
    "spawn_y": "0.0",
    "spawn_z": "1.02",
    "spawn_roll": "0.0",
    "spawn_pitch": "0.0",
    "spawn_yaw": "0.0",
    "robot_spawn_delay": "5.0",
    "controller_spawn_delay": "8.0",
}


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    handeye_share = get_package_share_directory("hand_eye_calibration")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            **GAZEBO_LAUNCH_ARGUMENTS,
            "robot_profile": LaunchConfiguration("robot_profile"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "camera_profile": LaunchConfiguration("camera_profile"),
            "camera_profile_file": LaunchConfiguration("camera_profile_file"),
            "camera_noise_mode": LaunchConfiguration("camera_noise_mode"),
            "camera_depth_far_m": LaunchConfiguration("camera_depth_far_m"),
            "camera_fps": LaunchConfiguration("camera_fps"),
            "enable_servo": LaunchConfiguration("enable_servo"),
            "rviz_config": LaunchConfiguration("rviz_config"),
            **{
                name: LaunchConfiguration(name)
                for name in CALIBRATION_BOARD_MOUNT_DEFAULTS
            },
        }.items(),
    )
    marker_spawn = TimerAction(
        period=10.0,
        condition=IfCondition(LaunchConfiguration("spawn_fixed_board")),
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-file",
                    os.path.join(
                        gz_share,
                        "worlds",
                        "models",
                        "aruco_5x5_250_id1",
                        "model.sdf",
                    ),
                    "-name",
                    "calibration_aruco_board",
                    "-x", "0.0", "-y", "0.38", "-z", "1.03",
                    "-R", "1.5708", "-P", "0.0", "-Y", "0.0",
                    "-allow_renaming", "false",
                ],
            )
        ],
    )
    aruco_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            os.path.join(handeye_share, "config", "aruco_parameters.yaml"),
        ],
        additional_env=PYTHON_NO_USER_SITE_ENV,
        output="screen",
    )
    aruco_tf = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        parameters=[{
            "tracking_base_frame": TRACKING_BASE_FRAME,
            "tracking_marker_frame": TRACKING_MARKER_FRAME,
            "marker_id": MARKER_ID,
            "aruco_topic": ARUCO_TOPIC,
            "stamp_policy": "now",
            "log_every_sec": 5.0,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
        additional_env=PYTHON_NO_USER_SITE_ENV,
        output="screen",
    )
    visualize = Node(
        package="hand_eye_calibration",
        executable="visualize_aruco_marker.py",
        name="aruco_pose_estimator",
        parameters=[{
            "image_topic": IMAGE_TOPIC,
            "camera_info_topic": CAMERA_INFO_TOPIC,
            "output_topic": "/aruco_image",
            "marker_size": MARKER_SIZE_M,
            "aruco_dictionary_id": ARUCO_DICTIONARY_ID,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
        additional_env=PYTHON_NO_USER_SITE_ENV,
        output="screen",
    )
    return LaunchDescription([
        *_declare_launch_arguments(),
        *[
            DeclareLaunchArgument(name, default_value=default)
            for name, default in CALIBRATION_BOARD_MOUNT_DEFAULTS.items()
        ],
        gazebo,
        marker_spawn,
        aruco_node,
        aruco_tf,
        visualize,
    ])
