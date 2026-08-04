import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}
USE_SIM_TIME = True

ROBOT_BASE_FRAME = "base_link"
ROBOT_EFFECTOR_FRAME = "grasp_frame"
TRACKING_BASE_FRAME = "camera_color_optical_frame"
TRACKING_MARKER_FRAME = "calibration_aruco"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
ARUCO_DICTIONARY_ID = "DICT_5X5_250"
MARKER_ID = 1
MARKER_SIZE_M = 0.07

GAZEBO_LAUNCH_ARGUMENTS = {
    "robot_profile": "fairino_arm_gripper_handeye",
    "world": "calibration_table",
    "enable_rviz": "true",
    "use_sim_time": "true",
    "publish_frequency": "30.0",
    "enable_camera_model": "true",
    "enable_camera_bridge": "true",
    "camera_info_remap": CAMERA_INFO_TOPIC,
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

VISUALIZE_ARUCO_PARAMS = {
    "use_sim_time": USE_SIM_TIME,
    "image_topic": IMAGE_TOPIC,
    "camera_info_topic": CAMERA_INFO_TOPIC,
    "output_topic": "/aruco_image",
    "marker_size": MARKER_SIZE_M,
    "aruco_dictionary_id": ARUCO_DICTIONARY_ID,
}


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            **GAZEBO_LAUNCH_ARGUMENTS,
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "camera_profile": LaunchConfiguration("camera_profile"),
            "camera_profile_file": LaunchConfiguration("camera_profile_file"),
            "camera_noise_mode": LaunchConfiguration("camera_noise_mode"),
            "camera_depth_far_m": LaunchConfiguration("camera_depth_far_m"),
            "rviz_config": os.path.join(gz_share, "rviz", "calibration_gazebo.rviz"),
        }.items(),
    )

    marker_spawn = TimerAction(
        period=10.0,
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
                    "-x",
                    "0.35",
                    "-y",
                    "0.0",
                    "-z",
                    "1.03",
                    "-R",
                    "1.5708",
                    "-P",
                    "0.0",
                    "-Y",
                    "0.0",
                    "-allow_renaming",
                    "false",
                ],
            )
        ],
    )

    aruco_visualizer = Node(
        package="hand_eye_calibration",
        executable="visualize_aruco_marker.py",
        name="aruco_pose_estimator",
        output="screen",
        additional_env=PYTHON_NO_USER_SITE_ENV,
        parameters=[VISUALIZE_ARUCO_PARAMS],
    )

    actions = [
        DeclareLaunchArgument(
            "enable_rviz",
            default_value="true",
            description="Show RViz while starting the calibration simulation.",
        ),
        DeclareLaunchArgument(
            "camera_profile",
            default_value="d435_color_1280x720x30_depth_848x480x30",
            description="Named D435 profile for the calibration camera simulation.",
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
        marker_spawn,
        aruco_visualizer,
    ]

    return LaunchDescription(actions)
