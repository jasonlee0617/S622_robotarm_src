import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}
USE_SIM_TIME = True
AUTO_COLLECT = False
VISUALIZE_ARUCO = True

ROBOT_BASE_FRAME = "base_link"
ROBOT_EFFECTOR_FRAME = "grasp_frame"
TRACKING_BASE_FRAME = "camera_color_optical_frame"
TRACKING_MARKER_FRAME = "calibration_aruco"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
ARUCO_TOPIC = "/aruco_markers"
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

EASY_HANDEYE_LAUNCH_ARGUMENTS = {
    "name": "robot_calibration",
    "calibration_type": "eye_in_hand",
    "robot_base_frame": ROBOT_BASE_FRAME,
    "robot_effector_frame": ROBOT_EFFECTOR_FRAME,
    "tracking_base_frame": TRACKING_BASE_FRAME,
    "tracking_marker_frame": TRACKING_MARKER_FRAME,
    "use_sim_time": "true",
}

ARUCO_TF_PARAMS = {
    "tracking_base_frame": TRACKING_BASE_FRAME,
    "tracking_marker_frame": TRACKING_MARKER_FRAME,
    "marker_id": MARKER_ID,
    "aruco_topic": ARUCO_TOPIC,
    "stamp_policy": "now",
    "log_every_sec": 5.0,
    "use_sim_time": USE_SIM_TIME,
}

COLLECTOR_SCENE_PARAMS = {
    "use_sim_time": USE_SIM_TIME,
    "base_frame": ROBOT_BASE_FRAME,
    "ee_frame": ROBOT_EFFECTOR_FRAME,
    "tracking_base_frame": TRACKING_BASE_FRAME,
    "tracking_marker_frame": TRACKING_MARKER_FRAME,
    "marker_id": MARKER_ID,
    "aruco_topic": ARUCO_TOPIC,
    "marker_size_m": MARKER_SIZE_M,
    "image_topic": IMAGE_TOPIC,
    "camera_info_topic": CAMERA_INFO_TOPIC,
    "aruco_dictionary_id": ARUCO_DICTIONARY_ID,
}

VISUALIZE_ARUCO_PARAMS = {
    "image_topic": IMAGE_TOPIC,
    "camera_info_topic": CAMERA_INFO_TOPIC,
    "marker_size": MARKER_SIZE_M,
    "aruco_dictionary_id": ARUCO_DICTIONARY_ID,
    "use_sim_time": USE_SIM_TIME,
}


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    handeye_share = get_package_share_directory("hand_eye_calibration")
    easy_share = get_package_share_directory("easy_handeye2")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            **GAZEBO_LAUNCH_ARGUMENTS,
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
                    "0.25",
                    "-y",
                    "0.0",
                    "-z",
                    "1.02",
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

    aruco_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[
            os.path.join(handeye_share, "config", "aruco_parameters.yaml"),
            {"use_sim_time": USE_SIM_TIME},
        ],
        additional_env=PYTHON_NO_USER_SITE_ENV,
        output="screen",
    )

    aruco_tf = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        additional_env=PYTHON_NO_USER_SITE_ENV,
        parameters=[
            ARUCO_TF_PARAMS,
        ],
    )

    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(easy_share, "launch", "calibrate.launch.py")
        ),
        launch_arguments=EASY_HANDEYE_LAUNCH_ARGUMENTS.items(),
    )

    actions = [
        gazebo,
        marker_spawn,
        aruco_node,
        aruco_tf,
        TimerAction(period=12.0, actions=[easy_handeye2]),
    ]

    if AUTO_COLLECT:
        actions.append(
            TimerAction(
                period=15.0,
                actions=[
                    Node(
                        package="hand_eye_calibration",
                        executable="auto_calibration_collector.py",
                        name="auto_calibration_collector",
                        output="screen",
                        additional_env=PYTHON_NO_USER_SITE_ENV,
                        parameters=[
                            os.path.join(
                                handeye_share,
                                "config",
                                "auto_calibration_collector.yaml",
                            ),
                            COLLECTOR_SCENE_PARAMS,
                        ],
                    )
                ],
            )
        )

    if VISUALIZE_ARUCO:
        actions.append(
            Node(
                package="hand_eye_calibration",
                executable="visualize_aruco_marker.py",
                name="aruco_pose_estimator",
                output="screen",
                parameters=[
                    VISUALIZE_ARUCO_PARAMS,
                ],
                additional_env=PYTHON_NO_USER_SITE_ENV,
            )
        )

    return LaunchDescription(actions)
