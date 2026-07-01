import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.launch_parsing import as_bool
from manipulation_common.launch_utils.yaml_loader import load_ros_parameters_yaml


_GAZEBO_SHARE = get_package_share_directory("gazebo_launch")
_COLLECTOR_DEFAULTS = load_ros_parameters_yaml(
    "hand_eye_calibration",
    "config/auto_calibration_collector.yaml",
    "auto_calibration_collector",
)
_PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}


def _collector_default(name: str, fallback: str) -> str:
    return str(_COLLECTOR_DEFAULTS.get(name, fallback))


_CALIBRATION_GAZEBO_DEFAULTS = {
    "robot_profile": "s622_gripper_handeye",
    "world": "calibration_table",
    "rviz_config": os.path.join(_GAZEBO_SHARE, "rviz", "calibration_gazebo.rviz"),
    "enable_rviz": "true",
    "use_sim_time": "true",
    "publish_frequency": "30.0",
    "default_planning_pipeline": "fairino",
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
    "calibration_name": "robot_calibration",
    "robot_base_frame": _collector_default("base_frame", "base_link"),
    "robot_effector_frame": _collector_default("ee_frame", "grasp_frame"),
    "tracking_base_frame": _collector_default(
        "tracking_base_frame", "camera_color_optical_frame"
    ),
    "tracking_marker_frame": _collector_default(
        "tracking_marker_frame", "calibration_aruco"
    ),
    "marker_id": _collector_default("marker_id", "1"),
    "marker_size": _collector_default("marker_size_m", "0.07"),
    "marker_model_name": "calibration_aruco_board",
    "marker_x": "0.25",
    "marker_y": "0.0",
    "marker_z": "1.02",
    "marker_roll": "1.5708",
    "marker_pitch": "0.0",
    "marker_yaw": "0.0",
    "marker_spawn_delay": "10.0",
    "visualize_aruco": "true",
    "easy_handeye2_delay": "12.0",
    "aruco_tf_stamp_policy": "now",
    "aruco_tf_log_every_sec": "5.0",
    "image_topic": _collector_default(
        "image_topic", "/camera/camera/color/image_raw"
    ),
    "camera_info_remap": _collector_default(
        "camera_info_topic", "/camera/camera/color/camera_info"
    ),
    "camera_info_topic": _collector_default(
        "camera_info_topic", "/camera/camera/color/camera_info"
    ),
    "camera_fps": _collector_default("camera_fps", "30"),
    "camera_image_width": _collector_default("camera_image_width", "1280"),
    "camera_image_height": _collector_default("camera_image_height", "720"),
    "aruco_dictionary_id": _collector_default(
        "aruco_dictionary_id", "DICT_5X5_250"
    ),
    "auto_collect": "false",
    "auto_collector_delay": "15.0",
    "auto_collector_ik_plugin": _collector_default("ik_plugin", "fairino"),
    "auto_collector_planning_pipeline": _collector_default(
        "planning_pipeline_id", "fairino"
    ),
    "auto_collector_planner_id": _collector_default("planner_id", "birrt*"),
}


def _declare_launch_arguments(defaults: dict):
    return [
        DeclareLaunchArgument(name, default_value=str(value))
        for name, value in defaults.items()
    ]


def _value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context)).strip()


def _float_value(context, name: str) -> float:
    return float(_value(context, name))


def _launch_setup(context, *args, **kwargs):
    gz_share = get_package_share_directory("gazebo_launch")
    handeye_share = get_package_share_directory("hand_eye_calibration")
    easy_share = get_package_share_directory("easy_handeye2")

    use_sim_time = as_bool(_value(context, "use_sim_time"))
    marker_id = int(_value(context, "marker_id"))

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo_yolo.launch.py")
        ),
        launch_arguments={
            "robot_profile": _value(context, "robot_profile"),
            "world": _value(context, "world"),
            "rviz_config": _value(context, "rviz_config"),
            "enable_rviz": _value(context, "enable_rviz"),
            "use_sim_time": _value(context, "use_sim_time"),
            "publish_frequency": _value(context, "publish_frequency"),
            "default_planning_pipeline": _value(context, "default_planning_pipeline"),
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "camera_info_remap": _value(context, "camera_info_remap"),
            "camera_fps": _value(context, "camera_fps"),
            "camera_image_width": _value(context, "camera_image_width"),
            "camera_image_height": _value(context, "camera_image_height"),
            "enable_servo": _value(context, "enable_servo"),
            "spawn_name": _value(context, "spawn_name"),
            "spawn_x": _value(context, "spawn_x"),
            "spawn_y": _value(context, "spawn_y"),
            "spawn_z": _value(context, "spawn_z"),
            "spawn_roll": _value(context, "spawn_roll"),
            "spawn_pitch": _value(context, "spawn_pitch"),
            "spawn_yaw": _value(context, "spawn_yaw"),
            "robot_spawn_delay": _value(context, "robot_spawn_delay"),
            "controller_spawn_delay": _value(context, "controller_spawn_delay"),
        }.items(),
    )

    marker_sdf = os.path.join(
        gz_share,
        "worlds",
        "models",
        "aruco_5x5_250_id1",
        "model.sdf",
    )
    marker_spawn = TimerAction(
        period=_float_value(context, "marker_spawn_delay"),
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-file",
                    marker_sdf,
                    "-name",
                    _value(context, "marker_model_name"),
                    "-x",
                    _value(context, "marker_x"),
                    "-y",
                    _value(context, "marker_y"),
                    "-z",
                    _value(context, "marker_z"),
                    "-R",
                    _value(context, "marker_roll"),
                    "-P",
                    _value(context, "marker_pitch"),
                    "-Y",
                    _value(context, "marker_yaw"),
                    "-allow_renaming",
                    "false",
                ],
            )
        ],
    )

    aruco_params = os.path.join(
        handeye_share,
        "config",
        "aruco_parameters_gazebo.yaml",
    )
    aruco_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[aruco_params, {"use_sim_time": use_sim_time}],
        additional_env=_PYTHON_NO_USER_SITE_ENV,
        output="screen",
    )

    tracking_base_frame = _value(context, "tracking_base_frame")
    tracking_marker_frame = _value(context, "tracking_marker_frame")
    aruco_tf = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        additional_env=_PYTHON_NO_USER_SITE_ENV,
        parameters=[
            {
                "tracking_base_frame": tracking_base_frame,
                "tracking_marker_frame": tracking_marker_frame,
                "marker_id": marker_id,
                "aruco_topic": "/aruco_markers",
                "stamp_policy": _value(context, "aruco_tf_stamp_policy"),
                "log_every_sec": _float_value(context, "aruco_tf_log_every_sec"),
                "use_sim_time": use_sim_time,
            }
        ],
    )

    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(easy_share, "launch", "calibrate.launch.py")
        ),
        launch_arguments={
            "name": _value(context, "calibration_name"),
            "calibration_type": "eye_in_hand",
            "robot_base_frame": _value(context, "robot_base_frame"),
            "robot_effector_frame": _value(context, "robot_effector_frame"),
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
            "use_sim_time": _value(context, "use_sim_time"),
        }.items(),
    )

    actions = [
        gazebo,
        marker_spawn,
        aruco_node,
        aruco_tf,
        TimerAction(
            period=_float_value(context, "easy_handeye2_delay"),
            actions=[easy_handeye2],
        ),
    ]

    if as_bool(_value(context, "auto_collect")):
        auto_params = os.path.join(
            handeye_share,
            "config",
            "auto_calibration_collector.yaml",
        )
        launch_fallback_params = {
            "use_sim_time": use_sim_time,
            "base_frame": _value(context, "robot_base_frame"),
            "ee_frame": _value(context, "robot_effector_frame"),
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
            "marker_id": marker_id,
            "marker_size_m": _float_value(context, "marker_size"),
            "image_topic": _value(context, "image_topic"),
            "camera_info_topic": _value(context, "camera_info_topic"),
            "aruco_dictionary_id": _value(context, "aruco_dictionary_id"),
            "ik_plugin": _value(context, "auto_collector_ik_plugin"),
            "planning_pipeline_id": _value(
                context, "auto_collector_planning_pipeline"
            ),
            "planner_id": _value(context, "auto_collector_planner_id"),
        }
        actions.append(
            TimerAction(
                period=_float_value(context, "auto_collector_delay"),
                actions=[
                    Node(
                        package="hand_eye_calibration",
                        executable="auto_calibration_collector.py",
                        name="auto_calibration_collector",
                        output="screen",
                        additional_env=_PYTHON_NO_USER_SITE_ENV,
                        parameters=[
                            launch_fallback_params,
                            auto_params,
                        ],
                    )
                ],
            )
        )

    if as_bool(_value(context, "visualize_aruco")):
        actions.append(
            Node(
                package="hand_eye_calibration",
                executable="visualize_aruco_marker.py",
                name="aruco_pose_estimator",
                output="screen",
                parameters=[
                    {
                        "image_topic": _value(context, "image_topic"),
                        "camera_info_topic": _value(context, "camera_info_topic"),
                        "marker_size": _float_value(context, "marker_size"),
                        "aruco_dictionary_id": _value(context, "aruco_dictionary_id"),
                        "use_sim_time": use_sim_time,
                    }
                ],
                additional_env=_PYTHON_NO_USER_SITE_ENV,
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            *_declare_launch_arguments(_CALIBRATION_GAZEBO_DEFAULTS),
            OpaqueFunction(function=_launch_setup),
        ]
    )
