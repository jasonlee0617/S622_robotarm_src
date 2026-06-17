import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml() -> dict:
    path = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "handeye_profiles.yaml",
    )
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _load_profile(calibration_type: str) -> dict:
    data = _load_yaml()
    profiles = data.get("profiles", {})
    if calibration_type not in profiles:
        raise RuntimeError(
            f"Unknown calibration_type '{calibration_type}'. "
            f"Available profiles: {sorted(profiles.keys())}"
        )
    return dict(profiles[calibration_type])


def _default_from_settings(key: str, fallback: str) -> str:
    try:
        data = _load_yaml()
        return str(data.get("settings", {}).get(key, fallback))
    except Exception:
        return fallback


def _value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context)).strip()


def _profile_value(context, profile: dict, name: str) -> str:
    override = _value(context, name)
    if override:
        return override
    return str(profile.get(name, ""))


def _camera_launch(camera_type: str):
    if camera_type == "oak":
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("depthai_ros_driver"),
                    "launch",
                    "camera.launch.py",
                )
            ),
            launch_arguments={"rs_compat": "true"}.items(),
        )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                )
            ]
        )
    )


def _launch_setup(context, *args, **kwargs):
    calibration_type = _value(context, "calibration_type")
    camera_type = _value(context, "camera_type")
    use_rviz = _value(context, "use_rviz").lower()
    profile = _load_profile(calibration_type)

    calibration_name = _profile_value(context, profile, "calibration_name")
    camera_link_frame = _profile_value(context, profile, "camera_link_frame")
    publish_child_frame = _profile_value(context, profile, "publish_camera_link_frame")
    tracking_base_frame = _profile_value(context, profile, "tracking_base_frame")
    marker_id = int(_profile_value(context, profile, "marker_id"))
    use_compensation = str(profile.get("use_tracking_to_camera_link_compensation", True)).lower()

    aruco_params = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "aruco_parameters.yaml",
    )
    aruco_recognition_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[aruco_params],
        output="screen",
    )

    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[
            {
                "calibration_name": calibration_name,
                "camera_link_frame": camera_link_frame,
                "publish_child_frame": publish_child_frame,
                "publish_rate_hz": 10.0,
                "use_tracking_to_camera_link_compensation": use_compensation in ("1", "true", "yes"),
            }
        ],
        output="screen",
    )

    follow_aruco_node = Node(
        package="hand_eye_calibration",
        executable="follow_aruco_marker.py",
        name="follow_aruco_marker",
        output="screen",
    )

    rviz_config_file = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "rviz",
        "validate.rviz",
    )
    ar_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("s622_moveit_config"),
                    "launch",
                    "demo.launch.py",
                )
            ]
        ),
        launch_arguments={
            "include_gripper": "False",
            "use_rviz": use_rviz,
            "rviz_config": rviz_config_file,
        }.items(),
    )

    return [
        _camera_launch(camera_type),
        hand_eye_tf_publisher,
        aruco_recognition_node,
        follow_aruco_node,
        ar_moveit,
    ]


def generate_launch_description():
    default_calib_type = _default_from_settings("calibration_type", "eye_on_base")
    default_camera_type = _default_from_settings("camera_type", "realsense")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "calibration_type",
                default_value=default_calib_type,
                choices=["eye_on_base", "eye_in_hand"],
                description="Hand-eye calibration mode (default from handeye_profiles.yaml settings).",
            ),
            DeclareLaunchArgument(
                "camera_type",
                default_value=default_camera_type,
                choices=["realsense", "oak"],
                description="Camera type (default from handeye_profiles.yaml settings).",
            ),
            DeclareLaunchArgument("calibration_name", default_value="", description="Override profile calibration name."),
            DeclareLaunchArgument("camera_link_frame", default_value="", description="Override profile camera link frame."),
            DeclareLaunchArgument("publish_camera_link_frame", default_value="", description="Override profile published child frame."),
            DeclareLaunchArgument("tracking_base_frame", default_value="", description="Reserved profile override."),
            DeclareLaunchArgument("marker_id", default_value="", description="Reserved profile override."),
            DeclareLaunchArgument("use_rviz", default_value="true", description="Forwarded to MoveIt demo launch."),
            OpaqueFunction(function=_launch_setup),
        ]
    )
