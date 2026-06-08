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
    """Read default value from YAML settings section, falling back to hardcoded."""
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
    robot_base_frame = _profile_value(context, profile, "robot_base_frame")
    robot_effector_frame = _profile_value(context, profile, "robot_effector_frame")
    tracking_base_frame = _profile_value(context, profile, "tracking_base_frame")
    tracking_marker_frame = _profile_value(context, profile, "tracking_marker_frame")
    marker_id = int(_profile_value(context, profile, "marker_id"))

    rviz_config_file = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "rviz",
        str(profile.get("rviz_config", "moveit_with_camera.rviz")),
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
            "use_rviz": use_rviz,
            "rviz_config": rviz_config_file,
        }.items(),
    )

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

    calibration_aruco_publisher = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        parameters=[
            {
                "tracking_base_frame": tracking_base_frame,
                "tracking_marker_frame": tracking_marker_frame,
                "marker_id": marker_id,
            }
        ],
    )

    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("easy_handeye2"),
                    "launch",
                    "calibrate.launch.py",
                )
            ]
        ),
        launch_arguments={
            "name": calibration_name,
            "calibration_type": calibration_type,
            "robot_base_frame": robot_base_frame,
            "robot_effector_frame": robot_effector_frame,
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
        }.items(),
    )

    aruco_visualize = Node(
        package="hand_eye_calibration",
        executable="visualize_aruco_marker.py",
        name="aruco_pose_estimator",
        output="screen",
    )

    return [
        _camera_launch(camera_type),
        ar_moveit,
        aruco_recognition_node,
        calibration_aruco_publisher,
        easy_handeye2,
        aruco_visualize,
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
            DeclareLaunchArgument("robot_base_frame", default_value="", description="Override profile robot base frame."),
            DeclareLaunchArgument("robot_effector_frame", default_value="", description="Override profile robot effector frame."),
            DeclareLaunchArgument("tracking_base_frame", default_value="", description="Override profile tracking base frame."),
            DeclareLaunchArgument("tracking_marker_frame", default_value="", description="Override profile tracking marker frame."),
            DeclareLaunchArgument("marker_id", default_value="", description="Override profile ArUco marker id."),
            DeclareLaunchArgument("use_rviz", default_value="true", description="Forwarded to MoveIt demo launch."),
            OpaqueFunction(function=_launch_setup),
        ]
    )
