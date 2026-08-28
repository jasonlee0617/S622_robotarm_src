import os
from pathlib import Path
from types import MappingProxyType

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_DEFAULT_SETTINGS = MappingProxyType({
    "calibration_type": "eye_in_hand",
    "camera_type": "realsense",
})
_HAND_EYE_PROFILES = MappingProxyType({
    calibration_type: MappingProxyType({
        "calibration_name": "robot_calibration",
        "robot_base_frame": "base_link",
        "robot_effector_frame": "tool0",
        "tracking_base_frame": "camera_color_optical_frame",
        "tracking_marker_frame": "calibration_aruco",
        "marker_id": 1,
        "camera_link_frame": "camera_link",
        "camera_optical_frame": "camera_color_optical_frame",
        "publish_camera_link_frame": "camera_link",
        "use_tracking_to_camera_link_compensation": True,
        "rviz_config": "calibrate.rviz",
    })
    for calibration_type in ("eye_in_hand", "eye_on_base")
})


def default_storage_directory(mode: str) -> str:
    return str(
        Path.home() / "fairino_robotarm" / "src" / "calibration_ws"
        / "hand_eye_calibration" / "calib" / mode
    )


def load_handeye_profile(calibration_type: str) -> dict:
    try:
        profile = _HAND_EYE_PROFILES[calibration_type]
    except KeyError as exc:
        raise RuntimeError(
            f"未知 calibration_type '{calibration_type}'；可用值：{sorted(_HAND_EYE_PROFILES)}"
        ) from exc
    return {"calibration_type": calibration_type, **profile}


def default_from_settings(key: str, fallback: str) -> str:
    return str(_DEFAULT_SETTINGS.get(key, fallback))


def value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context)).strip()


def profile_value(context, profile: dict, name: str) -> str:
    return str(profile.get(name, ""))


def camera_launch(camera_type: str, *, realsense_args=None, oak_args=None):
    if camera_type == "oak":
        arguments = {"rs_compat": "true", **(oak_args or {})}
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("depthai_ros_driver"), "launch", "camera.launch.py"
            )),
            launch_arguments=arguments.items(),
        )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"
        )]),
        launch_arguments=(realsense_args or {}).items(),
    )
