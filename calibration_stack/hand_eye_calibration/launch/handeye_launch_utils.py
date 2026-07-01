import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _load_yaml() -> dict:
    path = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "handeye_profiles.yaml",
    )
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_handeye_profile(calibration_type: str) -> dict:
    data = _load_yaml()
    profiles = data.get("profiles", {})
    if calibration_type not in profiles:
        raise RuntimeError(
            f"Unknown calibration_type '{calibration_type}'. "
            f"Available profiles: {sorted(profiles.keys())}"
        )
    return dict(profiles[calibration_type])


def default_from_settings(key: str, fallback: str) -> str:
    try:
        data = _load_yaml()
        return str(data.get("settings", {}).get(key, fallback))
    except Exception:
        return fallback


def value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context)).strip()


def profile_value(context, profile: dict, name: str) -> str:
    override = value(context, name)
    if override:
        return override
    return str(profile.get(name, ""))


def camera_launch(camera_type: str):
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
