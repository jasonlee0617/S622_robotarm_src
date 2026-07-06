import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _load_yaml() -> dict:
    """
    从 hand_eye_calibration 包的 config 目录下加载 handeye_profiles.yaml 文件，
    并返回解析后的字典。若文件为空则返回空字典。
    """
    path = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "handeye_profiles.yaml",
    )
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_handeye_profile(calibration_type: str) -> dict:
    """
    根据 calibration_type（如 "eye_in_hand"、"eye_to_hand"）加载对应的标定配置字典。
    如果指定的类型不存在，则抛出 RuntimeError 并列出所有可用的配置文件名。
    """
    data = _load_yaml()
    profiles = data.get("profiles", {})
    if calibration_type not in profiles:
        raise RuntimeError(
            f"Unknown calibration_type '{calibration_type}'. "
            f"Available profiles: {sorted(profiles.keys())}"
        )
    return dict(profiles[calibration_type])


def default_from_settings(key: str, fallback: str) -> str:
    """
    从 YAML 文件的 settings 节中读取默认值。
    若 key 不存在或读取失败，则返回 fallback 字符串。
    该函数通常用于在启动文件中获取未通过 LaunchConfiguration 指定的配置项。
    """
    try:
        data = _load_yaml()
        return str(data.get("settings", {}).get(key, fallback))
    except Exception:
        return fallback


def value(context, name: str) -> str:
    """
    在启动文件的上下文（LaunchContext）中获取名为 name 的 LaunchConfiguration 的值，
    并去除前后空白。如果未设置，则返回空字符串。
    """
    return str(LaunchConfiguration(name).perform(context)).strip()


def profile_value(context, profile: dict, name: str) -> str:
    """
    优先从 LaunchConfiguration 中获取值（即允许命令行覆盖），
    若未提供则回退到 profile 字典中的对应值。
    """
    override = value(context, name)
    if override:
        return override
    return str(profile.get(name, ""))


def camera_launch(camera_type: str):
    """
    根据 camera_type 返回对应的相机驱动启动描述（IncludeLaunchDescription）。
    - "oak"：启动 OAK-D / DepthAI 相机驱动，并传入 rs_compat:=true 参数。
    - 其他值（默认）：启动 RealSense 相机驱动（rs_launch.py）。
    """
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
    # 默认启动 RealSense 相机
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