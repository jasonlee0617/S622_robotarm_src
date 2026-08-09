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
    "eye_in_hand": MappingProxyType({
        "calibration_name": "robot_calibration",
        "calibration_type": "eye_in_hand",
        "robot_base_frame": "base_link",
        "robot_effector_frame": "grasp_frame",
        "tracking_base_frame": "camera_color_optical_frame",
        "tracking_marker_frame": "calibration_aruco",
        "marker_id": 1,
        "camera_link_frame": "camera_link",
        "camera_optical_frame": "camera_color_optical_frame",
        "publish_camera_link_frame": "camera_link",
        "use_tracking_to_camera_link_compensation": True,
        "rviz_config": "calibrate.rviz",
    }),
    "eye_on_base": MappingProxyType({
        "calibration_name": "robot_calibration",
        "calibration_type": "eye_on_base",
        "robot_base_frame": "base_link",
        "robot_effector_frame": "grasp_frame",
        "tracking_base_frame": "camera_color_optical_frame",
        "tracking_marker_frame": "calibration_aruco",
        "marker_id": 1,
        "camera_link_frame": "camera_link",
        "camera_optical_frame": "camera_color_optical_frame",
        "publish_camera_link_frame": "camera_link",
        "use_tracking_to_camera_link_compensation": True,
        "rviz_config": "calibrate.rviz",
    }),
})


def default_storage_directory(mode: str) -> str:
    return str(
        Path.home()
        / "fairino_robotarm"
        / "src"
        / "calibration_ws"
        / "hand_eye_calibration"
        / "calib"
        / mode
    )


def load_handeye_profile(calibration_type: str) -> dict:
    """返回指定标定类型的内置实机相机配置副本."""
    if calibration_type not in _HAND_EYE_PROFILES:
        raise RuntimeError(
            f"未知 calibration_type '{calibration_type}'；可用值：{sorted(_HAND_EYE_PROFILES)}"
        )
    return dict(_HAND_EYE_PROFILES[calibration_type])


def default_from_settings(key: str, fallback: str) -> str:
    """返回内置启动默认值，不读取运行时 YAML profile."""
    return str(_DEFAULT_SETTINGS.get(key, fallback))


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


def camera_launch(camera_type: str, *, realsense_args=None):
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
        ),
        launch_arguments=(realsense_args or {}).items(),
    )
