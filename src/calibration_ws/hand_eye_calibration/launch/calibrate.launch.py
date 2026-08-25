"""实机标定环境：相机、ArUco、标记 TF、MoveIt 与可视化."""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_LAUNCH_DIR = os.path.dirname(__file__)
if _LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _LAUNCH_DIR)

from handeye_launch_utils import camera_launch, default_from_settings, load_handeye_profile
from handeye_launch_utils import profile_value


PYTHON_NO_USER_SITE_ENV = {"PYTHONNOUSERSITE": "1"}

# 集中管理实机标定环境入口的启动默认值；手眼 profile 仍只在
# handeye_launch_utils.py 中定义，不能在此重复坐标系配置。
_LAUNCH_DEFAULTS = {
    "calibration_type": default_from_settings("calibration_type", "eye_in_hand"),
    "camera_type": default_from_settings("camera_type", "realsense"),
    "camera_serial_no": "",
    "color_profile": "1280x720x30",
    "depth_profile": "848x480x30",
    "tracking_base_frame": "",
    "tracking_marker_frame": "",
    "marker_id": "",
    "use_rviz": "true",
    "active_executor": "fairino",
    "debug": "false",
    "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false",
    "capabilities": "",
    "disable_capabilities": "",
    "publish_frequency": "100.0",
}

_LAUNCH_CHOICES = {
    "calibration_type": ["eye_in_hand", "eye_on_base"],
    "camera_type": ["realsense", "oak"],
    "active_executor": ["fairino", "kdl"],
}

_LAUNCH_DESCRIPTIONS = {
    "calibration_type": "标定类型；坐标系由内置 profile 在运行时派生。",
    "camera_type": "相机驱动类型。",
    "camera_serial_no": "RealSense 设备序列号；留空时自动选择设备。",
    "color_profile": "RealSense 彩色流 profile，例如 1280x720x30。",
    "depth_profile": "RealSense 深度流 profile，例如 848x480x30。",
    "tracking_base_frame": "视觉跟踪基准坐标系覆盖；留空时使用 profile。",
    "tracking_marker_frame": "视觉标记坐标系覆盖；留空时使用 profile。",
    "marker_id": "ArUco 编号覆盖；留空时使用 profile。",
    "use_rviz": "是否启动 MoveIt RViz。",
    "active_executor": "唯一允许真实执行轨迹的 MoveIt 实例。",
    "debug": "是否开启 MoveIt 调试日志。",
    "allow_trajectory_execution": "是否允许活动 MoveIt 执行真实轨迹。",
    "publish_monitored_planning_scene": "是否发布 monitored planning scene。",
    "monitor_dynamics": "是否监控关节动力学状态。",
    "capabilities": "附加 MoveIt capability 列表。",
    "disable_capabilities": "禁用的 MoveIt capability 列表。",
    "publish_frequency": "robot_state_publisher 发布频率。",
}

_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_DEFAULTS
}


def _launch_value(context, name: str) -> str:
    """读取本入口已集中声明的 launch 参数。"""
    return str(_LAUNCH_CONFIGURATIONS[name].perform(context)).strip()


def _launch_setup(context, *_args, **_kwargs):
    calibration_type = _launch_value(context, "calibration_type")
    profile = load_handeye_profile(calibration_type)
    handeye_share = get_package_share_directory("hand_eye_calibration")
    tracking_base_frame = profile_value(context, profile, "tracking_base_frame")
    tracking_marker_frame = profile_value(context, profile, "tracking_marker_frame")
    marker_id = int(profile_value(context, profile, "marker_id"))
    rviz_config_file = os.path.join(
        handeye_share, "rviz", str(profile.get("rviz_config", "calibrate.rviz"))
    )

    camera = camera_launch(
        _launch_value(context, "camera_type"),
        realsense_args={
            "serial_no": _launch_value(context, "camera_serial_no"),
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": _launch_value(context, "color_profile"),
            "depth_module.depth_profile": _launch_value(context, "depth_profile"),
            "align_depth.enable": "true",
        },
    )
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fairino_arm_moveit_config"),
                "launch",
                "moveit_hardware.launch.py",
            )
        ),
        launch_arguments={
            "use_rviz": _launch_value(context, "use_rviz"),
            "rviz_config": rviz_config_file,
            "active_executor": _launch_value(context, "active_executor"),
            "debug": _launch_value(context, "debug"),
            "allow_trajectory_execution": _launch_value(
                context, "allow_trajectory_execution"
            ),
            "publish_monitored_planning_scene": _launch_value(
                context, "publish_monitored_planning_scene"
            ),
            "monitor_dynamics": _launch_value(context, "monitor_dynamics"),
            "capabilities": _launch_value(context, "capabilities"),
            "disable_capabilities": _launch_value(context, "disable_capabilities"),
            "publish_frequency": _launch_value(context, "publish_frequency"),
        }.items(),
    )
    aruco_parameters = os.path.join(handeye_share, "config", "aruco_parameters.yaml")
    actions = [
        camera,
        moveit,
        Node(
            package="ros2_aruco",
            executable="aruco_node",
            parameters=[{"use_sim_time": False}, aruco_parameters],
            additional_env=PYTHON_NO_USER_SITE_ENV,
            output="screen",
        ),
        Node(
            package="hand_eye_calibration",
            executable="aruco_marker_pose_publisher.py",
            name="aruco_marker_pose_publisher",
            parameters=[{
                "marker_id": marker_id,
                "aruco_topic": "/aruco_markers",
                "use_sim_time": False,
            }],
            additional_env=PYTHON_NO_USER_SITE_ENV,
            output="screen",
        ),
        Node(
            package="hand_eye_calibration",
            executable="calibration_aruco_publisher.py",
            name="calibration_aruco_publisher",
            parameters=[{
                "tracking_base_frame": tracking_base_frame,
                "tracking_marker_frame": tracking_marker_frame,
                "marker_pose_topic": "/aruco_marker/pose",
                "use_sim_time": False,
            }],
            additional_env=PYTHON_NO_USER_SITE_ENV,
            output="screen",
        ),
        Node(
            package="hand_eye_calibration",
            executable="visualize_aruco_marker.py",
            name="aruco_pose_estimator",
            parameters=[{
                "image_topic": "/camera/camera/color/image_raw",
                "camera_info_topic": "/camera/camera/color/camera_info",
                "marker_size": 0.07,
                "aruco_dictionary_id": "DICT_5X5_250",
                "use_sim_time": False,
            }],
            additional_env=PYTHON_NO_USER_SITE_ENV,
            output="screen",
        ),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "calibration_type",
            default_value=_LAUNCH_DEFAULTS["calibration_type"],
            choices=_LAUNCH_CHOICES["calibration_type"],
            description=_LAUNCH_DESCRIPTIONS["calibration_type"],
        ),
        DeclareLaunchArgument(
            "camera_type",
            default_value=_LAUNCH_DEFAULTS["camera_type"],
            choices=_LAUNCH_CHOICES["camera_type"],
            description=_LAUNCH_DESCRIPTIONS["camera_type"],
        ),
        *[
            DeclareLaunchArgument(
                name,
                default_value=_LAUNCH_DEFAULTS[name],
                choices=_LAUNCH_CHOICES.get(name),
                description=_LAUNCH_DESCRIPTIONS[name],
            )
            for name in (
                "camera_serial_no", "color_profile", "depth_profile",
                "tracking_base_frame", "tracking_marker_frame", "marker_id",
                "use_rviz",
            )
        ],
        DeclareLaunchArgument(
            "active_executor",
            default_value=_LAUNCH_DEFAULTS["active_executor"],
            choices=_LAUNCH_CHOICES["active_executor"],
            description=_LAUNCH_DESCRIPTIONS["active_executor"],
        ),
        *[
            DeclareLaunchArgument(
                name,
                default_value=_LAUNCH_DEFAULTS[name],
                description=_LAUNCH_DESCRIPTIONS[name],
            )
            for name in (
                "debug", "allow_trajectory_execution",
                "publish_monitored_planning_scene", "monitor_dynamics",
                "capabilities", "disable_capabilities", "publish_frequency",
            )
        ],
        OpaqueFunction(function=_launch_setup),
    ])
