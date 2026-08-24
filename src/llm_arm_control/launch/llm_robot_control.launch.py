#!/usr/bin/env python3
"""Real-hardware LLM Robot entry point with gated YOLO and GraspNet modes."""

import os
import shlex
import sys
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_HANDEYE_LAUNCH_DIR = os.path.join(
    get_package_share_directory("hand_eye_calibration"), "launch"
)
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)
from handeye_launch_utils import camera_launch, value  # noqa: E402


DEFAULTS = {
    "use_sim_time": "false",
    "camera_type": "realsense",
    "camera_serial_no": "",
    "color_profile": "1280x720x30",
    "depth_profile": "848x480x30",
    "pointcloud_enable": "false",
    "use_rviz": "true",
    "active_executor": "fairino",
    "debug": "false",
    "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false",
    "capabilities": "",
    "disable_capabilities": "",
    "publish_frequency": "100.0",
    "graspnet_model_profile": "rs",
    "command_burst_count": "1",
    "use_continuous_yolo": "true",
}


def _argument(name: str, default: str) -> DeclareLaunchArgument:
    """创建启动参数，按名称添加可选值约束。"""
    kwargs = {"default_value": default, "description": name}

    if name == "camera_type":
        kwargs["choices"] = ["realsense", "oak"]
    elif name == "active_executor":
        kwargs["choices"] = ["fairino", "kdl"]
    elif name == "graspnet_model_profile":
        kwargs["choices"] = ["rs", "kn"]

    return DeclareLaunchArgument(name, **kwargs)


def _graspnet_inference_process(context):
    """返回在 conda 环境中运行的 GraspNet 推理进程。"""
    llm_share = get_package_share_directory("llm_arm_control")
    source_share = get_package_share_directory("graspnet_source")
    install_setup = str(
        Path(get_package_prefix("graspnet_bringup")).parent / "setup.bash"
    )
    profile = value(context, "graspnet_model_profile")

    command = (
        "set -e; "
        f"source {shlex.quote(os.path.expanduser('~/miniconda3/etc/profile.d/conda.sh'))}; "
        "conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(install_setup)}; "
        "export PYTHONUNBUFFERED=1 "
        "MPLCONFIGDIR=/tmp/graspnet_mpl_config "
        "XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_bringup.graspnet_inference_node --ros-args "
        f"--params-file {shlex.quote(os.path.join(llm_share, 'config', 'llm_robot_control.yaml'))} "
        "-r __node:=graspnet_inference "
        f"-p use_sim_time:={value(context, 'use_sim_time')} "
        f"-p baseline_dir:={shlex.quote(os.path.join(source_share, 'graspnet_baseline'))} "
        f"-p checkpoint_path:={shlex.quote(os.path.join(source_share, 'models', f'checkpoint-{profile}.tar'))}"
    )

    return [ExecuteProcess(cmd=["bash", "-lc", command], output="screen")]


def _launch_setup(context):
    """组装真实硬件启动描述。"""
    llm_share = get_package_share_directory("llm_arm_control")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_continuous_yolo = LaunchConfiguration("use_continuous_yolo")
    command_burst_count = LaunchConfiguration("command_burst_count")

    # 相机驱动
    camera = camera_launch(
        value(context, "camera_type"),
        realsense_args={
            "serial_no": value(context, "camera_serial_no"),
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": value(context, "color_profile"),
            "depth_module.depth_profile": value(context, "depth_profile"),
            "align_depth.enable": "true",
            "enable_sync": "true",
            "pointcloud.enable": value(context, "pointcloud_enable"),
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
        },
        oak_args={
            "enable_color": "true",
            "enable_depth": "true",
        },
    )

    # MoveIt 真实机械臂
    moveit_launch_args = {
        name: LaunchConfiguration(name)
        for name in (
            "use_rviz",
            "active_executor",
            "debug",
            "allow_trajectory_execution",
            "publish_monitored_planning_scene",
            "monitor_dynamics",
            "capabilities",
            "disable_capabilities",
            "publish_frequency",
            "rviz_config",
        )
    }
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fairino_arm_moveit_config"),
                "launch",
                "moveit_hardware.launch.py",
            )
        ),
        launch_arguments=moveit_launch_args.items(),
    )

    # 手眼标定发布器
    shared_config = os.path.join(llm_share, "config", "llm_robot_control.yaml")
    handeye = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "calibration_name": "robot_calibration",
            "storage_directory": str(
                Path.home()
                / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"
            ),
        }],
    )

    # 轨迹时间重规划服务
    retime = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    # YOLO 感知
    yolo_obb = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("visual_perception"),
                        "launch",
                        "llm_visual_perception.launch.py",
                    )
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "use_continuous_yolo": use_continuous_yolo,
                }.items(),
            )
        ],
    )

    # 机器人位姿监控
    monitor = Node(
        package="llm_arm_control",
        executable="robot_pose_monitor_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # LLM 任务服务器
    task = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="llm_arm_control",
                executable="llm_control_task_server",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "use_continuous_yolo": use_continuous_yolo,
                    },
                    shared_config,
                ],
            )
        ],
    )

    # 独立终端中的 CLI
    cli = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "gnome-terminal",
                    "--title=LLM Robot CLI",
                    "--wait",
                    "--",
                    "ros2",
                    "run",
                    "llm_arm_control",
                    "llm_control_cli",
                    "--ros-args",
                    "-p",
                    ["use_sim_time:=", use_sim_time],
                    "-p",
                    ["command_burst_count:=", command_burst_count],
                ],
                output="screen",
            )
        ],
    )

    return [
        camera,
        moveit,
        handeye,
        retime,
        yolo_obb,
        monitor,
        task,
        OpaqueFunction(function=_graspnet_inference_process),
        cli,
    ]


def generate_launch_description():
    """返回真实硬件 LLM 机器人的启动描述。"""
    launch_arguments = [
        _argument(name, default) for name, default in DEFAULTS.items()
    ]
    rviz_config_argument = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(
            get_package_share_directory("llm_arm_control"),
            "rviz",
            "llm_robot_control.rviz",
        ),
    )

    return LaunchDescription([
        *launch_arguments,
        rviz_config_argument,
        OpaqueFunction(function=_launch_setup),
    ])
