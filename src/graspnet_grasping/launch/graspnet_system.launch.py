#!/usr/bin/env python3
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# 真实相机流参数是启动拓扑参数；GraspNet YAML 是包内固定资源。
_LAUNCH_DEFAULTS = {
    "use_sim_time": "false",
    "depth_profile": "640x480x30",
    "color_profile": "640x480x30",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_DEFAULTS
}


def _graspnet_inference_process(use_sim_time):
    install_setup = str(Path.home() / "fairino_robotarm/install/setup.bash")
    config_path = os.path.join(
        get_package_share_directory("graspnet_grasping"),
        "config",
        "graspnet_visual_grasping.yaml",
    )
    baseline_dir = str(Path.home() / "manipulator_grasp/graspnet-baseline")
    checkpoint_path = str(Path.home() / "manipulator_grasp/logs/log_rs/checkpoint-rs.tar")
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    command_prefix = (
        "set -e; "
        f"source {conda_setup}; "
        "conda activate graspnet; "
        "source /opt/ros/humble/setup.bash; "
        f"source {install_setup}; "
        "export PYTHONUNBUFFERED=1; "
        "export MPLCONFIGDIR=/tmp/graspnet_mpl_config; "
        "export XDG_CACHE_HOME=/tmp/graspnet_xdg_cache; "
        "mkdir -p $MPLCONFIGDIR $XDG_CACHE_HOME; "
        "exec python -m graspnet_grasping.graspnet_inference_node "
        "--ros-args "
        f"--params-file {config_path} "
        "-r __node:=graspnet_inference "
        "-p use_sim_time:="
    )
    command_suffix = (
        " "
        "-p rgb_topic:=/camera/camera/color/image_raw "
        "-p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw "
        "-p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info "
        f"-p baseline_dir:={baseline_dir} "
        f"-p checkpoint_path:={checkpoint_path} "
        "-p num_point:=20000 "
        "-p top_k_publish:=5 "
        "-p min_valid_points:=2000 "
        "-p roi_norm:='[0.20, 0.20, 0.90, 0.85]' "
        "-p confirm_before_publish:=true "
        "-p confirm_visual_top_k:=50"
    )
    return ExecuteProcess(
        cmd=["bash", "-lc", [command_prefix, use_sim_time, command_suffix]],
        output="screen",
    )


def generate_launch_description():
    graspnet_share = get_package_share_directory("graspnet_grasping")

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "depth_module.profile": _LAUNCH_CONFIGURATIONS["depth_profile"],
            "rgb_camera.profile": _LAUNCH_CONFIGURATIONS["color_profile"],
            "pointcloud.enable": "true",
            "align_depth.enable": "true",
            "enable_sync": "true",
            "temporal_filter.enable": "true",
            "spatial_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
    )

    graspnet_visual_grasping = Node(
        package="graspnet_grasping",
        executable="graspnet_visual_grasping",
        name="graspnet_visual_grasping",
        output="screen",
        parameters=[
            {
                "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            },
            os.path.join(graspnet_share, "config", "graspnet_visual_grasping.yaml"),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value=_LAUNCH_DEFAULTS["use_sim_time"],
                description="是否使用仿真时间。",
            ),
            DeclareLaunchArgument(
                "depth_profile",
                default_value=_LAUNCH_DEFAULTS["depth_profile"],
                description="D435 深度流配置。",
            ),
            DeclareLaunchArgument(
                "color_profile",
                default_value=_LAUNCH_DEFAULTS["color_profile"],
                description="D435 彩色流配置。",
            ),
            realsense_launch,
            _graspnet_inference_process(_LAUNCH_CONFIGURATIONS["use_sim_time"]),
            graspnet_visual_grasping,
        ]
    )
