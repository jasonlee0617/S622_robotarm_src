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


def _graspnet_inference_process():
    install_setup = str(Path.home() / "fairino_robotarm/install/setup.bash")
    baseline_dir = str(Path.home() / "manipulator_grasp/graspnet-baseline")
    checkpoint_path = str(Path.home() / "manipulator_grasp/logs/log_rs/checkpoint-rs.tar")
    conda_setup = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
    cmd = (
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
        "-r __node:=graspnet_inference "
        "-p use_sim_time:=false "
        "-p rgb_topic:=/camera/camera/color/image_raw "
        "-p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw "
        "-p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info "
        "-p camera_frame:=camera_color_optical_frame "
        f"-p baseline_dir:={baseline_dir} "
        f"-p checkpoint_path:={checkpoint_path} "
        "-p num_point:=20000 "
        "-p top_k_publish:=5 "
        "-p min_valid_points:=2000 "
        "-p roi_norm:='[0.20, 0.20, 0.90, 0.85]' "
        "-p auto_once:=false "
        "-p auto_visualize:=false "
        "-p confirm_before_publish:=true "
        "-p confirm_visual_top_k:=50"
    )
    return ExecuteProcess(cmd=["bash", "-lc", cmd], output="screen")


def generate_launch_description():
    graspnet_share = get_package_share_directory("graspnet_grasping")
    graspnet_visual_grasping_config = os.path.join(
        graspnet_share,
        "config",
        "graspnet_visual_grasping.yaml",
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "depth_module.profile": "640x480x30",
            "rgb_camera.profile": "640x480x30",
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
                "use_sim_time": False,
            },
            LaunchConfiguration("graspnet_visual_grasping_config"),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "graspnet_visual_grasping_config",
                default_value=graspnet_visual_grasping_config,
                description="YAML file for the graspnet_visual_grasping executor node.",
            ),
            realsense_launch,
            _graspnet_inference_process(),
            graspnet_visual_grasping,
        ]
    )
