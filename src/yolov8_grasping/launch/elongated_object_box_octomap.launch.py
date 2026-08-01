#!/usr/bin/env python3
"""Elongated-object-box octomap launch: detection, grasping, and octomap.

Orchestrates YOLO detection, semantic cloud filtering, elongated-object-box grasping,
and dynamic collision objects — all from yolo_perception and yolov8_grasping.

Key parameters (all default False to preserve existing run behaviour):
  enable_semantic_cloud_filter  — start semantic_octomap_cloud_filter
  enable_dynamic_collision_objects — start dynamic_collision_objects
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    this_package_path = get_package_share_directory("yolov8_grasping")

    # ===== 相机启动 =====
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ]),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "depth_module.profile": "640x480x15",
            "rgb_camera.profile": "640x480x15",
            "align_depth.enable": "true",
        }.items(),
    )

    # ===== MoveIt配置和启动 =====
    ar_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("fairino_arm_moveit_config"),
                "launch",
                "demo.launch.py",
            )
        ]),
        launch_arguments={
            "use_rviz": "true",
            "rviz_config": os.path.join(
                this_package_path, "rviz", "elongated_object_box_octomap.rviz",
            ),
        }.items(),
    )

    # ===== YOLO检测节点 =====
    yolo_detector_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="yolo_perception",
                executable="yolo_detector_obb.py",
                name="yolo_detector_obb",
                parameters=[{
                    "model_path": os.path.join(
                        get_package_share_directory("yolo_perception"),
                        "models", "yolo-obb3.pt",
                    ),
                    "device": "auto",
                    "conf": 0.55,
                    "imgsz": 640,
                    "enable_visualization": True,
                    "enable_ema_smoothing": True,
                    "ema_alpha": 0.35,
                    "sync_slop": 0.03,
                }],
            )
        ],
    )

    # ===== 手眼标定发布节点 =====
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{
            "calibration_name": "robot_calibration",
            "storage_directory": str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"),
        }],
        output="screen",
    )

    # ===== 时间戳轨迹节点启动 =====
    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    # ===== semantic_octomap_cloud_filter节点（默认关闭）=====
    semantic_octomap_cloud_filter_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="yolo_perception",
                executable="semantic_octomap_cloud_filter.py",
                name="semantic_octomap_cloud_filter_node",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("enable_semantic_cloud_filter")
                ),
            )
        ],
    )

    # ===== dynamic_collision_objects节点（默认关闭）=====
    dynamic_collision_objects_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="yolov8_grasping",
                executable="dynamic_collision_objects",
                name="dynamic_collision_objects_node",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("enable_dynamic_collision_objects")
                ),
            )
        ],
    )

    # ===== Elongated-object-box 抓取任务节点 =====
    visual_grasping_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="yolov8_grasping",
                executable="visual_grasping",
                name="visual_grasping",
                output="screen",
            )
        ],
    )

    enable_semantic_cloud_filter_arg = DeclareLaunchArgument(
        "enable_semantic_cloud_filter",
        default_value="false",
        description="Enable semantic octomap cloud filter node.",
    )
    enable_dynamic_collision_objects_arg = DeclareLaunchArgument(
        "enable_dynamic_collision_objects",
        default_value="false",
        description="Enable dynamic collision objects node.",
    )

    return LaunchDescription([
        enable_semantic_cloud_filter_arg,
        enable_dynamic_collision_objects_arg,
        realsense_launch,
        ar_moveit,
        yolo_detector_node,
        hand_eye_tf_publisher,
        retime_server_launch,
        semantic_octomap_cloud_filter_node,
        dynamic_collision_objects_node,
        visual_grasping_node,
    ])
