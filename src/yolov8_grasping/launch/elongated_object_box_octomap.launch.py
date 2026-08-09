#!/usr/bin/env python3
"""细长目标抓取与 OctoMap 场景入口."""

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


# 节点标量统一通过 launch 参数注入；模型、RViz 与保存路径均是固定资源。
_LAUNCH_DEFAULTS = {
    "use_sim_time": "false",
    "enable_semantic_cloud_filter": "false",
    "enable_dynamic_collision_objects": "false",
    "device": "auto",
    "conf": "0.55",
    "imgsz": "640",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in _LAUNCH_DEFAULTS
}


def _declare_launch_arguments():
    """声明可通过 CLI 覆盖的节点运行参数."""
    descriptions = {
        "use_sim_time": "是否使用仿真时间。",
        "enable_semantic_cloud_filter": "是否启动语义点云过滤节点。",
        "enable_dynamic_collision_objects": "是否启动动态碰撞体节点。",
        "device": "YOLO 推理设备。",
        "conf": "YOLO 置信度阈值。",
        "imgsz": "YOLO 推理图像尺寸。",
    }
    return [
        DeclareLaunchArgument(name, default_value=value, description=descriptions[name])
        for name, value in _LAUNCH_DEFAULTS.items()
    ]


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
                    "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                    "device": _LAUNCH_CONFIGURATIONS["device"],
                    "conf": _LAUNCH_CONFIGURATIONS["conf"],
                    "imgsz": _LAUNCH_CONFIGURATIONS["imgsz"],
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
            "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
            "calibration_name": "robot_calibration",
            "storage_directory": str(
                Path.home()
                / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/real"
            ),
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
                parameters=[{"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]}],
                condition=IfCondition(
                    _LAUNCH_CONFIGURATIONS["enable_semantic_cloud_filter"]
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
                parameters=[{"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]}],
                condition=IfCondition(
                    _LAUNCH_CONFIGURATIONS["enable_dynamic_collision_objects"]
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
                parameters=[{"use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"]}],
            )
        ],
    )

    return LaunchDescription([
        *_declare_launch_arguments(),
        realsense_launch,
        ar_moveit,
        yolo_detector_node,
        hand_eye_tf_publisher,
        retime_server_launch,
        semantic_octomap_cloud_filter_node,
        dynamic_collision_objects_node,
        visual_grasping_node,
    ])
