#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from manipulation_common.launch_utils.yaml_loader import load_yaml
from moveit_configs_utils import MoveItConfigsBuilder
from visual_servo_bringup.position_servo_config import (
    visual_servo_parameters,
    yolo_kalman_parameters,
)


_HANDEYE_LAUNCH_DIR = os.path.join(
    get_package_share_directory("hand_eye_calibration"), "launch"
)
if _HANDEYE_LAUNCH_DIR not in sys.path:
    sys.path.insert(0, _HANDEYE_LAUNCH_DIR)

from handeye_launch_utils import camera_launch, value  # noqa: E402


DEFAULTS = {
    "use_sim_time": "false",
    "camera_serial_no": "",
    "color_profile": "640x480x60",
    "depth_profile": "640x480x60",
    "pointcloud_enable": "false",
    "use_rviz": "true",
    "rviz_config": os.path.join(
        get_package_share_directory("visual_servo_bringup"),
        "rviz",
        "visual_position_servo.rviz",
    ),
    "debug": "false",
    "allow_trajectory_execution": "true",
    "publish_monitored_planning_scene": "true",
    "monitor_dynamics": "false",
    "capabilities": "",
    "disable_capabilities": "",
    "publish_frequency": "100.0",
}

DESCRIPTIONS = {
    "use_sim_time": "实机默认 false。",
    "camera_serial_no": "RealSense 相机序列号；留空时由驱动选择。",
    "color_profile": "彩色流 profile。",
    "depth_profile": "深度流 profile。",
    "pointcloud_enable": "是否启用驱动点云。",
    "use_rviz": "是否启动 MoveIt RViz。",
    "rviz_config": "RViz 配置文件绝对路径。",
    "debug": "MoveIt 调试模式。",
    "allow_trajectory_execution": "是否允许 MoveIt 执行轨迹。",
    "publish_monitored_planning_scene": "是否发布监控规划场景。",
    "monitor_dynamics": "是否监控机器人动力学。",
    "capabilities": "额外 MoveIt capabilities。",
    "disable_capabilities": "禁用的 MoveIt capabilities。",
    "publish_frequency": "MoveIt 状态发布频率。",
}


def _argument(name, default):
    kwargs = {"default_value": default, "description": DESCRIPTIONS[name]}
    return DeclareLaunchArgument(name, **kwargs)


def _hardware_moveit_config(kinematics_file: str):
    return (
        MoveItConfigsBuilder(
            "fairino_arm_moveit_descriptions",
            package_name="fairino_arm_moveit_config",
        )
        .robot_description_kinematics(file_path=f"config/{kinematics_file}")
        .planning_pipelines(default_planning_pipeline="fairino")
        .trajectory_execution(
            file_path="config/moveit_controllers_hardware.yaml",
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
    )


def _launch_setup(context):
    use_sim_time = LaunchConfiguration("use_sim_time")
    visual_servo_params = visual_servo_parameters()
    servo_ik = visual_servo_params["ik_moveit_servo"]
    yolo_params = yolo_kalman_parameters()
    cartesian_path_planner_params = load_yaml(
        "myrobot_planning_core", "config/cartesian_path_planner_params.yaml"
    )
    moveit_config = _hardware_moveit_config(f"kinematics_{servo_ik}.yaml")
    servo_yaml = load_yaml("fairino_arm_moveit_config", "config/servo_parameters.yaml")
    servo_yaml.update({
        "move_group_name": "robot_arm",
        "planning_frame": "base_link",
        "ee_frame_name": "tool0",
        "command_out_topic": "/robot_arm_controller/joint_trajectory",
        "command_out_type": "trajectory_msgs/JointTrajectory",
        "cartesian_command_in_topic": "/servo_node/delta_twist_cmds",
        "joint_command_in_topic": "/servo_node/delta_joint_cmds",
        "monitored_planning_scene_topic": "/move_group_fairino/monitored_planning_scene",
    })

    camera = camera_launch(
        "realsense",
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
            "hole_filling_filter.enable": "true",
        },
    )
    moveit_args = {
        name: value(context, name)
        for name in (
            "use_rviz", "rviz_config", "debug", "allow_trajectory_execution",
            "publish_monitored_planning_scene", "monitor_dynamics", "capabilities",
            "disable_capabilities", "publish_frequency",
        )
    }
    moveit_args["execution_ik"] = visual_servo_params["ik_plugin"]
    moveit_args["execution_pipeline"] = visual_servo_params["planning_pipeline_id"]
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("fairino_arm_moveit_config"),
                "launch",
                "moveit_hardware.launch.py",
            )
        ),
        launch_arguments=moveit_args.items(),
    )
    yolo_detector = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="visual_perception",
                executable="yolo_kalman_detector_obb.py",
                name="yolo_kalman_detector_obb",
                parameters=[{"use_sim_time": use_sim_time}, yolo_params],
            )
        ],
    )
    servo = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="moveit_servo",
                executable="servo_node_main",
                name="servo_node",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {"use_sim_time": use_sim_time, "moveit_servo": servo_yaml},
                ],
            )
        ],
    )
    handeye = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
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
    retime = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )
    visual_servo = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="visual_servo_bringup",
                executable="visual_servo_grasping",
                name="visual_servo_grasping_node",
                output="screen",
                parameters=[
                    cartesian_path_planner_params,
                    {"use_sim_time": use_sim_time, "allow_cross_client_fallback": False},
                    visual_servo_params,
                ],
            )
        ],
    )
    return [camera, moveit, servo, yolo_detector, handeye, retime, visual_servo]


def generate_launch_description():
    return LaunchDescription([
        *(_argument(name, default) for name, default in DEFAULTS.items()),
        OpaqueFunction(function=_launch_setup),
    ])
