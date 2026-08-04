#!/usr/bin/env python3
import os
import yaml
import sys
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.moveit_stack import build_moveit_config  # noqa: E402
from launch_utils.d435_profile import d435_mappings  # noqa: E402
from launch_utils.robot_profiles import load_robot_profile  # noqa: E402
from manipulation_common.launch_utils.yaml_loader import load_ros_parameters_yaml, load_yaml  # noqa: E402


_SERVO_RUNTIME_DEFAULTS = load_ros_parameters_yaml(
    "visual_servo",
    "config/visual_servo_params.yaml",
    "/**",
)


def _visual_servo_config_path(name: str) -> str:
    return os.path.join(get_package_share_directory("visual_servo"), "config", name)


def _visual_servo_param_files() -> list[str]:
    controller_type = str(
        _SERVO_RUNTIME_DEFAULTS.get("servo_controller_type", "NLADRC")
    ).strip().upper()
    file_map = {
        "PID": ["visual_servo_pid_params.yaml"],
        "PD": ["visual_servo_pid_params.yaml"],
        "PI_FF": ["visual_servo_pid_params.yaml"],
        "ADAPTIVE_PID": [
            "visual_servo_pid_params.yaml",
            "visual_servo_adaptive_pid_params.yaml",
        ],
        "LADRC": ["visual_servo_ladrc_params.yaml"],
        "NLADRC": ["visual_servo_nladrc_params.yaml"],
        "MPC": ["visual_servo_mpc_params.yaml"],
    }
    selected = file_map.get(controller_type)
    if selected is None:
        raise RuntimeError(f"Unsupported servo_controller_type: {controller_type}")
    return [
        _visual_servo_config_path("visual_servo_params.yaml"),
        *[_visual_servo_config_path(name) for name in selected],
    ]


def _launch_default_from_mapping(mapping: dict, name: str, fallback: str) -> str:
    value = mapping.get(name, fallback)
    return str(value)


def _launch_setup(context, *args, **kwargs):
    camera_profile = LaunchConfiguration("camera_profile").perform(context)
    camera_profile_file = LaunchConfiguration("camera_profile_file").perform(context)
    camera_noise_mode = LaunchConfiguration("camera_noise_mode").perform(context)
    camera_depth_far_m = LaunchConfiguration("camera_depth_far_m").perform(context)
    camera_mappings = {
        "camera_fps": _launch_default_from_mapping(
            _SERVO_RUNTIME_DEFAULTS, "camera_fps", "60"
        ),
        "camera_image_width": _launch_default_from_mapping(
            _SERVO_RUNTIME_DEFAULTS, "camera_image_width", "640"
        ),
        "camera_image_height": _launch_default_from_mapping(
            _SERVO_RUNTIME_DEFAULTS, "camera_image_height", "480"
        ),
    }
    camera_mappings.update(
        d435_mappings(
            camera_profile,
            camera_profile_file,
            camera_noise_mode,
            camera_fps=camera_mappings["camera_fps"],
            camera_depth_far_m=camera_depth_far_m,
        )
    )
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_launch') + '/launch/gazebo.launch.py']),
        launch_arguments={
            "robot_profile": "fairino_arm_gripper",
            "world": "arm_on_the_table",
            "rviz_config": os.path.join(get_package_share_directory('gazebo_launch'), "rviz", "visual_servo_gazebo.rviz"),
            "publish_frequency": "30.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "true",
            "camera_info_remap": "/camera/camera/aligned_depth_to_color/camera_info",
            "camera_fps": camera_mappings["camera_fps"],
            "camera_image_width": camera_mappings["camera_image_width"],
            "camera_image_height": camera_mappings["camera_image_height"],
            "camera_profile": camera_profile,
            "camera_profile_file": camera_profile_file,
            "camera_noise_mode": camera_noise_mode,
            "camera_depth_far_m": camera_mappings["camera_depth_far_m"],
            "spawn_z": "1.02",
            "controller_spawn_delay": "5.0",
        }.items(),
    )

    # ===== 延迟启动YOLO检测节点 =====
        # ===== YOLO检测节点（延迟3秒启动）=====
    yolo_obb = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='yolo_perception',
                executable='yolo_kalman_detector_obb.py',
                name='yolo_Kalman_detector_obb_node',
                output='screen',
                parameters=[{
                    "use_sim_time": True,
                    "backend": "tensorrt",
                    # "backend": "torch",
                    # "backend": "tensorrt",
                    "model_path": os.path.join(get_package_share_directory("yolo_perception"), "models", "yolo-obb-gazebo.engine"),
                    "engine_path": os.path.join(get_package_share_directory("yolo_perception"), "models", "yolo-obb-gazebo.engine"),
                    "device": "cuda:0",
                    "imgsz": 640,
                    "conf": 0.2,
                }],
            )
        ]
    )


    # ===== 时间戳轨迹节点启动（延迟启动）=====
    # 使用与基础 Gazebo 入口一致的 robot_profile 驱动配置，避免旧 wrapper xacro 依赖。
    profile = load_robot_profile("fairino_arm_gripper")
    moveit_config = build_moveit_config(
        profile,
        enable_camera_model=True,
        extra_mappings=camera_mappings,
    )
    cartesian_path_planner_params = load_yaml(
        "fairino_planning_core", "config/cartesian_path_planner_params.yaml"
    )

    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        ),
        launch_arguments={
            "robot_description": moveit_config.robot_description["robot_description"],
            "robot_description_semantic": moveit_config.robot_description_semantic[
                "robot_description_semantic"
            ],
            "robot_description_kinematics": yaml.safe_dump(
                moveit_config.robot_description_kinematics[
                    "robot_description_kinematics"
                ]
            ),
        }.items(),
    )


    servo_gazebo_grasping_node = TimerAction(
        period=5.0,  # 5秒后启动，确保MoveIt完全启动
        actions=[
            Node(
                package='visual_servo',
                executable='servo_yolo_grasping',  
                name='servo_yolo_grasping_node',
                output='screen',
                parameters=[
                    {"use_sim_time": True},
                    *_visual_servo_param_files(),
                    _visual_servo_config_path("moveit_client.yaml"),
                    _visual_servo_config_path("grasp_task.yaml"),
                    cartesian_path_planner_params,
                ],
            )
        ]
    )
    box_cmd_vel_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/cube_model/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
        output='screen',
        parameters=[{"use_sim_time": True}],
    )

    semantic_octomap_cloud_filter_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='yolo_perception',
                executable='semantic_octomap_cloud_filter.py',
                name='semantic_octomap_cloud_filter_node',
                output='screen',
                parameters=[{
                    "use_sim_time": True,
                    "input_cloud_topic": "/camera/camera/depth/color/points",
                    "output_cloud_topic": "/octomap_cloud_filtered",
                }],
            )
        ],
    )
    vision_velocity_evaluator_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='yolo_perception',
                executable='vision_velocity_evaluator.py',
                name='vision_velocity_evaluator',
                output='screen',
                parameters=[{
                    "use_sim_time": True,
                    "trajectory_type": "circle",
                    "cmd_internal_topic": "/cube_truth/cmd_vel_command_internal",
                    "truth_topic": "/cube_truth/cmd_vel",
                    "enable_velocity_eval": "true",
                }],
            )
        ],
    )
    cube_controller_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='gazebo_launch',
                executable='cube_controller_node.py',
                name='cube_velocity_keyboard_node',
                output='screen',
                parameters=[{
                    'use_sim_time': True,
                    'auto_start': False,
                    'trajectory_type': 'circle',
                    'model_name': 'cube_model',
                    'cmd_topic': '/model/cube_model/cmd_vel',
                    'cmd_internal_topic': '/cube_truth/cmd_vel_command_internal',
                }],
            )
        ]
    )
  
    return [
        # 参数声明
        gazebo_launch,
        box_cmd_vel_bridge_node,
        # gazebo_node,
        # 启动YOLO检测节点
        yolo_obb,
        # semantic_octomap_cloud_filter_node,
        vision_velocity_evaluator_node,
        retime_server_launch,
        # 延迟启动抓取任务节点
        cube_controller_node,
        servo_gazebo_grasping_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_profile",
            default_value=_launch_default_from_mapping(
                _SERVO_RUNTIME_DEFAULTS,
                "camera_profile",
                "d435_color_640x480x30_depth_640x480x30",
            ),
            description="Named D435 profile for the visual servo camera simulation.",
        ),
        DeclareLaunchArgument(
            "camera_profile_file",
            default_value="",
            description="External D435 profile YAML; set camera_profile:='' when using it.",
        ),
        DeclareLaunchArgument(
            "camera_noise_mode",
            default_value=_launch_default_from_mapping(
                _SERVO_RUNTIME_DEFAULTS, "camera_noise_mode", "off"
            ),
            choices=["off", "d435_empirical"],
        ),
        DeclareLaunchArgument(
            "camera_depth_far_m",
            default_value="10.0",
            description="D435 depth far clip in metres; valid up to 10.0.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
