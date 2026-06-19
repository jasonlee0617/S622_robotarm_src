#!/usr/bin/env python3
import os
import yaml
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.moveit_stack import build_moveit_config  # noqa: E402
from launch_utils.robot_profiles import load_robot_profile  # noqa: E402
from manipulation_common.launch_utils.yaml_loader import load_ros_parameters_yaml, load_yaml  # noqa: E402


_SERVO_RUNTIME_DEFAULTS = load_ros_parameters_yaml(
    "visual_servo",
    "config/servo_runtime.yaml",
    "/**",
)


def _launch_default_from_mapping(mapping: dict, name: str, fallback: str) -> str:
    value = mapping.get(name, fallback)
    return str(value)


def generate_launch_description():
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_launch') + '/launch/gazebo_yolo.launch.py']),
        launch_arguments={
            "robot_profile": LaunchConfiguration("robot_profile"),
            "world": LaunchConfiguration("world"),
            "camera_info_remap": "/camera/camera/aligned_depth_to_color/camera_info",
            "camera_fps": LaunchConfiguration("camera_fps"),
            "camera_image_width": LaunchConfiguration("camera_image_width"),
            "camera_image_height": LaunchConfiguration("camera_image_height"),
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
                    "backend": LaunchConfiguration("backend"),
                    # "backend": "torch",
                    # "backend": "tensorrt",
                    "model_path": LaunchConfiguration("model_path"),
                    "engine_path": LaunchConfiguration("engine_path"),
                    "device": "cuda:0",
                    "imgsz": 640,
                    "conf": 0.2,
                }],
            )
        ]
    )

    backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="tensorrt",
        description="YOLO inference backend: torch or tensorrt"
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.join(get_package_share_directory("yolo_perception"), "models", "yolo-obb-gazebo.pt"),
        description="YOLO model path. Defaults to yolo_perception package share; relative values are resolved by the node.",
    )
    engine_path_arg = DeclareLaunchArgument(
        "engine_path",
        default_value=os.path.join(get_package_share_directory("yolo_perception"), "models", "yolo-obb-gazebo.engine"),
        description="TensorRT engine path. Defaults to yolo_perception package share; relative values are resolved by the node.",
    )
    robot_profile_arg = DeclareLaunchArgument(
        "robot_profile",
        default_value="s622_gripper",
        description="Robot profile passed to gazebo_yolo.launch.py.",
    )
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="arm_on_the_table",
        description="Gazebo world name passed to gazebo_yolo.launch.py.",
    )
    ik_plugin_arg = DeclareLaunchArgument(
        "ik_plugin",
        default_value="fairino",
        description="IK solver for grasp pipeline: fairino or kdl.",
    )
    enable_velocity_eval_arg = DeclareLaunchArgument(
        "enable_velocity_eval",
        default_value="true",
        description="Start cube velocity truth/evaluation telemetry nodes.",
    )
    camera_fps_arg = DeclareLaunchArgument(
        "camera_fps",
        default_value=_launch_default_from_mapping(_SERVO_RUNTIME_DEFAULTS, "camera_fps", "60"),
        description="Gazebo camera FPS for the visual servo pipeline.",
    )
    camera_image_width_arg = DeclareLaunchArgument(
        "camera_image_width",
        default_value=_launch_default_from_mapping(
            _SERVO_RUNTIME_DEFAULTS, "camera_image_width", "640"
        ),
        description="Gazebo camera image width for the visual servo pipeline.",
    )
    camera_image_height_arg = DeclareLaunchArgument(
        "camera_image_height",
        default_value=_launch_default_from_mapping(
            _SERVO_RUNTIME_DEFAULTS, "camera_image_height", "480"
        ),
        description="Gazebo camera image height for the visual servo pipeline.",
    )
 
    # ===== 时间戳轨迹节点启动（延迟启动）=====
    # 使用与 gazebo_yolo.launch.py 一致的 robot_profile 驱动配置，避免旧 wrapper xacro 依赖。
    profile = load_robot_profile("s622_gripper")
    moveit_config = build_moveit_config(profile, default_planning_pipeline="fairino", enable_camera_model=True)
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
        period=8.0,  # 8秒后启动，确保MoveIt完全启动
        actions=[
            Node(
                package='visual_servo',
                executable='servo_yolo_grasping',  
                name='servo_yolo_grasping_node',
                output='screen',
                parameters=[
                    {"use_sim_time": True,
                     "ik_plugin": LaunchConfiguration("ik_plugin")},
                    os.path.join(
                        get_package_share_directory("visual_servo"),
                        "config",
                        "moveit_client.yaml",
                    ),
                    os.path.join(
                        get_package_share_directory("visual_servo"),
                        "config",
                        "grasp_task.yaml",
                    ),
                    os.path.join(
                        get_package_share_directory("visual_servo"),
                        "config",
                        "servo_runtime.yaml",
                    ),
                    cartesian_path_planner_params,
                ],
            )
        ]
    )
    box_cmd_vel_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/box_model/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
        output='screen',
        parameters=[{"use_sim_time": True}],
    )
    enable_semantic_filter_arg = DeclareLaunchArgument(
        "enable_semantic_cloud_filter",
        default_value="false",
        description="Enable semantic octomap cloud filter from yolo_perception.",
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
                condition=IfCondition(
                    LaunchConfiguration("enable_semantic_cloud_filter"),
                ),
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
                }],
                condition=IfCondition(LaunchConfiguration("enable_velocity_eval")),
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
                    'model_name': 'box_model',
                    'cmd_topic': '/model/box_model/cmd_vel',
                    'cmd_internal_topic': '/cube_truth/cmd_vel_command_internal',
                }],
            )
        ]
    )
  
    return LaunchDescription([
        # 参数声明
        backend_arg,
        model_path_arg,
        engine_path_arg,
        robot_profile_arg,
        world_arg,
        ik_plugin_arg,
        enable_velocity_eval_arg,
        enable_semantic_filter_arg,
        camera_fps_arg,
        camera_image_width_arg,
        camera_image_height_arg,
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
    ])
