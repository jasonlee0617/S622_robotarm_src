#!/usr/bin/env python3
import os
import yaml
import sys
import math
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
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
from manipulation_common.launch_utils.yaml_loader import load_yaml  # noqa: E402
from visual_servo_bringup.position_servo_config import (  # noqa: E402
    gazebo_camera_defaults,
    visual_servo_parameters,
    yolo_kalman_parameters,
)


_GAZEBO_CAMERA_DEFAULTS = gazebo_camera_defaults()
_VISUAL_SERVO_PARAMS = visual_servo_parameters()

# 公开场景参数集中管理。固定 YAML、模型和 RViz 文件仍由包共享目录定位。
_LAUNCH_ARGUMENT_SPECS = (
    ("use_sim_time", "true", "是否使用 Gazebo 的 /clock。", None),
    ("camera_profile", str(_GAZEBO_CAMERA_DEFAULTS["camera_profile"]), "命名 D435 配置。", None),
    ("camera_profile_file", "", "外部 D435 配置文件；使用时清空 camera_profile。", None),
    ("camera_noise_mode", str(_GAZEBO_CAMERA_DEFAULTS["camera_noise_mode"]), "相机噪声模型。", ("off", "d435_empirical")),
    ("camera_depth_far_m", "10.0", "D435 深度远裁剪距离（米）。", None),
    ("camera_fps", str(_GAZEBO_CAMERA_DEFAULTS["camera_fps"]), "仿真相机帧率。", None),
    ("camera_image_width", str(_GAZEBO_CAMERA_DEFAULTS["camera_image_width"]), "仿真彩色图像宽度。", None),
    ("camera_image_height", str(_GAZEBO_CAMERA_DEFAULTS["camera_image_height"]), "仿真彩色图像高度。", None),
)


def _declare_launch_arguments():
    declarations = []
    for name, default_value, description, choices in _LAUNCH_ARGUMENT_SPECS:
        kwargs = {"default_value": default_value, "description": description}
        if choices:
            kwargs["choices"] = list(choices)
        declarations.append(DeclareLaunchArgument(name, **kwargs))
    return declarations


def _gazebo_yolo_parameters() -> dict:
    params = yolo_kalman_parameters()
    models_dir = os.path.join(get_package_share_directory("visual_perception"), "models")
    for name in ("model_path", "engine_path"):
        params[name] = os.path.join(models_dir, params[name])
    return params


def _launch_setup(context, *args, **kwargs):
    camera_profile = LaunchConfiguration("camera_profile").perform(context)
    camera_profile_file = LaunchConfiguration("camera_profile_file").perform(context)
    camera_noise_mode = LaunchConfiguration("camera_noise_mode").perform(context)
    camera_depth_far_m = LaunchConfiguration("camera_depth_far_m").perform(context)
    camera_mappings = {
        "camera_fps": LaunchConfiguration("camera_fps").perform(context),
        "camera_image_width": LaunchConfiguration("camera_image_width").perform(context),
        "camera_image_height": LaunchConfiguration("camera_image_height").perform(context),
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
    camera_profile_summary = LogInfo(
        msg=(
            "D435 camera profile: "
            f"source={camera_profile or camera_profile_file}, "
            f"color={camera_mappings['camera_image_width']}x"
            f"{camera_mappings['camera_image_height']}@"
            f"{camera_mappings['camera_fps']}Hz, "
            f"fx={float(camera_mappings['camera_fx']):.3f}, "
            f"fy={float(camera_mappings['camera_fy']):.3f}, "
            f"cx={float(camera_mappings['camera_cx']):.3f}, "
            f"cy={float(camera_mappings['camera_cy']):.3f}, "
            f"h_fov={math.degrees(float(camera_mappings['camera_h_fov'])):.3f}deg, "
            f"v_fov={math.degrees(float(camera_mappings['camera_v_fov'])):.3f}deg"
        )
    )
    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('myrobot_simulation') + '/launch/gazebo.launch.py']),
        launch_arguments={
            "robot_profile": "fairino_arm_gripper_onbase",
            "world": "robotarm_world",
            "rviz_config": os.path.join(
                get_package_share_directory("visual_servo_bringup"),
                "rviz",
                "visual_position_servo.rviz",
            ),
            "publish_frequency": "30.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "true",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
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
                package='visual_perception',
                executable='yolo_kalman_detector_obb.py',
                name='yolo_kalman_detector_obb',
                output='screen',
                parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}, _gazebo_yolo_parameters()],
            )
        ]
    )


    # ===== 时间戳轨迹节点启动（延迟启动）=====
    # 使用与基础 Gazebo 入口一致的 robot_profile 驱动配置，避免旧 wrapper xacro 依赖。
    profile = load_robot_profile("fairino_arm_gripper_onbase")
    moveit_config = build_moveit_config(
        profile,
        enable_camera_model=True,
        extra_mappings=camera_mappings,
    )
    cartesian_path_planner_params = load_yaml(
        "myrobot_planning_core", "config/cartesian_path_planner_params.yaml"
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


    visual_servo_grasping_node = TimerAction(
        period=5.0,  # 5秒后启动，确保MoveIt完全启动
        actions=[
            Node(
                package='visual_servo_bringup',
                executable='visual_servo_grasping',
                name='visual_servo_grasping_node',
                output='screen',
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                    _VISUAL_SERVO_PARAMS,
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
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    semantic_octomap_cloud_filter_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='visual_perception',
                executable='semantic_octomap_cloud_filter.py',
                name='semantic_octomap_cloud_filter_node',
                output='screen',
                parameters=[{
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "input_cloud_topic": "/camera/camera/depth/color/points",
                    "output_cloud_topic": "/octomap_cloud_filtered",
                }],
            )
        ],
    )
    cube_controller_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='myrobot_simulation',
                executable='cube_controller_node.py',
                name='cube_velocity_keyboard_node',
                output='screen',
                parameters=[{
                    'use_sim_time': LaunchConfiguration("use_sim_time"),
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
        camera_profile_summary,
        myrobot_simulation,
        box_cmd_vel_bridge_node,
        # 启动YOLO检测节点
        yolo_obb,
        # semantic_octomap_cloud_filter_node,
        retime_server_launch,
        # 延迟启动抓取任务节点
        cube_controller_node,
        visual_servo_grasping_node,
    ]


def generate_launch_description():
    return LaunchDescription([*_declare_launch_arguments(), OpaqueFunction(function=_launch_setup)])
