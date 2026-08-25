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
    gazebo_aruco_motion_parameters,
    visual_servo_parameters,
    yolo_kalman_parameters,
)


_GAZEBO_CAMERA_DEFAULTS = gazebo_camera_defaults()
_GAZEBO_ARUCO_MOTION = gazebo_aruco_motion_parameters()
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
    (
        "open_gripper_after_home",
        str(bool(_VISUAL_SERVO_PARAMS.get("open_gripper_after_home", False))).lower(),
        "回 Home 后是否自动张开夹爪；默认值来自 visual_position_servo.yaml。",
        None,
    ),
    ("robot_profile", "fairino3_v6", "Gazebo 机器人配置。", ("fairino_arm_gripper_onbase", "fairino_arm_gripper_inhand", "fairino3_v6")),
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


def _bool_launch_value(context, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _launch_setup(context, *args, **kwargs):
    robot_profile = LaunchConfiguration("robot_profile").perform(context)
    camera_profile = LaunchConfiguration("camera_profile").perform(context)
    camera_profile_file = LaunchConfiguration("camera_profile_file").perform(context)
    camera_noise_mode = LaunchConfiguration("camera_noise_mode").perform(context)
    camera_depth_far_m = LaunchConfiguration("camera_depth_far_m").perform(context)
    open_gripper_after_home = _bool_launch_value(context, "open_gripper_after_home")
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
            "robot_profile": robot_profile,
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

    profile = load_robot_profile(robot_profile)
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
    visual_position_servo_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='visual_servo_bringup',
                executable='visual_servo_grasping',
                name='visual_position_servo_node',
                output='screen',
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                    _VISUAL_SERVO_PARAMS,
                    cartesian_path_planner_params,
                    {"open_gripper_after_home": open_gripper_after_home},
                ],
            )
        ]
    )
    source_actions = []
    if _VISUAL_SERVO_PARAMS["perception_source"] == "yolo_kalman":
        source_actions.extend([
            Node(
                package='ros_gz_bridge', executable='parameter_bridge',
                arguments=['/model/cube_model/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
                output='screen', parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            ),
            TimerAction(period=2.0, actions=[Node(
                package='myrobot_simulation', executable='target_motion_controller_node.py',
                name='cube_target_motion_controller', output='screen', parameters=[{
                    'use_sim_time': LaunchConfiguration("use_sim_time"), 'auto_start': False,
                    'trajectory_type': 'circle', 'model_name': 'cube_model',
                    'cmd_topic': '/model/cube_model/cmd_vel',
                    'cmd_internal_topic': '/cube_truth/cmd_vel_command_internal',
                    'auto_start_topic': '/cube_auto_start',
                }],
            )]),
            TimerAction(period=3.0, actions=[Node(
                package='visual_perception', executable='yolo_kalman_detector_obb.py',
                name='yolo_kalman_detector_obb', output='screen',
                parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}, _gazebo_yolo_parameters()],
            )]),
        ])
    else:
        aruco_parameters = {
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "marker_size": _VISUAL_SERVO_PARAMS["aruco_marker_size_m"],
            "aruco_dictionary_id": _VISUAL_SERVO_PARAMS["aruco_dictionary"],
            "image_topic": "/camera/camera/color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "visualization_image_topic": _VISUAL_SERVO_PARAMS["aruco_visualization_image_topic"],
            "visualization_marker_id": _VISUAL_SERVO_PARAMS["aruco_visualization_marker_id"],
        }
        source_actions.extend([
            Node(
                package='ros_gz_bridge', executable='parameter_bridge',
                arguments=[f"{_GAZEBO_ARUCO_MOTION['cmd_topic']}@geometry_msgs/msg/Twist@gz.msgs.Twist"],
                output='screen', parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            ),
            TimerAction(period=4.0, actions=[Node(
                package='ros2_aruco', executable='aruco_node', name='aruco_node',
                output='screen', parameters=[aruco_parameters],
            )]),
            TimerAction(period=4.5, actions=[Node(
                package='hand_eye_calibration', executable='aruco_marker_pose_publisher.py',
                name='aruco_marker_pose_publisher', output='screen', parameters=[{
                    'use_sim_time': LaunchConfiguration("use_sim_time"),
                    'marker_id': _VISUAL_SERVO_PARAMS['aruco_marker_id'],
                    'output_topic': _VISUAL_SERVO_PARAMS['aruco_marker_pose_topic'],
                }],
            )]),
            TimerAction(period=4.0, actions=[Node(
                package='myrobot_simulation', executable='target_motion_controller_node.py',
                name='aruco_marker_target_motion_controller', output='screen', parameters=[{
                    'use_sim_time': LaunchConfiguration("use_sim_time"), 'auto_start': False,
                    **_GAZEBO_ARUCO_MOTION,
                }],
            )]),
        ])
    return [
        camera_profile_summary,
        myrobot_simulation,
        retime_server_launch,
        *source_actions,
        visual_position_servo_node,
    ]


def generate_launch_description():
    return LaunchDescription([*_declare_launch_arguments(), OpaqueFunction(function=_launch_setup)])
