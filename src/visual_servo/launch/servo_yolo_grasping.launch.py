#!/usr/bin/env python3
import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from manipulation_common.launch_utils.yaml_loader import load_yaml


def absolute_moveit_controller_config():
    return {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": [
                "/robot_arm_controller",
                "/hand_controller",
            ],
            "/robot_arm_controller": {
                "type": "FollowJointTrajectory",
                "joints": ["j1", "j2", "j3", "j4", "j5", "j6"],
                "action_ns": "follow_joint_trajectory",
                "default": True,
            },
            "/hand_controller": {
                "type": "FollowJointTrajectory",
                "joints": ["finger1_joint", "finger2_joint"],
                "action_ns": "follow_joint_trajectory",
                "default": True,
            },
        },
    }


def _load_ros_params(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["/**"]["ros__parameters"]


def _visual_servo_param_files(config_dir):
    common_yaml = os.path.join(config_dir, "visual_servo_params.yaml")
    controller_type = str(
        _load_ros_params(common_yaml).get("servo_controller_type", "LADRC")
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
    return [common_yaml, *[os.path.join(config_dir, name) for name in selected]]


def generate_launch_description():

    this_package_path = get_package_share_directory('visual_servo')
    config_dir = os.path.join(this_package_path, "config")
    visual_servo_param_yamls = _visual_servo_param_files(config_dir)
    grasp_task_yaml = os.path.join(this_package_path, "config", "grasp_task.yaml")
    moveit_client_yaml = os.path.join(this_package_path, "config", "moveit_client.yaml")
    perception_yaml = os.path.join(this_package_path, "config", "perception_params.yaml")
    cartesian_path_planner_params = load_yaml(
        "fairino_planning_core", "config/cartesian_path_planner_params.yaml"
    )


    
    # ===== 相机启动 =====
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'depth_module.profile': '640x480x15',
            'rgb_camera.profile': '640x480x15',
            # 'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            # 'enable_sync': 'true',
            # 'temporal_filter.enable': 'true',
            # 'spatial_filter.enable': 'true',
            # 'hole_filling_filter.enable': 'true',
        }.items()
    )
    # ===== OAK-D-Lite相机启动 =====
    oak_camera = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory("depthai_ros_driver"),
            "launch", "camera.launch.py",
        )
    ),
    launch_arguments={
        'camera_model': 'OAK-D-LITE',
        'name': 'oak',
        'enable_depth': 'true',
        'enable_color': 'true',
        'rs_compat': 'true',  # 启用RealSense兼容模式
        'depth_module.depth_profile': '640,480,30',
        'rgb_camera.color_profile': '640,480,30',
    }.items()
    )


    # ===== MoveIt配置和启动 =====
    moveit_config_pkg = "s622_moveit_config"
    ar_moveit_launch = PythonLaunchDescriptionSource([
        os.path.join(
            # get_package_share_directory("fairino3_v6_moveit2_config"), 
            get_package_share_directory("s622_moveit_config"), 
            "launch",
            "demo.launch.py",
        )
    ])

    rviz_config_file = os.path.join(
        get_package_share_directory("visual_servo"),
        "rviz",
        "servo_yolo_grasping.rviz",
    )

    
    ar_moveit = IncludeLaunchDescription(
        ar_moveit_launch, 
        # launch_arguments=ar_moveit_args
        launch_arguments={
      "use_rviz": "true",             # 保证会 include MoveIt 的 rviz launch
      "rviz_config": rviz_config_file, # 传给 moveit_rviz.launch.py 的参数名
       }.items(),
    )




    # ===== 延迟启动YOLO检测节点 =====
        # ===== YOLO检测节点（延迟3秒启动）=====
    yolo_detector_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='yolo_perception',
                executable='yolo_kalman_detector_obb.py',
                name='yolo_Kalman_detector_obb_node',
                parameters=[perception_yaml],
                # output='screen'
            )
        ]
    )

    moveit_config = (
        MoveItConfigsBuilder("s622", package_name=moveit_config_pkg).to_moveit_configs()
    )
    kinematics_fairino = load_yaml(moveit_config_pkg, "config/kinematics_fairino.yaml")
    kinematics_kdl = load_yaml(moveit_config_pkg, "config/kinematics_kdl.yaml")
    fairino_planning_cfg = load_yaml(moveit_config_pkg, "config/fairino_planning.yaml")
    absolute_controllers_cfg = absolute_moveit_controller_config()

    servo_yaml = load_yaml("s622_moveit_config", "config/servo_parameters.yaml")
    # 覆盖为你的机器人
    servo_yaml["move_group_name"] = "robot_arm"
    servo_yaml["planning_frame"] = "base_link"
    servo_yaml["ee_frame_name"] = "grasp_frame"
    servo_yaml["command_out_topic"] = "/robot_arm_controller/joint_trajectory"
    servo_yaml["command_out_type"] = "trajectory_msgs/JointTrajectory"
    servo_yaml["cartesian_command_in_topic"] = "/servo_node/delta_twist_cmds"
    servo_yaml["joint_command_in_topic"] = "/servo_node/delta_joint_cmds"

    servo_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package="moveit_servo",
                executable="servo_node_main",
                name="servo_node",
                output="screen",
                parameters=[moveit_config.to_dict(), {"moveit_servo": servo_yaml}],
            )
        ],
    )
    dual_move_group_nodes = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                namespace="move_group_fairino",
                name="move_group",
                output="screen",
                remappings=[
                    ("joint_states", "/joint_states"),
                    ("robot_arm_controller/follow_joint_trajectory", "/robot_arm_controller/follow_joint_trajectory"),
                    ("hand_controller/follow_joint_trajectory", "/hand_controller/follow_joint_trajectory"),
                ],
                parameters=[
                    moveit_config.to_dict(),
                    kinematics_fairino,
                    absolute_controllers_cfg,
                    fairino_planning_cfg,
                ],
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                namespace="move_group_kdl",
                name="move_group",
                output="screen",
                remappings=[
                    ("joint_states", "/joint_states"),
                    ("robot_arm_controller/follow_joint_trajectory", "/robot_arm_controller/follow_joint_trajectory"),
                    ("hand_controller/follow_joint_trajectory", "/hand_controller/follow_joint_trajectory"),
                ],
                parameters=[
                    moveit_config.to_dict(),
                    kinematics_kdl,
                    absolute_controllers_cfg,
                    fairino_planning_cfg,
                ],
            ),
        ],
    )
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("gazebo_launch"), 
                "launch",
                "gazebo.launch.py",
            )
        ]),
        launch_arguments={
            # ✅ 覆盖 RViz 配置文件路径
            'rviz_config': os.path.join(
                this_package_path, 
                'rviz', 
                'servo_gazebo_grasping.rviz'
            )
        }.items()
    )
    # ===== 手眼标定发布节点 =====
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{"calibration_name": "robot_calibration",}],  # 直接使用固定值
        output='screen'
    )

    # ===== 时间戳轨迹节点启动（延迟启动）=====
    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )

    # ===== Pen-Box抓取任务节点（延迟启动）=====
    servo_yolo_grasping_node = TimerAction(
        period=5.0,  # 8秒后启动，确保MoveIt完全启动
        actions=[
            Node(
                package='visual_servo',
                executable='servo_yolo_grasping',  
                name='servo_yolo_grasping_node',
                output='screen',
                parameters=[
                    *visual_servo_param_yamls,
                    moveit_client_yaml,
                    grasp_task_yaml,
                    cartesian_path_planner_params,
                ],
            )
        ]
    )


    return LaunchDescription([
        # 参数声明
        # 启动相机
        realsense_launch,
        # oak_camera,
        
        # 启动MoveIt（包含机器人模型、规划器等）
        ar_moveit,
        dual_move_group_nodes,

        # gazebo_node,
        servo_node,
        # 延迟启动YOLO检测节点
        yolo_detector_node,
        # gazebo_node,
        # 启动手眼标定发布器
        hand_eye_tf_publisher,
        retime_server_launch,
        # 延迟启动抓取任务节点
        servo_yolo_grasping_node,
    ])
