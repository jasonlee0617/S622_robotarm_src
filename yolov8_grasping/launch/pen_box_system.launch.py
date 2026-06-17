#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
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
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            # 'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
            'temporal_filter.enable': 'true',
            'spatial_filter.enable': 'true',
            'hole_filling_filter.enable': 'true',
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
    ar_moveit_launch = PythonLaunchDescriptionSource([
        os.path.join(
            # get_package_share_directory("fairino3_v6_moveit2_config"), 
            get_package_share_directory("s622_moveit_config"), 
            "launch",
            "demo.launch.py",
        )
    ])

    rviz_config_file = os.path.join(
        get_package_share_directory("yolov8_grasping"),
        "rviz",
        "yolo_grasping.rviz",
    )
    pen_box_moveit_config = os.path.join(
        get_package_share_directory("yolov8_grasping"),
        "config",
        "pen_box_moveit.yaml",
    )
    pen_box_task_config = os.path.join(
        get_package_share_directory("yolov8_grasping"),
        "config",
        "pen_box_task.yaml",
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
                executable='yolo_detector_obb.py',
                name='yolo_detector_obb',
                parameters=[{
                    # 'model_path': os.path.join(get_package_share_directory('yolo_perception'), 'models', 'yolov8n-obb.pt'),
                    # 'model_path': os.path.join(get_package_share_directory('yolo_perception'), 'models', 'yolov8n-obb.pt'),
                    'model_path': os.path.join(get_package_share_directory('yolo_perception'), 'models', 'yolo-obb3.pt'),
                    # 'model_path': os.path.join(get_package_share_directory('yolo_perception'), 'models', 'yolov8n.pt'),
                    # 'model_path': os.path.join(get_package_share_directory('yolo_perception'), 'models', 'best_stone.pt'),
                    'device': 'auto',
                    'conf': 0.3,
                    'imgsz': 640,
                    'enable_visualization': True, # 可以设为False禁用可视化
                    'enable_ema_smoothing': True,
                    'ema_alpha': 0.35,
                    'sync_slop': 0.03,
                }],
                # output='screen'
            )
        ]
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
    pen_box_grasping_node = TimerAction(
        period=8.0,  # 8秒后启动，确保MoveIt完全启动
        actions=[
            Node(
                package='yolov8_grasping',
                executable='pen_box_grasping',  
                name='pen_box_grasping',
                output='screen',
                parameters=[
                    pen_box_moveit_config,
                    pen_box_task_config,
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

        # 延迟启动YOLO检测节点
        yolo_detector_node,
        
        # 启动手眼标定发布器
        hand_eye_tf_publisher,
        retime_server_launch,
        # 延迟启动抓取任务节点
        pen_box_grasping_node,
    ])
