#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import LaunchConfigurationEquals
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    this_package_path = get_package_share_directory('visual_servo')


    
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
            'pointcloud.enable': 'true',
            # 'align_depth.enable': 'true',
            # 'enable_sync': 'true',
            'temporal_filter.enable': 'true',
            'spatial_filter.enable': 'true',
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
        "perception.demo.rviz",
    )

    
    ar_moveit = IncludeLaunchDescription(
        ar_moveit_launch, 
        # launch_arguments=ar_moveit_args
        launch_arguments={
      "use_rviz": "true",             # 保证会 include MoveIt 的 rviz launch
      "rviz_config": rviz_config_file, # 传给 moveit_rviz.launch.py 的参数名
       }.items(),
    )


    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("visual_servo"),
                "launch",
                "gazebo.launch.py",
            )
        ]),
        launch_arguments={
            # ✅ 覆盖 RViz 配置文件路径
            'rviz_config': os.path.join(
                this_package_path, 
                'rviz', 
                'perception.demo.rviz'
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
    retime_server_node = TimerAction(
    period=5.0,  # 比抓取节点更早
    actions=[
        Node(
            package="trajectory_retime_server",
            executable="retime_server",
            name="trajectory_retime_server",
            output="screen",
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

        # gazebo_node,
        # 启动手眼标定发布器
        hand_eye_tf_publisher,
        # retime_server_node,
    ])
