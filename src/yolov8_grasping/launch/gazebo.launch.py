# gazebo.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 获取当前包路径
    this_package_path = get_package_share_directory('yolov8_grasping')  
    robot_desc_path=get_package_share_directory('s622_moveit_descriptions')
    # robot_desc_path=get_package_share_directory('fairino_description')

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(this_package_path, 'rviz', 'pick_drop_moveit.rviz'),
        description='Path to RViz config file'
    )

    # 启动Gazebo
    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('ros_gz_sim') + '/launch/gz_sim.launch.py']),
        launch_arguments=[('gz_args', 'empty.sdf -r')]  # 使用自定义空世界
    )

    clock_bridge_node = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=['/world/empty/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
    output='both',
    parameters=[{'use_sim_time': True}],
    remappings=[('/world/empty/clock', '/clock')]  
    )

    packagepath = get_package_share_directory('s622_moveit_config')  
    # packagepath = get_package_share_directory('fairino3_v6_moveit2_config')  

    # MoveIt配置
    moveit_config = (MoveItConfigsBuilder("s622", package_name="s622_moveit_config") 
                .robot_description(this_package_path + '/config/yolov8_grasping_gazebo.friction.urdf.xacro') 
                .robot_description_semantic('config/s622_moveit_descriptions.srdf') 
                # .moveit_cpp(this_package_path + "/config/movelt_cpp.yaml") 
                .sensors_3d(packagepath + '/config/sensors_3d.yaml')
                .planning_pipelines(pipelines=["ompl"]).to_moveit_configs()
    )
    # moveit_config = MoveItConfigsBuilder("fairino3_v6", package_name="fairino3_v6_moveit2_config") \
    #                 .robot_description(this_package_path + '/config/yolov8_grasping_gazebo.friction.urdf.xacro') \
    #                 .robot_description_semantic('config/fairino3_v6_robot.srdf') \
    #                 .moveit_cpp(this_package_path + "/config/movelt_cpp.yaml") \
    #                 .to_moveit_configs()

    #　将机械臂添加到Gazebo
    gz_urdf= moveit_config.robot_description['robot_description'].replace('package://s622_moveit_descriptions',robot_desc_path) 
    robot_to_gazebo_node = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-string', gz_urdf,
               '-x','0.0','-y','0.0','-z','0.0',
               '-R','0','-P','0','-Y','0',   # yaw=90°
               '-name','robot_arm']
    )
    # 发布机械臂状态
    robot_desc_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description,
                     {'use_sim_time': True},    #必须使用仿真时间
                     { "publish_frequency":100.0,},
                     ],
    )

    # 启动RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration('rviz_config')],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': True},
        ],
    )

    #启动关节状态发布器，arm组控制器，夹抓控制器
    controller_spawner_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster","robot_arm_controller","hand_controller"
            # "joint_state_broadcaster","fairino3_controller"
        ],
        parameters=[{'use_sim_time': True}],
        output="screen",
    )

    # 启动move_group
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {'use_sim_time': True}],
    )
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
    # ===== 手眼标定发布节点 =====
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{"calibration_name": "robot_calibration",}],  # 直接使用固定值
        output='screen'
    )
    return LaunchDescription([
        rviz_config_arg,
        gazebo_node,  # 启动Gazebo仿真环境
        clock_bridge_node,  # 时钟桥接
        robot_to_gazebo_node,#启动gazebo环境机械臂
        robot_desc_node, #启动机械臂状态节点
        rviz_node,  # 启动RViz
        controller_spawner_node,#启动关节状态发布器
        move_group_node,  # 启动MoveIt的move_group
        # realsense_launch,
        # hand_eye_tf_publisher
    ])
