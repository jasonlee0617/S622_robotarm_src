import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    # 加载gazebo.launch.py
    gazebo_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([
        get_package_share_directory('yolov8_grasping') + '/launch/gazebo.launch.py'])
    )

    this_package_path=get_package_share_directory('yolov8_grasping')
    robot_desc_path=get_package_share_directory('s622_moveit_descriptions')
    #打开模型文件
    box_desc=open(this_package_path+'/config/box.urdf').read()
    case_desc=open(this_package_path+'/config/case.urdf').read()

    box_to_gazebo_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[ '-string', box_desc, '-x', '0.29', '-y', '0.376', '-z','0.0' ,'-name', 'box']
    )
  
    case_to_gazebo_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[ '-string', case_desc, '-x', '0.0', '-y', '0.5', '-z','0.0' ,'-name', 'case']
    )

    ##pick_drop##
    #启动编写的MoveIt2　Python控制节点
    # pick_drop_node = Node(
    #     name="pick_drop",
    #     package="yolov8_grasping",
    #     executable="pick_drop",
    #     output="screen",
    #     # parameters=[{'use_sim_time': True}],
    # )
    ##pick_drop##

    pick_drop_node = TimerAction(
        period=8.0,  # 8秒后启动，确保MoveIt完全启动
        actions=[
            Node(
                package='yolov8_grasping',
                executable='pick_drop_ik',  
                name='pick_drop_ik',
                output='screen'
            )
        ]
    )
    ##pick_drop_ik##

    return LaunchDescription([
            gazebo_launch,
            box_to_gazebo_node,
            case_to_gazebo_node,
            pick_drop_node
        ])



