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
        get_package_share_directory('gazebo_launch') + '/launch/gazebo_yolo.launch.py'])
    )
    retime_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("trajectory_retime_server"),
                "launch",
                "retime_server.launch.py",
            )
        )
    )
    yolo_obb = Node(
        package='gazebo_launch',
        executable='yolo_detector_obb_node.py',  
        name='yolo_obb_detector',
        output='screen',
        parameters=[{"use_sim_time": True}],
    )
   
    pen_box_grasping_node = Node(
        package='gazebo_launch',
        executable='pen_box_grasping_node.py',  
        name='pen_box_grasping',
        output='screen',
        parameters=[{"use_sim_time": True}],
    )


    return LaunchDescription([
            gazebo_launch,
            retime_server_launch,
            yolo_obb,
            pen_box_grasping_node
        ])



