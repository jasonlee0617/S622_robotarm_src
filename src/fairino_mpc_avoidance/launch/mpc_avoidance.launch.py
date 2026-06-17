from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('fairino_mpc_avoidance')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation time'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(pkg_dir, 'rviz', 'mpc_avoidance.rviz'),
            description='RViz config file'),

        Node(
            package='fairino_mpc_avoidance',
            executable='mpc_avoidance_node',
            name='mpc_avoidance_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'mpc_params.yaml'),
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[
                ('/joint_states', '/joint_states'),
                ('/planned_trajectory', '/fairino_planner/trajectory'),
                ('/detected_obstacles', '/yolo/obstacles'),
            ]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
