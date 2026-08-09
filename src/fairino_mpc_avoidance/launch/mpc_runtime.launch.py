from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# 该入口仅组合 mpc_avoidance.launch.py，不创建 ROS 节点参数。

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("fairino_mpc_avoidance"),
                    "launch",
                    "mpc_avoidance.launch.py",
                ])
            )
        )
    ])
