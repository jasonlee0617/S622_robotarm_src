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
        get_package_share_directory('gz_launch') + '/launch/gazebo_yolo.launch.py'])
    )

    yolo_obb = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory("yolov8_obb"),
            "launch", "yolov8_obb.launch.py",
        )
    ),
    )
    yolo_pick_node = Node(
        name="yolo_pick_drop",
        package="gz_launch",
        executable="robot_control_from_UI_node.py",
        output="screen",
        # parameters=[moveit_config.to_dict(),
        #             # {"use_sim_time": True},
        #             ],
    )

    return LaunchDescription([
            gazebo_launch,
            yolo_obb,
            yolo_pick_node,
        ])



