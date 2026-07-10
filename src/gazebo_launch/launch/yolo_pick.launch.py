import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    gz_share = get_package_share_directory('gazebo_launch')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            "world": "arm_on_the_table",
            "rviz_config": os.path.join(gz_share, "rviz", "gazebo_launch.rviz"),
            "publish_frequency": "100.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "true",
            "spawn_z": "1.02",
            "controller_spawn_delay": "5.0",
        }.items(),
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

    yolo_obb = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory("yolo_perception"),
            "launch", "yolov8_obb.launch.py",
        )
    ),
    )
    yolo_pick_node = Node(
        name="yolo_pick_drop",
        package="gazebo_launch",
        executable="robot_control_from_UI_node.py",
        output="screen",
        # parameters=[moveit_config.to_dict(),
        #             # {"use_sim_time": True},
        #             ],
    )

    return LaunchDescription([
            gazebo_launch,
            retime_server_launch,
            yolo_obb,
            yolo_pick_node,
        ])

