import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    yolo_share = get_package_share_directory("yolo_perception")
    grasping_share = get_package_share_directory("yolov8_grasping")

    # ── Gazebo + YOLO perception (simulation side) ──
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo_yolo.launch.py")
        )
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
        package="yolo_perception",
        executable="yolo_detector_obb_gazebo.py",
        name="yolo_obb_detector",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── pen_box_grasping node (business logic) — launched directly to avoid
    #     duplicating camera / MoveIt / YOLO / retime startup that
    #     gazebo_launch already owns ──
    pen_box_grasping = Node(
        package="yolov8_grasping",
        executable="pen_box_grasping",
        name="pen_box_grasping",
        output="screen",
        parameters=[
            os.path.join(grasping_share, "config", "pen_box_moveit.yaml"),
            os.path.join(grasping_share, "config", "pen_box_task.yaml"),
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([
        gazebo_launch,
        retime_server_launch,
        yolo_obb,
        pen_box_grasping,
    ])
