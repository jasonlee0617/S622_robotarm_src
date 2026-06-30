import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    grasping_share = get_package_share_directory("yolov8_grasping")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo_yolo.launch.py")
        ),
        launch_arguments={
            "robot_profile": "s622_gripper_handeye",
            "world": "visual_grasping_table",
            "enable_rviz": "true",
            "use_sim_time": "true",
            "camera_fps": "60",
            "camera_image_width": "1024",
            "camera_image_height": "728",
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
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{"calibration_name": "robot_calibration"}],
        output="screen",
    )
    yolo_obb = Node(
        package="yolo_perception",
        executable="yolo_detector_obb_gazebo.py",
        name="yolo_obb_detector",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "model_path": "yolo-obb-gazebo-1024.pt",
                "imgsz": 1024,
                "conf": 0.5,
            }
        ],
    )
    visual_grasping = Node(
        package="yolov8_grasping",
        executable="visual_grasping",
        name="visual_grasping",
        output="screen",
        parameters=[
            os.path.join(grasping_share, "config", "pen_box_moveit.yaml"),
            os.path.join(grasping_share, "config", "pen_box_task.yaml"),
            {
                "use_sim_time": True,
                "camera_mode": "eye_in_hand",
            },
        ],
    )
    return LaunchDescription(
        [
            gazebo_launch,
            retime_server_launch,
            hand_eye_tf_publisher,
            yolo_obb,
            visual_grasping,
        ]
    )
