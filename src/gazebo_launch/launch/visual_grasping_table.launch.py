import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _load_srdf_group_state(package_name, relative_path, state_name, group_name):
    srdf_path = os.path.join(get_package_share_directory(package_name), relative_path)
    root = ET.parse(srdf_path).getroot()
    for group_state in root.findall("group_state"):
        if group_state.get("name") != state_name or group_state.get("group") != group_name:
            continue
        names = []
        positions = []
        for joint in group_state.findall("joint"):
            names.append(joint.get("name"))
            positions.append(float(joint.get("value")))
        if names and positions:
            return names, positions
    raise RuntimeError(f"Missing SRDF group_state '{state_name}' for group '{group_name}' in {srdf_path}")


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    grasping_share = get_package_share_directory("yolov8_grasping")
    pos1_joint_names, pos1_joint_positions = _load_srdf_group_state(
        "fairino_arm_moveit_config",
        "config/fairino_arm_moveit_descriptions.srdf",
        "pos1",
        "robot_arm",
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "robot_profile": "fairino_arm_gripper_handeye",
            "world": "visual_grasping_table",
            "rviz_config": os.path.join(gz_share, "rviz", "visual_grasping_table.rviz"),
            "enable_rviz": "true",
            "use_sim_time": "true",
            "publish_frequency": "30.0",
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": "true",
            "camera_fps": "60",
            "camera_image_width": "1024",
            "camera_image_height": "728",
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
    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{
            "calibration_name": "robot_calibration",
            "storage_directory": str(Path.home() / "fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim"),
        }],
        output="screen",
    )
    yolo_obb = Node(
        package="yolo_perception",
        executable="yolo_detector_obb.py",
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
            {
                "use_sim_time": True,
                "camera_mode": "eye_in_hand",
                "startup_joint_state_name": "pos1",
                "startup_joint_names": pos1_joint_names,
                "startup_joint_positions": pos1_joint_positions,
            },
            os.path.join(grasping_share, "config", "yolo_visual_grasping.yaml"),
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
