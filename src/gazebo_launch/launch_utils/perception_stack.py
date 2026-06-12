"""Perception and servo helpers for gazebo_yolo.launch.py."""

from launch_ros.actions import Node

from .robot_profiles import RobotProfile
from .yaml_loader import load_yaml


def camera_bridge_nodes(use_sim_time: bool):
    return [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/image_raw@sensor_msgs/msg/Image@gz.msgs.Image"],
            output="screen",
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/camera/image@sensor_msgs/msg/Image@ignition.msgs.Image",
                "/camera/depth_image@sensor_msgs/msg/Image@ignition.msgs.Image",
                "/camera/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo",
            ],
            remappings=[
                ("/camera/image", "/camera/camera/color/image_raw"),
                ("/camera/depth_image", "/camera/camera/aligned_depth_to_color/image_raw"),
                ("/camera/camera_info", "/camera/camera/aligned_depth_to_color/camera_info"),
            ],
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]


def servo_node(moveit_config, profile: RobotProfile, kinematics_kdl_config, use_sim_time: bool):
    servo_yaml = load_yaml(profile.moveit_config_package, profile.servo_parameters_file)
    servo_yaml["move_group_name"] = profile.group_name
    servo_yaml["planning_frame"] = profile.planning_frame
    servo_yaml["ee_frame_name"] = profile.ee_frame_name
    servo_yaml["command_out_topic"] = f"{profile.arm_controller}/joint_trajectory"
    servo_yaml["command_out_type"] = "trajectory_msgs/JointTrajectory"
    servo_yaml["cartesian_command_in_topic"] = "/servo_node/delta_twist_cmds"
    servo_yaml["joint_command_in_topic"] = "/servo_node/delta_joint_cmds"

    return Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            kinematics_kdl_config,
            {"moveit_servo": servo_yaml},
            {"use_sim_time": use_sim_time},
        ],
    )

