import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def load_yaml(package_name, file_name):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_name)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    ar_model_config = LaunchConfiguration("ar_model")
    ar_model_arg = DeclareLaunchArgument(
        "ar_model",
        default_value="mk1",
        choices=["mk1", "mk2", "mk3"],
        description="Model of dummy2",
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                )
            ]
        )
    )
    oak_camera = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory("depthai_ros_driver"),
            "launch", "camera.launch.py",
        )
    ),
    launch_arguments={
            # 启用 Realsense 兼容模式
            "rs_compat": "true",
            # "use_rviz": "true",
        }.items(),
    )

    aruco_params = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "aruco_parameters.yaml",
    )
    aruco_recognition_node = Node(
        package="ros2_aruco", executable="aruco_node", parameters=[aruco_params]
    )

    hand_eye_tf_publisher = Node(
        package="hand_eye_calibration",
        executable="handeye_publisher.py",
        name="handeye_publisher",
        parameters=[{"calibration_name": "robot_calibration"}],
    )

    follow_aruco_node = Node(
        package="hand_eye_calibration",
        executable="follow_aruco_marker.py",
        name="follow_aruco_marker",
        output="screen",
    )

    ar_moveit_launch = PythonLaunchDescriptionSource(
        [
            os.path.join(
                # get_package_share_directory("fairino3_v6_moveit2_config"),
                get_package_share_directory("s622_moveit_config"),
                "launch",
                "demo.launch.py",
            )
        ]
    )
    rviz_config_file = os.path.join(
        get_package_share_directory("hand_eye_calibration"), "rviz", "validate.rviz"
    )
    ar_moveit_args = {
        "include_gripper": "False",
        # "rviz_config_file": rviz_config_file,
        # "ar_model_config": ar_model_config,
        "use_rviz": "true",             # 保证会 include MoveIt 的 rviz launch
        "rviz_config": rviz_config_file 
    }.items()
    ar_moveit = IncludeLaunchDescription(
        ar_moveit_launch, launch_arguments=ar_moveit_args
    )

    return LaunchDescription(
        [
            # ar_model_arg,
            realsense,
            # oak_camera,
            hand_eye_tf_publisher,
            aruco_recognition_node,
            follow_aruco_node,
            ar_moveit,
        ]
    )
