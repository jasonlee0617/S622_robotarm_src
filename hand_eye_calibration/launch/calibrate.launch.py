import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
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
        get_package_share_directory("hand_eye_calibration"),
        "rviz",
        "moveit_with_camera.rviz",
    )
    ar_moveit_args = {
        # "rviz_config_file": rviz_config_file,
        #oak_camera
        "use_rviz": "true",   
        "rviz_config": rviz_config_file, # 传给 moveit_rviz.launch.py 的参数名
    }.items()
    ar_moveit = IncludeLaunchDescription(
        ar_moveit_launch, launch_arguments=ar_moveit_args
    )

    aruco_params = os.path.join(
        get_package_share_directory("hand_eye_calibration"),
        "config",
        "aruco_parameters.yaml",
    )
    aruco_recognition_node = Node(
        package="ros2_aruco", executable="aruco_node", parameters=[aruco_params]
    )

    calibration_args = {
        "name": "robot_calibration",
        "calibration_type": "eye_on_base",
        "robot_base_frame": "base_link",
        "robot_effector_frame": "calibration_marker",
        # "robot_effector_frame": "grasp_frame",
        # "robot_effector_frame": "wrist3_link",
        "tracking_base_frame": "camera_color_optical_frame",
        "tracking_marker_frame": "calibration_aruco",
    }

    calibration_aruco_publisher = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        parameters=[
            {
                "tracking_base_frame": calibration_args["tracking_base_frame"],
                "tracking_marker_frame": calibration_args["tracking_marker_frame"],
                "marker_id": 1,
            }
        ],
    )

    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("easy_handeye2"),
                    "launch",
                    "calibrate.launch.py",
                )
            ]
        ),
        launch_arguments=calibration_args.items(),
    )

    aruco_visualize = Node(
        package="hand_eye_calibration",
        executable="visualize_aruco_marker.py",
        name="aruco_pose_estimator",
        output="screen",
    )


    # static transform publisher for camera_link to world
    static_tf_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.48", "0.66", "0.73", "0", "0", "0", "world", "camera_link"],
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(realsense)
    # ld.add_action(oak_camera)
    ld.add_action(static_tf_publisher)
    ld.add_action(ar_moveit)
    ld.add_action(aruco_recognition_node)
    ld.add_action(calibration_aruco_publisher)
    ld.add_action(easy_handeye2)
    ld.add_action(aruco_visualize)
    return ld
