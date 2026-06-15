import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_GZ_SHARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _GZ_SHARE not in sys.path:
    sys.path.insert(0, _GZ_SHARE)

from launch_utils.launch_parsing import as_bool


def _value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context)).strip()


def _float_value(context, name: str) -> float:
    return float(_value(context, name))


def _launch_setup(context, *args, **kwargs):
    gz_share = get_package_share_directory("gazebo_launch")
    handeye_share = get_package_share_directory("hand_eye_calibration")
    easy_share = get_package_share_directory("easy_handeye2")

    use_sim_time = as_bool(_value(context, "use_sim_time"))
    marker_id = int(_value(context, "marker_id"))

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo_yolo.launch.py")
        ),
        launch_arguments={
            "robot_profile": _value(context, "robot_profile"),
            "world": _value(context, "world"),
            "rviz_config": _value(context, "rviz_config"),
            "enable_rviz": _value(context, "enable_rviz"),
            "use_sim_time": _value(context, "use_sim_time"),
            "publish_frequency": _value(context, "publish_frequency"),
            "default_planning_pipeline": _value(context, "default_planning_pipeline"),
            "enable_camera_model": "true",
            "enable_camera_bridge": "true",
            "enable_servo": _value(context, "enable_servo"),
            "spawn_name": _value(context, "spawn_name"),
            "spawn_x": _value(context, "spawn_x"),
            "spawn_y": _value(context, "spawn_y"),
            "spawn_z": _value(context, "spawn_z"),
            "spawn_roll": _value(context, "spawn_roll"),
            "spawn_pitch": _value(context, "spawn_pitch"),
            "spawn_yaw": _value(context, "spawn_yaw"),
            "robot_spawn_delay": _value(context, "robot_spawn_delay"),
            "controller_spawn_delay": _value(context, "controller_spawn_delay"),
        }.items(),
    )

    marker_sdf = os.path.join(
        gz_share,
        "worlds",
        "models",
        "aruco_5x5_250_id1",
        "model.sdf",
    )
    marker_spawn = TimerAction(
        period=_float_value(context, "marker_spawn_delay"),
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-file",
                    marker_sdf,
                    "-name",
                    _value(context, "marker_model_name"),
                    "-x",
                    _value(context, "marker_x"),
                    "-y",
                    _value(context, "marker_y"),
                    "-z",
                    _value(context, "marker_z"),
                    "-R",
                    _value(context, "marker_roll"),
                    "-P",
                    _value(context, "marker_pitch"),
                    "-Y",
                    _value(context, "marker_yaw"),
                    "-allow_renaming",
                    "false",
                ],
            )
        ],
    )

    aruco_params = os.path.join(
        handeye_share,
        "config",
        "aruco_parameters_gazebo.yaml",
    )
    aruco_node = Node(
        package="ros2_aruco",
        executable="aruco_node",
        parameters=[aruco_params, {"use_sim_time": use_sim_time}],
        additional_env={"PYTHONNOUSERSITE": "1"},
        output="screen",
    )

    tracking_base_frame = _value(context, "tracking_base_frame")
    tracking_marker_frame = _value(context, "tracking_marker_frame")
    aruco_tf = Node(
        package="hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            {
                "tracking_base_frame": tracking_base_frame,
                "tracking_marker_frame": tracking_marker_frame,
                "marker_id": marker_id,
                "aruco_topic": "/aruco_markers",
                "stamp_policy": _value(context, "aruco_tf_stamp_policy"),
                "log_every_sec": _float_value(context, "aruco_tf_log_every_sec"),
                "use_sim_time": use_sim_time,
            }
        ],
    )

    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(easy_share, "launch", "calibrate.launch.py")
        ),
        launch_arguments={
            "name": _value(context, "calibration_name"),
            "calibration_type": "eye_in_hand",
            "robot_base_frame": _value(context, "robot_base_frame"),
            "robot_effector_frame": _value(context, "robot_effector_frame"),
            "tracking_base_frame": tracking_base_frame,
            "tracking_marker_frame": tracking_marker_frame,
            "use_sim_time": _value(context, "use_sim_time"),
        }.items(),
    )

    actions = [
        gazebo,
        marker_spawn,
        aruco_node,
        aruco_tf,
        TimerAction(
            period=_float_value(context, "easy_handeye2_delay"),
            actions=[easy_handeye2],
        ),
    ]

    if as_bool(_value(context, "auto_collect")):
        auto_params = os.path.join(
            handeye_share,
            "config",
            "auto_calibration_collector.yaml",
        )
        actions.append(
            TimerAction(
                period=_float_value(context, "auto_collector_delay"),
                actions=[
                    Node(
                        package="hand_eye_calibration",
                        executable="auto_calibration_collector.py",
                        name="auto_calibration_collector",
                        output="screen",
                        additional_env={"PYTHONNOUSERSITE": "1"},
                        parameters=[
                            auto_params,
                            {
                                "use_sim_time": use_sim_time,
                                "base_frame": _value(context, "robot_base_frame"),
                                "ee_frame": _value(context, "robot_effector_frame"),
                                "tracking_base_frame": tracking_base_frame,
                                "tracking_marker_frame": tracking_marker_frame,
                                "marker_id": marker_id,
                                "marker_size_m": _float_value(context, "marker_size"),
                                "image_topic": _value(context, "image_topic"),
                                "camera_info_topic": _value(context, "camera_info_topic"),
                                "aruco_dictionary_id": _value(context, "aruco_dictionary_id"),
                                "ik_plugin": _value(context, "auto_collector_ik_plugin"),
                                "planning_pipeline_id": _value(
                                    context, "auto_collector_planning_pipeline"
                                ),
                                "planner_id": _value(context, "auto_collector_planner_id"),
                            },
                        ],
                    )
                ],
            )
        )

    if as_bool(_value(context, "visualize_aruco")):
        actions.append(
            Node(
                package="hand_eye_calibration",
                executable="visualize_aruco_marker.py",
                name="aruco_pose_estimator",
                output="screen",
                parameters=[
                    {
                        "image_topic": _value(context, "image_topic"),
                        "camera_info_topic": _value(context, "camera_info_topic"),
                        "marker_size": _float_value(context, "marker_size"),
                        "aruco_dictionary_id": _value(context, "aruco_dictionary_id"),
                        "use_sim_time": use_sim_time,
                    }
                ],
                additional_env={"PYTHONNOUSERSITE": "1"},
            ) 
        )

    return actions


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_profile", default_value="s622_gripper_handeye"),
            DeclareLaunchArgument("world", default_value="calibration_table"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(gz_share, "rviz", "calibration_gazebo.rviz"),
            ),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("publish_frequency", default_value="30.0"),
            DeclareLaunchArgument("default_planning_pipeline", default_value="ompl"),
            DeclareLaunchArgument("enable_servo", default_value="false"),
            DeclareLaunchArgument("spawn_name", default_value=""),
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="0.0"),
            DeclareLaunchArgument("spawn_z", default_value="1.02"),
            DeclareLaunchArgument("spawn_roll", default_value="0.0"),
            DeclareLaunchArgument("spawn_pitch", default_value="0.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("robot_spawn_delay", default_value="5.0"),
            DeclareLaunchArgument("controller_spawn_delay", default_value="8.0"),
            DeclareLaunchArgument("calibration_name", default_value="robot_calibration"),
            DeclareLaunchArgument("robot_base_frame", default_value="base_link"),
            DeclareLaunchArgument("robot_effector_frame", default_value="grasp_frame"),
            DeclareLaunchArgument("tracking_base_frame", default_value="camera_color_optical_frame"),
            DeclareLaunchArgument("tracking_marker_frame", default_value="calibration_aruco"),
            DeclareLaunchArgument("marker_id", default_value="1"),
            DeclareLaunchArgument("marker_size", default_value="0.07"),
            DeclareLaunchArgument("marker_model_name", default_value="calibration_aruco_board"),
            DeclareLaunchArgument("marker_x", default_value="0.25"),
            DeclareLaunchArgument("marker_y", default_value="0.0"),
            DeclareLaunchArgument("marker_z", default_value="1.02"),
            DeclareLaunchArgument("marker_roll", default_value="1.5708"),
            DeclareLaunchArgument("marker_pitch", default_value="0.0"),
            DeclareLaunchArgument("marker_yaw", default_value="0.0"),
            DeclareLaunchArgument("marker_spawn_delay", default_value="10.0"),
            DeclareLaunchArgument("visualize_aruco", default_value="true"),
            DeclareLaunchArgument("easy_handeye2_delay", default_value="12.0"),
            DeclareLaunchArgument("aruco_tf_stamp_policy", default_value="now"),
            DeclareLaunchArgument("aruco_tf_log_every_sec", default_value="5.0"),
            DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera/aligned_depth_to_color/camera_info"),
            DeclareLaunchArgument("aruco_dictionary_id", default_value="DICT_5X5_250"),
            DeclareLaunchArgument("auto_collect", default_value="false"),
            DeclareLaunchArgument("auto_collector_delay", default_value="15.0"),
            DeclareLaunchArgument("auto_collector_ik_plugin", default_value="fairino"),
            DeclareLaunchArgument("auto_collector_planning_pipeline", default_value="fairino"),
            DeclareLaunchArgument("auto_collector_planner_id", default_value="birrt*"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
