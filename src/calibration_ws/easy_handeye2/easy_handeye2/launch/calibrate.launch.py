# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch.conditions import IfCondition
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import Node

# from easy_handeye2.common_launch import arg_calibration_type, arg_tracking_base_frame, arg_tracking_marker_frame, arg_robot_base_frame, \
#     arg_robot_effector_frame


# def generate_launch_description():
#     arg_name = DeclareLaunchArgument('name')

#     calibration_type = LaunchConfiguration('calibration_type')
#     node_dummy_calib_eih = Node(package='tf2_ros', executable='static_transform_publisher', name='dummy_publisher',
#                                 condition=IfCondition('False'),
#                                 arguments=f'--x 0 --y 0 --z 0.1 --qx 0 --qy 0 --qz 0 --qw 1'.split(' ') + ['--frame-id', LaunchConfiguration('robot_effector_frame'),
#                                                                                                            '--child-frame-id', LaunchConfiguration('tracking_base_frame')])

#     node_dummy_calib_eob = Node(package='tf2_ros', executable='static_transform_publisher', name='dummy_publisher',
#                                 condition=IfCondition('True'),
#                                 arguments=f'--x 1 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1'.split(' ') + ['--frame-id', LaunchConfiguration('robot_base_frame'),
#                                                                                                          '--child-frame-id', LaunchConfiguration('tracking_base_frame')])

#     handeye_server = Node(package='easy_handeye2', executable='handeye_server', name='handeye_server', parameters=[{
#         'name': LaunchConfiguration('name'),
#         'calibration_type': LaunchConfiguration('calibration_type'),
#         'tracking_base_frame': LaunchConfiguration('tracking_base_frame'),
#         'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame'),
#         'robot_base_frame': LaunchConfiguration('robot_base_frame'),
#         'robot_effector_frame': LaunchConfiguration('robot_effector_frame'),
#     }])

#     handeye_rqt_calibrator = Node(package='easy_handeye2', executable='rqt_calibrator.py',
#                                   name='handeye_rqt_calibrator',
#                                   # arguments=['--ros-args', '--log-level', 'debug'],
#                                   parameters=[{
#                                       'name': LaunchConfiguration('name'),
#                                       'calibration_type': LaunchConfiguration('calibration_type'),
#                                       'tracking_base_frame': LaunchConfiguration('tracking_base_frame'),
#                                       'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame'),
#                                       'robot_base_frame': LaunchConfiguration('robot_base_frame'),
#                                       'robot_effector_frame': LaunchConfiguration('robot_effector_frame'),
#                                   }])

#     return LaunchDescription([
#         arg_name,
#         arg_calibration_type,
#         arg_tracking_base_frame,
#         arg_tracking_marker_frame,
#         arg_robot_base_frame,
#         arg_robot_effector_frame,
#         node_dummy_calib_eih,
#         node_dummy_calib_eob,
#         handeye_server,
#         handeye_rqt_calibrator,
#     ])
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from easy_handeye2.common_launch import (
    arg_calibration_type,
    arg_tracking_base_frame,
    arg_tracking_marker_frame,
    arg_robot_base_frame,
    arg_robot_effector_frame,
)

def generate_launch_description():
    arg_name = DeclareLaunchArgument("name")
    arg_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="false")
    arg_storage_directory = DeclareLaunchArgument("storage_directory", default_value="")

    handeye_server = Node(
        package="easy_handeye2",
        executable="handeye_server",
        name="handeye_server",
        parameters=[{
            "name": LaunchConfiguration("name"),
            "calibration_type": LaunchConfiguration("calibration_type"),
            "tracking_base_frame": LaunchConfiguration("tracking_base_frame"),
            "tracking_marker_frame": LaunchConfiguration("tracking_marker_frame"),
            "robot_base_frame": LaunchConfiguration("robot_base_frame"),
            "robot_effector_frame": LaunchConfiguration("robot_effector_frame"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "storage_directory": LaunchConfiguration("storage_directory"),
        }],
        output="screen",
    )

    handeye_rqt_calibrator = Node(
        package="easy_handeye2",
        executable="rqt_calibrator.py",
        name="handeye_rqt_calibrator",
        parameters=[{
            "name": LaunchConfiguration("name"),
            "calibration_type": LaunchConfiguration("calibration_type"),
            "tracking_base_frame": LaunchConfiguration("tracking_base_frame"),
            "tracking_marker_frame": LaunchConfiguration("tracking_marker_frame"),
            "robot_base_frame": LaunchConfiguration("robot_base_frame"),
            "robot_effector_frame": LaunchConfiguration("robot_effector_frame"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
        output="screen",
    )

    return LaunchDescription([
        arg_name,
        arg_calibration_type,
        arg_tracking_base_frame,
        arg_tracking_marker_frame,
        arg_robot_base_frame,
        arg_robot_effector_frame,
        arg_use_sim_time,
        arg_storage_directory,
        handeye_server,
        handeye_rqt_calibrator,
    ])
