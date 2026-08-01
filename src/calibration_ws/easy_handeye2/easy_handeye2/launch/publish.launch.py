from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arg_name = DeclareLaunchArgument('name')
    arg_storage_directory = DeclareLaunchArgument('storage_directory', default_value='')

    handeye_publisher = Node(package='easy_handeye2', executable='handeye_publisher', name='handeye_publisher', parameters=[{
        'name': LaunchConfiguration('name'),
        'storage_directory': LaunchConfiguration('storage_directory'),
    }])

    return LaunchDescription([
        arg_name,
        arg_storage_directory,
        handeye_publisher,
    ])
