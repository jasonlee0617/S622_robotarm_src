import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")

    robot_profile_arg = DeclareLaunchArgument(
        "robot_profile", default_value="s622_gripper",
        description="Robot profile used by gazebo.launch.py.")
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config", default_value=os.path.join(gz_share, "rviz", "ik_test.rviz"),
        description="RViz config for IK test.")
    enable_rviz_arg = DeclareLaunchArgument("enable_rviz", default_value="true")
    world_arg = DeclareLaunchArgument("world", default_value="empty")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")

    fairino_ns_arg = DeclareLaunchArgument(
        "fairino_move_group_namespace", default_value="/move_group_fairino")
    kdl_ns_arg = DeclareLaunchArgument(
        "kdl_move_group_namespace", default_value="/move_group_kdl")

    group_name_arg = DeclareLaunchArgument("group_name", default_value="robot_arm")
    base_frame_arg = DeclareLaunchArgument("base_frame_name", default_value="base_link")
    ee_frame_arg = DeclareLaunchArgument("ee_frame_name", default_value="grasp_frame")
    joint_names_arg = DeclareLaunchArgument("joint_names", default_value="j1,j2,j3,j4,j5,j6")
    home_joints_arg = DeclareLaunchArgument(
        "home_joints", default_value="-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")

    ik_plugin_arg = DeclareLaunchArgument(
        "ik_plugin", default_value="fairino",
        description="IK solver to execute: fairino or kdl.")
    planning_pipeline_arg = DeclareLaunchArgument(
        "planning_pipeline", default_value="fairino",
        description="Planning pipeline: fairino or ompl.")
    planning_algorithm_arg = DeclareLaunchArgument(
        "planning_algorithm", default_value="birrt*",
        description="Planner id: birrt*, rrt*, RRTConnect, RRTConnectFast, etc. Fairino planner ids are lowercase only.")
    ik_timeout_arg = DeclareLaunchArgument(
        "ik_timeout", default_value="3.0",
        description="IK service timeout in seconds.")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            "robot_profile": LaunchConfiguration("robot_profile"),
            "rviz_config": LaunchConfiguration("rviz_config"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "world": LaunchConfiguration("world"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "default_planning_pipeline": LaunchConfiguration("planning_pipeline"),
            "enable_camera_model": "false",
        }.items(),
    )

    ik_test_node = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg=[
                "[ik_test_demo] ik=", LaunchConfiguration("ik_plugin"),
                ", pipeline=", LaunchConfiguration("planning_pipeline"),
                ", planner=", LaunchConfiguration("planning_algorithm"),
            ]),
            Node(
                package="gazebo_launch",
                executable="ik_test_node.py",
                name="ik_test_node",
                output="screen",
                emulate_tty=True,
                parameters=[{
                    "fairino_move_group_namespace": LaunchConfiguration("fairino_move_group_namespace"),
                    "kdl_move_group_namespace": LaunchConfiguration("kdl_move_group_namespace"),
                    "group_name": LaunchConfiguration("group_name"),
                    "base_frame_name": LaunchConfiguration("base_frame_name"),
                    "ee_frame_name": LaunchConfiguration("ee_frame_name"),
                    "joint_names": LaunchConfiguration("joint_names"),
                    "home_joints": LaunchConfiguration("home_joints"),
                    "ik_timeout": LaunchConfiguration("ik_timeout"),
                    "execution_ik_plugin": LaunchConfiguration("ik_plugin"),
                    "execution_pipeline": LaunchConfiguration("planning_pipeline"),
                    "planning_algorithm": LaunchConfiguration("planning_algorithm"),
                }],
            ),
        ],
    )

    return LaunchDescription([
        robot_profile_arg, rviz_config_arg, enable_rviz_arg, world_arg,
        use_sim_time_arg, fairino_ns_arg, kdl_ns_arg, group_name_arg,
        base_frame_arg, ee_frame_arg, joint_names_arg, home_joints_arg,
        ik_plugin_arg, planning_pipeline_arg, planning_algorithm_arg,
        ik_timeout_arg, gazebo_launch, ik_test_node,
    ])
