"""Fairino 路径规划 Gazebo 演示入口."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_SCENE_DEFAULTS = {
    "robot_profile": "fairino_arm_gripper",
    "enable_rviz": "true",
    "world": "empty",
    "use_sim_time": "true",
    "initial_positions_file": "",
    "enable_camera_model": "false",
    "scene_name": "multi_obstacle_3d_avoidance",
    "spawn_gazebo_scene_models": "true",
    "publish_planning_scene": "true",
    "publish_obstacle_markers": "true",
    "obstacle_marker_topic": "/demo_pathplanning/obstacle_markers",
}
_NODE_DEFAULTS = {
    "planning_client": "fairino",
    "move_group_namespace": "",
    "group_name": "robot_arm",
    "base_frame_name": "base_link",
    "ee_frame_name": "tool0",
    "joint_names": "j1,j2,j3,j4,j5,j6",
    "home_joints": "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0",
    "home_settle_timeout_s": "6.0",
    "default_pipeline_id": "fairino",
    "default_planner_id": "aapf_birrt*",
    "target_rpy_deg": "0,-180,0",
    "go_home_before_demo": "false",
    "auto_add_obstacle": "true",
    "remove_obstacle_after_demo": "true",
    "obstacle_name": "birrt_test_obstacle",
    "obstacle_position": "0.35,0.05,0.28",
    "obstacle_size": "0.18,0.45,0.35",
    "obstacle_boxes": "",
    "planning_scene_obstacle_padding_m": "0.03",
}
_LAUNCH_CONFIGURATIONS = {
    name: LaunchConfiguration(name) for name in (*_SCENE_DEFAULTS, *_NODE_DEFAULTS)
}


def _declare_launch_arguments():
    return [
        DeclareLaunchArgument(name, default_value=value, description="Gazebo 场景参数。")
        for name, value in _SCENE_DEFAULTS.items()
    ] + [
        DeclareLaunchArgument(name, default_value=value, description="路径规划节点参数。")
        for name, value in _NODE_DEFAULTS.items()
    ]


def _scene_paths(gz_share):
    scene_assets_dir = os.path.join(gz_share, "config", "scenes")
    return {
        "scene_assets_dir": scene_assets_dir,
        "scene_config_file": os.path.join(scene_assets_dir, "pathplanning_scenes.yaml"),
    }


def generate_launch_description():
    gz_share = get_package_share_directory("myrobot_simulation")
    scene_paths = _scene_paths(gz_share)
    myrobot_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **{name: _LAUNCH_CONFIGURATIONS[name] for name in _SCENE_DEFAULTS},
            **scene_paths,
            "rviz_config": os.path.join(gz_share, "rviz", "fairino_planning_test.rviz"),
        }.items(),
    )
    trajectory_plan_node = Node(
        package="myrobot_simulation",
        executable="trajectory_plan_node.py",
        name="trajectory_plan_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {**_NODE_DEFAULTS, **scene_paths},
            {name: _LAUNCH_CONFIGURATIONS[name] for name in _NODE_DEFAULTS},
            {
                "use_sim_time": _LAUNCH_CONFIGURATIONS["use_sim_time"],
                "scene_name": _LAUNCH_CONFIGURATIONS["scene_name"],
                "spawn_gazebo_scene_models": _LAUNCH_CONFIGURATIONS[
                    "spawn_gazebo_scene_models"
                ],
                "publish_planning_scene": _LAUNCH_CONFIGURATIONS[
                    "publish_planning_scene"
                ],
                "publish_obstacle_markers": _LAUNCH_CONFIGURATIONS[
                    "publish_obstacle_markers"
                ],
                "obstacle_marker_topic": _LAUNCH_CONFIGURATIONS[
                    "obstacle_marker_topic"
                ],
                "gazebo_world": _LAUNCH_CONFIGURATIONS["world"],
            },
        ],
    )
    delayed_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg="[trajectory_plan_demo] 启动 Fairino 路径规划演示。"),
            trajectory_plan_node,
        ],
    )
    return LaunchDescription([*_declare_launch_arguments(), myrobot_simulation, delayed_node])
