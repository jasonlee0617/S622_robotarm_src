import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


GAZEBO_LAUNCH_ARGUMENTS = {
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

NODE_PARAMS = {
    "planning_client": "fairino",
    "move_group_namespace": "",
    "group_name": "robot_arm",
    "base_frame_name": "base_link",
    "ee_frame_name": "grasp_frame",
    "joint_names": "j1,j2,j3,j4,j5,j6",
    "home_joints": "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0",
    "home_settle_timeout_s": 6.0,
    "default_pipeline_id": "fairino",
    "default_planner_id": "aapf_birrt*",
    "target_rpy_deg": "0,-180,0",
    "go_home_before_demo": False,
    "auto_add_obstacle": True,
    "remove_obstacle_after_demo": True,
    "obstacle_name": "birrt_test_obstacle",
    "obstacle_position": "0.35,0.05,0.28",
    "obstacle_size": "0.18,0.45,0.35",
    "obstacle_boxes": "",
    "scene_name": "multi_obstacle_3d_avoidance",
    "spawn_gazebo_scene_models": True,
    "gazebo_world": "empty",
    "publish_planning_scene": True,
    "publish_obstacle_markers": True,
    "obstacle_marker_topic": "/demo_pathplanning/obstacle_markers",
    "planning_scene_obstacle_padding_m": 0.03,
}


def _scene_paths(gz_share):
    scene_assets_dir = os.path.join(gz_share, "config", "scenes")
    return {
        "scene_assets_dir": scene_assets_dir,
        "scene_config_file": os.path.join(scene_assets_dir, "pathplanning_scenes.yaml"),
    }


def generate_launch_description():
    gz_share = get_package_share_directory("gazebo_launch")
    scene_paths = _scene_paths(gz_share)

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gazebo.launch.py")),
        launch_arguments={
            **GAZEBO_LAUNCH_ARGUMENTS,
            **scene_paths,
            "rviz_config": os.path.join(gz_share, "rviz", "fairino_planning_test.rviz"),
        }.items(),
    )

    trajectory_plan_node = Node(
        package="gazebo_launch",
        executable="trajectory_plan_node.py",
        name="trajectory_plan_node",
        output="screen",
        emulate_tty=True,
        parameters=[{**NODE_PARAMS, **scene_paths}],
    )

    delayed_node = TimerAction(
        period=5.0,
        actions=[
            LogInfo(
                msg=(
                    "[trajectory_plan_demo] client=fairino, namespace_override=, "
                    "pipeline=fairino, planner=aapf_birrt*"
                )
            ),
            trajectory_plan_node,
        ],
    )

    return LaunchDescription([gazebo_launch, delayed_node])
