"""MoveIt-related launch helpers for gz_launch."""

from typing import Dict, Optional

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from .controllers import moveit_controller_config
from .robot_profiles import RobotProfile
from .yaml_loader import load_yaml, package_file


def _as_xacro_bool(value: bool) -> str:
    return "true" if value else "false"


def build_moveit_config(
    profile: RobotProfile,
    default_planning_pipeline: Optional[str] = None,
    enable_camera_model: Optional[bool] = None,
):
    """Create a MoveItConfigs object for the selected robot profile."""
    default_pipeline = default_planning_pipeline or profile.default_planning_pipeline
    camera_enabled = profile.has_camera if enable_camera_model is None else enable_camera_model
    return (
        MoveItConfigsBuilder(profile.moveit_config_name, package_name=profile.moveit_config_package)
        .robot_description(
            package_file("gz_launch", profile.gazebo_xacro),
            mappings={
                "enable_camera": _as_xacro_bool(camera_enabled),
                "controllers_file": package_file(
                    profile.moveit_config_package, profile.controllers_file
                ),
                "initial_positions_file": package_file(
                    profile.moveit_config_package, profile.initial_positions_file
                ),
            },
        )
        .robot_description_semantic(profile.semantic_file)
        .robot_description_kinematics(
            package_file(profile.moveit_config_package, profile.default_kinematics_file)
        )
        .planning_pipelines(
            pipelines=profile.planning_pipelines,
            default_planning_pipeline=default_pipeline,
        )
        .to_moveit_configs()
    )


def planning_parameter_configs(profile: RobotProfile) -> Dict[str, Dict]:
    """Load planning and kinematics YAML as dictionaries for launch injection."""
    return {
        "fairino_planning": load_yaml(profile.moveit_config_package, profile.planning_pipeline_file),
        "kinematics_fairino": load_yaml(profile.moveit_config_package, profile.kinematics_fairino_file),
        "kinematics_kdl": load_yaml(profile.moveit_config_package, profile.kinematics_kdl_file),
        "planning_core": load_yaml("fairino_planning_core", "config/common_planning_params.yaml"),
        "aapf_birrt_star_core": load_yaml("fairino_planning_core", "config/aapf_birrt*_params.yaml"),
        "birrt_star_core": load_yaml("fairino_planning_core", "config/birrt*_params.yaml"),
        "rrt_star_core": load_yaml("fairino_planning_core", "config/rrt*_params.yaml"),
        "ik_core": load_yaml("fairino_planning_core", "config/ik_params.yaml"),
        "cartesian_path_planner": load_yaml(
            "fairino_planning_core", "config/cartesian_path_planner_params.yaml"
        ),
        "controllers": moveit_controller_config(profile),
    }


def robot_description_with_package_paths(moveit_config, profile: RobotProfile) -> str:
    """Expand ALL package:// references to filesystem paths for ros_gz_sim."""
    import re
    description = moveit_config.robot_description["robot_description"]

    def resolve(m):
        pkg = m.group(1)
        try:
            return "file://" + get_package_share_directory(pkg)
        except Exception:
            return m.group(0)

    return re.sub(r'package://([^/]+)', resolve, description)


def rviz_node(moveit_config, profile: RobotProfile, rviz_config: str, use_sim_time: bool):
    params = planning_parameter_configs(profile)
    # RViz MotionPlanning display defaults to root namespace; remap to fairino
    # move_group so GUI planning/execution stays aligned with simulation.
    remappings = [
        ("get_planning_scene", "/move_group_fairino/get_planning_scene"),
        ("plan_kinematic_path", "/move_group_fairino/plan_kinematic_path"),
        ("query_planner_interfaces", "/move_group_fairino/query_planner_interfaces"),
        ("compute_cartesian_path", "/move_group_fairino/compute_cartesian_path"),
        ("execute_trajectory", "/move_group_fairino/execute_trajectory"),
        ("move_action", "/move_group_fairino/move_action"),
        ("monitored_planning_scene", "/move_group_fairino/monitored_planning_scene"),
    ]
    return Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        remappings=remappings,
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            params["fairino_planning"],
            {"use_sim_time": use_sim_time},
        ],
    )


def move_group_nodes(moveit_config, profile: RobotProfile, use_sim_time: bool):
    """Build fairino and kdl move_group nodes for the selected profile."""
    params = planning_parameter_configs(profile)
    remappings = [
        ("joint_states", "/joint_states"),
        # Obstacle publishers in demos use root topics. Keep both move_group
        # instances subscribed there so PlanningScene collision objects are not
        # lost under /move_group_fairino or /move_group_kdl namespaces.
        ("planning_scene", "/planning_scene"),
        ("collision_object", "/collision_object"),
        ("attached_collision_object", "/attached_collision_object"),
        (
            f"{profile.arm_controller.lstrip('/')}/follow_joint_trajectory",
            f"{profile.arm_controller}/follow_joint_trajectory",
        ),
    ]
    if profile.has_gripper and profile.hand_controller:
        remappings.append(
            (
                f"{profile.hand_controller.lstrip('/')}/follow_joint_trajectory",
                f"{profile.hand_controller}/follow_joint_trajectory",
            )
        )

    fairino_parameters = [
        moveit_config.to_dict(),
        params["kinematics_fairino"],
        params["controllers"],
        params["fairino_planning"],
        params["planning_core"],
        params["aapf_birrt_star_core"],
        params["birrt_star_core"],
        params["rrt_star_core"],
        params["ik_core"],
        {"use_sim_time": use_sim_time},
    ]
    fairino_cartesian_parameters = [
        *fairino_parameters,
        params["cartesian_path_planner"],
    ]

    kdl_parameters = [
        moveit_config.to_dict(),
        params["kinematics_kdl"],
        params["controllers"],
        {"use_sim_time": use_sim_time},
    ]
    if profile.enable_fairino_pipeline_on_kdl:
        kdl_parameters.insert(3, params["fairino_planning"])

    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace="move_group_fairino",
            name="move_group",
            output="screen",
            remappings=remappings,
            parameters=fairino_parameters,
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace="move_group_kdl",
            name="move_group",
            output="screen",
            remappings=remappings,
            parameters=kdl_parameters,
        ),
        Node(
            package="fairino_planning_ros",
            executable="fairino_cartesian_path_server",
            name="fairino_cartesian_path_server",
            output="screen",
            parameters=fairino_cartesian_parameters,
        ),
    ]
