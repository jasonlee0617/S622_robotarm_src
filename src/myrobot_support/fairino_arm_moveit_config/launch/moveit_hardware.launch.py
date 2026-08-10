"""Real Fairino hardware: one control stack and one safe active MoveIt executor."""

from copy import deepcopy
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg
from nav2_common.launch import ReplaceString


_EXECUTION_CAPABILITIES = " ".join((
    "move_group/MoveGroupExecuteTrajectoryAction",
    "move_group/MoveGroupExecuteService",
))
_PLANNING_ONLY_CONTROLLER = "__planning_only_controller__"
_PLANNING_ONLY_JOINT = "__planning_only_joint__"


def _condition(executor: str) -> IfCondition:
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration("active_executor"), f"' == '{executor}'",
    ]))


def _inactive_condition(executor: str) -> IfCondition:
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration("active_executor"), f"' != '{executor}'",
    ]))


def _planning_only_parameters(common: dict) -> dict:
    """Keep planning available without exposing a real controller endpoint."""
    parameters = deepcopy(common)
    for name in (
        "moveit_simple_controller_manager",
        "moveit_manage_controllers",
        "trajectory_execution",
    ):
        parameters.pop(name, None)
    parameters.update({
        "allow_trajectory_execution": False,
        "disable_capabilities": _EXECUTION_CAPABILITIES,
        # Humble constructs TrajectoryExecutionManager even with execution disabled.
        # ROS 2 cannot represent an empty controller_names array, so provide a
        # local, non-executable placeholder instead of any real controller.
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": {
            "controller_names": [_PLANNING_ONLY_CONTROLLER],
            _PLANNING_ONLY_CONTROLLER: {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "joints": [_PLANNING_ONLY_JOINT],
                "default": False,
            },
        },
    })
    return parameters


def _moveit_parameters(kinematics_file: str, default_pipeline: str) -> dict:
    return (
        MoveItConfigsBuilder(
            "fairino_arm_moveit_descriptions",
            package_name="fairino_arm_moveit_config",
        )
        .robot_description_kinematics(file_path=f"config/{kinematics_file}")
        .planning_pipelines(default_planning_pipeline=default_pipeline)
        .trajectory_execution(
            file_path="config/moveit_controllers_hardware.yaml",
            moveit_manage_controllers=False,
        )
        .to_moveit_configs()
        .to_dict()
    )


def _move_group_configuration(*, active: bool) -> dict:
    should_publish = ParameterValue(
        LaunchConfiguration("publish_monitored_planning_scene"), value_type=bool
    )
    return {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": (
            ParameterValue(LaunchConfiguration("allow_trajectory_execution"), value_type=bool)
            if active
            else False
        ),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": (
            ParameterValue(LaunchConfiguration("disable_capabilities"), value_type=str)
            if active
            else ParameterValue(
                [
                    LaunchConfiguration("disable_capabilities"),
                    " ",
                    _EXECUTION_CAPABILITIES,
                ],
                value_type=str,
            )
        ),
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": ParameterValue(
            LaunchConfiguration("monitor_dynamics"), value_type=bool
        ),
    }


def _move_group(namespace: str, parameters: list[dict], condition: IfCondition) -> GroupAction:
    common = {
        "package": "moveit_ros_move_group",
        "executable": "move_group",
        "namespace": namespace,
        "name": "move_group",
        "output": "screen",
        "remappings": [
            ("joint_states", "/joint_states"),
            ("planning_scene", "/planning_scene"),
            ("collision_object", "/collision_object"),
            ("attached_collision_object", "/attached_collision_object"),
            ("trajectory_execution_event", "/trajectory_execution_event"),
        ],
        "parameters": parameters,
    }
    commands_file = (
        Path(get_package_share_directory("fairino_arm_moveit_config"))
        / "launch"
        / "gdb_settings.gdb"
    )
    return GroupAction(
        condition=condition,
        actions=[
            Node(
                condition=UnlessCondition(LaunchConfiguration("debug")),
                **common,
            ),
            Node(
                condition=IfCondition(LaunchConfiguration("debug")),
                prefix=[f"gdb -x {commands_file} --ex run --args"],
                arguments=["--debug"],
                **common,
            ),
        ],
    )


def _planning_pipeline_parameters(moveit_parameters: dict) -> dict:
    names = moveit_parameters["planning_pipelines"]
    return {
        "planning_pipelines": names,
        "default_planning_pipeline": moveit_parameters["default_planning_pipeline"],
        **{name: moveit_parameters[name] for name in names},
    }


def _rviz(config, namespace: str, moveit_parameters: dict) -> Node:
    """Run native MoveIt RViz against the selected namespaced server."""
    move_group_namespace = f"/{namespace}"
    planning_scene_topic = f"{move_group_namespace}/monitored_planning_scene"
    configured_rviz = ReplaceString(
        source_file=LaunchConfiguration("rviz_config"),
        replacements={
            'Move Group Namespace: ""': f"Move Group Namespace: {move_group_namespace}",
            "Move Group Namespace: /move_group_fairino": (
                f"Move Group Namespace: {move_group_namespace}"
            ),
            "Move Group Namespace: /move_group_kdl": (
                f"Move Group Namespace: {move_group_namespace}"
            ),
            "Planning Scene Topic: monitored_planning_scene": (
                f"Planning Scene Topic: {planning_scene_topic}"
            ),
            "Planning Scene Topic: /move_group_fairino/monitored_planning_scene": (
                f"Planning Scene Topic: {planning_scene_topic}"
            ),
            "Planning Scene Topic: /move_group_kdl/monitored_planning_scene": (
                f"Planning Scene Topic: {planning_scene_topic}"
            ),
        },
    )
    return Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", configured_rviz],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        remappings=[("joint_states", "/joint_states")],
        prefix=PythonExpression([
            "'gdb --ex run --args' if '",
            LaunchConfiguration("debug"),
            "'.lower() == 'true' else ''",
        ]),
        parameters=[
            config.robot_description,
            config.robot_description_semantic,
            config.joint_limits,
            {"robot_description_kinematics": moveit_parameters["robot_description_kinematics"]},
            _planning_pipeline_parameters(moveit_parameters),
        ],
    )


def generate_launch_description():
    package = Path(get_package_share_directory("fairino_arm_moveit_config"))
    rviz_config = MoveItConfigsBuilder(
        "fairino_arm_moveit_descriptions",
        package_name="fairino_arm_moveit_config",
    ).to_moveit_configs()
    robot_description = rviz_config.robot_description
    fairino_parameters = _moveit_parameters("kinematics_fairino.yaml", "fairino")
    kdl_parameters = _moveit_parameters("kinematics_kdl.yaml", "ompl")
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    robot_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robot_arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    hand_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["hand_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    moveit_actions = [
        _move_group(
            "move_group_fairino",
            [fairino_parameters, _move_group_configuration(active=True)],
            _condition("fairino"),
        ),
        _move_group(
            "move_group_fairino",
            [
                _planning_only_parameters(fairino_parameters),
                _move_group_configuration(active=False),
            ],
            _inactive_condition("fairino"),
        ),
        _move_group(
            "move_group_kdl",
            [kdl_parameters, _move_group_configuration(active=True)],
            _condition("kdl"),
        ),
        _move_group(
            "move_group_kdl",
            [
                _planning_only_parameters(kdl_parameters),
                _move_group_configuration(active=False),
            ],
            _inactive_condition("kdl"),
        ),
        GroupAction(
            condition=_condition("fairino"),
            actions=[_rviz(
                rviz_config,
                "move_group_fairino",
                fairino_parameters,
            )],
        ),
        GroupAction(
            condition=_condition("kdl"),
            actions=[_rviz(
                rviz_config,
                "move_group_kdl",
                kdl_parameters,
            )],
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "active_executor",
            default_value="fairino",
            choices=["fairino", "kdl"],
            description="Only this MoveIt server may execute real trajectories; restart to change it.",
        ),
        DeclareBooleanLaunchArg("debug", default_value=False),
        DeclareBooleanLaunchArg("use_rviz", default_value=True),
        DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True),
        DeclareBooleanLaunchArg(
            "publish_monitored_planning_scene", default_value=True
        ),
        DeclareBooleanLaunchArg("monitor_dynamics", default_value=False),
        DeclareLaunchArgument(
            "capabilities",
            default_value=rviz_config.move_group_capabilities["capabilities"],
        ),
        DeclareLaunchArgument(
            "disable_capabilities",
            default_value=rviz_config.move_group_capabilities["disable_capabilities"],
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(package / "config" / "moveit.rviz"),
        ),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(package / "launch" / "static_virtual_joint_tfs.launch.py")
        )),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(package / "launch" / "rsp.launch.py")
        )),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[robot_description, str(package / "config" / "ros2_controllers.yaml")],
        ),
        joint_state_broadcaster,
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[robot_arm_controller],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=robot_arm_controller,
            on_exit=[hand_controller],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=hand_controller,
            on_exit=moveit_actions,
        )),
    ])
