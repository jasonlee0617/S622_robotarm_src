"""Controller launch helpers for Gazebo + MoveIt simulations."""

from typing import Dict, List

from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

from .robot_profiles import RobotProfile


def moveit_controller_config(profile: RobotProfile) -> Dict:
    """Build a MoveItSimpleControllerManager config from a robot profile."""
    controller_names: List[str] = profile.controller_names
    config = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": controller_names,
        },
    }

    config["moveit_simple_controller_manager"][profile.arm_controller] = {
        "type": "FollowJointTrajectory",
        "joints": profile.arm_joints,
        "action_ns": "follow_joint_trajectory",
        "default": True,
    }

    if profile.has_gripper and profile.hand_controller and profile.hand_joints:
        config["moveit_simple_controller_manager"][profile.hand_controller] = {
            "type": "FollowJointTrajectory",
            "joints": profile.hand_joints,
            "action_ns": "follow_joint_trajectory",
            "default": True,
        }

    return config


def controller_spawner_actions(profile: RobotProfile):
    """Spawn ros2_control controllers one at a time after the robot exists."""
    spawners = [
        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                controller,
                "-c",
                "/controller_manager",
            ],
            output="screen",
        )
        for controller in profile.spawner_controller_names
    ]
    if not spawners:
        return []

    # Concurrent load/configure requests race the Gazebo ros2_control manager.
    # In particular, a failed joint-state broadcaster leaves MoveIt without
    # /joint_states until the whole simulation is restarted.
    actions = [spawners[0]]
    for previous, current in zip(spawners, spawners[1:]):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=previous,
                    on_exit=[current],
                )
            )
        )
    return actions
