"""ros2_control controller spawning helpers for Gazebo simulations."""

from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

from .robot_profiles import RobotProfile


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
                OnProcessExit(target_action=previous, on_exit=[current])
            )
        )
    return actions
