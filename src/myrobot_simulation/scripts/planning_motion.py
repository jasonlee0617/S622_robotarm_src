"""Small MoveIt operations shared by the planning demo and benchmark."""

import numpy as np


def joint_trajectory_path_length(trajectory):
    if trajectory is None:
        return 0.0
    return sum(float(np.linalg.norm(np.asarray(second.positions) - np.asarray(first.positions)))
               for first, second in zip(trajectory.points, trajectory.points[1:]))


def execute_joint_trajectory(moveit_arm, trajectory, error_code):
    try:
        moveit_arm.execute(trajectory)
        success = moveit_arm.wait_until_executed()
        return success, "" if success else error_code()
    except Exception:
        return False, "execution_exception"
