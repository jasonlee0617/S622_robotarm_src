import numpy as np


def path_length(trajectory) -> float:
    """所有关节的路径长度（L2）"""
    points = trajectory.points
    total = 0.0
    for i in range(1, len(points)):
        p1 = np.array(points[i - 1].positions, dtype=float)
        p2 = np.array(points[i].positions, dtype=float)
        total += float(np.linalg.norm(p2 - p1))
    return total


def joint_subset_path_length(trajectory, joint_indices) -> float:
    """某些关节子集的路径长度（例如手腕关节）"""
    points = trajectory.points
    total = 0.0
    for i in range(1, len(points)):
        p1 = np.array([points[i - 1].positions[j] for j in joint_indices], dtype=float)
        p2 = np.array([points[i].positions[j] for j in joint_indices], dtype=float)
        total += float(np.linalg.norm(p2 - p1))
    return total


def select_best_path(paths, wrist_weight: float = 50.0, wrist_joint_indices=(2, 3, 4)):
    """
    选 cost 最小的轨迹：
    cost = path_length(all) + wrist_weight * path_length(wrist joints)
    """
    best = None
    best_cost = float("inf")
    for traj in paths:
        cost = path_length(traj) + wrist_weight * joint_subset_path_length(traj, wrist_joint_indices)
        if cost < best_cost:
            best_cost = cost
            best = traj
    return best
