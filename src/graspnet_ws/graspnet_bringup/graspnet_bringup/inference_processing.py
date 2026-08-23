"""Pure point-cloud and GraspGroup processing for the GraspNet ROS adapter."""

from typing import List, Optional, Tuple

import numpy as np
def rotmat_to_quat_xyzw(rot: np.ndarray) -> Tuple[float, float, float, float]:
    """Preserve the GraspNet adapter's historical quaternion conversion."""
    matrix = np.asarray(rot, dtype=np.float64)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    return float(qx), float(qy), float(qz), float(qw)


def vector(values, count: int, fallback: float) -> np.ndarray:
    out = np.full((count,), fallback, dtype=np.float32)
    if values is not None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        out[: min(count, values.shape[0])] = values[:count]
    return out


def filter_grasp_group_by_width(grasp_group, min_width_m: float, max_width_m: float):
    widths = np.asarray(getattr(grasp_group, "widths", None), dtype=np.float32).reshape(-1)
    if widths.size != len(grasp_group):
        raise RuntimeError("GraspGroup widths are missing or do not match candidate count.")
    valid = np.isfinite(widths) & (widths >= min_width_m) & (widths <= max_width_m)
    return grasp_group[valid], int(valid.sum())


def valid_camera_info(info) -> bool:
    return (
        bool(info.header.frame_id.strip())
        and info.width > 0
        and info.height > 0
        and np.isfinite(info.k[0])
        and np.isfinite(info.k[4])
        and info.k[0] > 0.0
        and info.k[4] > 0.0
    )


def workspace_mask(points: np.ndarray, x_min_m: float, x_max_m: float, y_min_m: float, y_max_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (
        (points[:, 0] >= float(x_min_m)) & (points[:, 0] <= float(x_max_m))
        & (points[:, 1] >= float(y_min_m)) & (points[:, 1] <= float(y_max_m))
    )


def support_plane_signed_distances(
    points: np.ndarray,
    plane_model: np.ndarray,
    base_from_camera_rotation: np.ndarray,
    max_tilt_deg: float,
) -> Optional[np.ndarray]:
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(plane[:3]))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Support-plane normal is degenerate.")
    plane /= norm
    normal_base = np.asarray(base_from_camera_rotation, dtype=np.float64).reshape(3, 3) @ plane[:3]
    if abs(float(normal_base[2])) < np.cos(np.deg2rad(float(max_tilt_deg))):
        return None
    if normal_base[2] < 0.0:
        plane *= -1.0
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ plane[:3] + plane[3]


def object_height_mask(signed_distances_m: np.ndarray, min_height_m: float, max_height_m: float) -> np.ndarray:
    return (signed_distances_m > float(min_height_m)) & (signed_distances_m <= float(max_height_m))


def filter_collision_free_grasps(
    detector_class, scene_points: np.ndarray, grasp_group, voxel_size_m: float,
    approach_distance_m: float, collision_threshold: float,
):
    collision_mask = np.asarray(
        detector_class(scene_points, voxel_size=float(voxel_size_m)).detect(
            grasp_group,
            approach_dist=float(approach_distance_m),
            collision_thresh=float(collision_threshold),
        ),
        dtype=bool,
    ).reshape(-1)
    if collision_mask.size != len(grasp_group):
        raise RuntimeError("Collision detector output does not match grasp candidate count.")
    return grasp_group[~collision_mask], int(collision_mask.sum())


def graspgroup_to_pose_metadata(grasp_group) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    if hasattr(grasp_group, "translations") and hasattr(grasp_group, "rotation_matrices"):
        translations = np.asarray(grasp_group.translations)
        rotations = np.asarray(grasp_group.rotation_matrices)
        count = int(translations.shape[0])
        scores = vector(getattr(grasp_group, "scores", None), count, 1.0)
        widths = vector(getattr(grasp_group, "widths", None), count, np.nan)
        depths = vector(getattr(grasp_group, "depths", None), count, np.nan)
        poses = [
            [*translations[index, :3], *rotmat_to_quat_xyzw(rotations[index])]
            for index in range(count)
        ]
        metadata = [
            (float(scores[index]), float(widths[index]), float(depths[index]))
            for index in range(count)
        ]
        return np.asarray(poses, dtype=np.float32), metadata

    array = next(
        (np.asarray(getattr(grasp_group, name))
         for name in ("grasp_group_array", "grasp_group", "gg_array")
         if hasattr(grasp_group, name)),
        np.asarray(grasp_group),
    )
    if array.ndim != 2 or array.shape[1] < 17:
        raise RuntimeError(f"Unexpected GraspGroup shape: {array.shape}")
    scores = array[:, 0].astype(np.float32)
    widths = array[:, 1].astype(np.float32)
    depths = array[:, 3].astype(np.float32)
    rotations = array[:, 4:13].reshape(-1, 3, 3).astype(np.float32)
    translations = array[:, 13:16].astype(np.float32)
    poses = [
        [*translations[index], *rotmat_to_quat_xyzw(rotations[index])]
        for index in range(array.shape[0])
    ]
    metadata = [
        (float(scores[index]), float(widths[index]), float(depths[index]))
        for index in range(array.shape[0])
    ]
    return np.asarray(poses, dtype=np.float32), metadata
