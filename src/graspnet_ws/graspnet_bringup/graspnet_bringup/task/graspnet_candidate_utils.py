"""Pure GraspNet candidate geometry shared by grasp executors."""

import copy
from typing import Optional, Sequence

import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

from graspnet_bringup.task.task_types import GraspCandidate


def _copy_pose(pose: Pose) -> Pose:
    return copy.deepcopy(pose)


def _score_at(scores: Sequence[float], idx: int) -> Optional[float]:
    return float(scores[idx]) if idx < len(scores) else None


def _finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def _positive_or_none(value: Optional[float]) -> Optional[float]:
    return value if value is not None and value > 0.0 else None


def _metadata_at(metadata: Sequence[float], idx: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    offset = idx * 3
    if len(metadata) < offset + 3:
        return None, None, None
    score = _finite_or_none(metadata[offset])
    width = _positive_or_none(_finite_or_none(metadata[offset + 1]))
    depth = _positive_or_none(_finite_or_none(metadata[offset + 2]))
    return score, width, depth


def _preopen_positions_from_width(width_m: Optional[float]) -> Optional[tuple[float, float]]:
    if width_m is None:
        return None
    half_width = float(width_m) * 0.5
    return half_width, -half_width


def _candidate_indices(count: int, limit: int) -> list[int]:
    return list(range(min(count, max(1, int(limit)))))


def build_candidates(poses, scores: Sequence[float], metadata: Sequence[float], limit: int):
    """Build the bounded GraspNet candidate list published by inference."""
    return [
        GraspCandidate(
            idx=index,
            camera_pose=poses[index],
            score=score if score is not None else _score_at(scores, index),
            width_m=width,
            depth_m=depth,
        )
        for index in _candidate_indices(len(poses), limit)
        for score, width, depth in [_metadata_at(metadata, index)]
    ]


def _apply_orientation_correction(pose: Pose, correction_rpy_deg: Sequence[float]) -> Pose:
    corrected = _copy_pose(pose)
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    out = (R.from_quat(quat) * R.from_euler("xyz", correction_rpy_deg, degrees=True)).as_quat()
    corrected.orientation.x = float(out[0])
    corrected.orientation.y = float(out[1])
    corrected.orientation.z = float(out[2])
    corrected.orientation.w = float(out[3])
    return corrected


def _pose_axis(pose: Pose, axis_index: int) -> np.ndarray:
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return R.from_quat(quat).as_matrix()[:, int(axis_index)]


def _offset_pose_along_axis(pose: Pose, axis_index: int, distance_m: float) -> Pose:
    out = _copy_pose(pose)
    axis = _pose_axis(pose, axis_index)
    out.position.x += float(axis[0]) * float(distance_m)
    out.position.y += float(axis[1]) * float(distance_m)
    out.position.z += float(axis[2]) * float(distance_m)
    return out


def _make_lift_pose(grasp: Pose, lift_distance: float) -> Pose:
    lift_pose = _copy_pose(grasp)
    lift_pose.position.z += float(lift_distance)
    return lift_pose


def prepare_candidate(
    candidate: GraspCandidate,
    *,
    grasp_offset_m: float,
    orientation_rpy_deg: Sequence[float],
    approach_distance_m: float,
    lift_distance_m: float,
) -> GraspCandidate:
    """Fill execution poses from a candidate already expressed in ``base_link``."""
    if candidate.base_pose is None:
        raise ValueError("candidate has no base-frame pose")
    raw_grasp = candidate.base_pose
    if candidate.depth_m is not None:
        raw_grasp = _offset_pose_along_axis(
            raw_grasp, 0, candidate.depth_m + float(grasp_offset_m)
        )
    candidate.grasp = _apply_orientation_correction(raw_grasp, orientation_rpy_deg)
    candidate.approach = _offset_pose_along_axis(
        candidate.grasp, 2, -float(approach_distance_m)
    )
    candidate.lift = _make_lift_pose(candidate.grasp, lift_distance_m)
    candidate.preopen_positions = _preopen_positions_from_width(candidate.width_m)
    return candidate


def candidate_geometry_rejection(
    candidate: GraspCandidate,
    *,
    min_width_m: float,
    max_width_m: float,
    max_approach_tilt_deg: float,
    max_jaw_z_abs: float,
) -> str:
    """Return the existing geometric rejection text, or an empty string when safe."""
    if candidate.depth_m is None:
        return "missing_graspnet_depth"
    if candidate.width_m is None:
        return "missing_graspnet_width"
    if not min_width_m <= candidate.width_m <= max_width_m:
        return f"width_out_of_range:{candidate.width_m:.4f}"
    if candidate.grasp is None:
        return "missing_grasp_pose"
    approach_axis = _pose_axis(candidate.grasp, 2)
    jaw_axis = _pose_axis(candidate.grasp, 0)
    down_dot = float(np.clip(np.dot(approach_axis, [0.0, 0.0, -1.0]), -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(down_dot)))
    if tilt_deg > max_approach_tilt_deg:
        return f"approach_tilt:{tilt_deg:.1f}deg"
    jaw_z_abs = abs(float(jaw_axis[2]))
    if jaw_z_abs > max_jaw_z_abs:
        return f"jaw_not_horizontal:z_abs={jaw_z_abs:.3f}"
    return ""
