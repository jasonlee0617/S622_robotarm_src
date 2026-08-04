"""Small immutable records and SE(3) helpers used by the collector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class TransformMatrix:
    rotation: R
    translation: Tuple[float, float, float]

    def matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class ToolDeltaSpec:
    dx_m: float
    dy_m: float
    dz_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    @property
    def key(self) -> str:
        return (
            f"d=({self.dx_m:+.3f},{self.dy_m:+.3f},{self.dz_m:+.3f})m "
            f"r=({self.rx_deg:+.1f},{self.ry_deg:+.1f},{self.rz_deg:+.1f})deg"
        )


@dataclass(frozen=True)
class CandidatePose:
    idx: int
    description: str
    pose: PoseStamped
    base_T_ee: TransformMatrix
    spec: ToolDeltaSpec


@dataclass(frozen=True)
class AcceptedSampleQuality:
    center_error_px: float
    margin_px: float
    marker_side_px: float
    distance_m: float
    camera_model_error_px: float
    center_std_px: float
    depth_std_m: float
    marker_note: str
    model_note: str
    stable_note: str
    translation_mad_m: float = float("inf")
    rotation_mad_deg: float = float("inf")
    robot_translation_drift_m: float = float("inf")
    robot_rotation_drift_deg: float = float("inf")
    ippe_absolute_gap_px: float = float("inf")
    ippe_error_ratio: float = float("inf")
    ippe_non_ambiguous_frames: int = 0


@dataclass(frozen=True)
class AcceptedSampleRecord:
    robot_pose: TransformMatrix
    tracking_pose: TransformMatrix
    spec: ToolDeltaSpec
    quality: AcceptedSampleQuality
    candidate_idx: int
    image_stamp_ns: int


class CollectorGeometry:
    def __init__(self, *, base_frame: str):
        self.base_frame = base_frame

    @staticmethod
    def tf_to_matrix(transform) -> TransformMatrix:
        q, p = transform.transform.rotation, transform.transform.translation
        return TransformMatrix(R.from_quat([q.x, q.y, q.z, q.w]), (float(p.x), float(p.y), float(p.z)))

    @staticmethod
    def transform_from_xyz_rpy(xyz, rpy_deg) -> TransformMatrix:
        return TransformMatrix(R.from_euler("xyz", rpy_deg, degrees=True), tuple(float(value) for value in xyz))

    @staticmethod
    def matrix_to_pose_stamped(transform: TransformMatrix, frame_id: str, stamp) -> PoseStamped:
        q = transform.rotation.as_quat()
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose = Pose(
            position=Point(x=float(transform.translation[0]), y=float(transform.translation[1]), z=float(transform.translation[2])),
            orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        )
        return pose

    @staticmethod
    def compose(a: TransformMatrix, b: TransformMatrix) -> TransformMatrix:
        matrix = a.matrix() @ b.matrix()
        return TransformMatrix(R.from_matrix(matrix[:3, :3]), tuple(float(value) for value in matrix[:3, 3]))

    @staticmethod
    def from_matrix(matrix) -> TransformMatrix:
        return TransformMatrix(R.from_matrix(matrix[:3, :3]), tuple(float(value) for value in matrix[:3, 3]))

    @staticmethod
    def rotation_delta_deg(a: R, b: R) -> float:
        return math.degrees(float((a.inv() * b).magnitude()))

    def build_root_relative_candidate(self, *, idx: int, spec: ToolDeltaSpec,
                                      root_base_T_ee: TransformMatrix, now_msg) -> CandidatePose:
        delta = TransformMatrix(
            R.from_euler("xyz", (spec.rx_deg, spec.ry_deg, spec.rz_deg), degrees=True),
            (spec.dx_m, spec.dy_m, spec.dz_m),
        )
        base_T_ee = self.compose(root_base_T_ee, delta)
        return CandidatePose(
            idx=idx,
            description=f"#{idx} {spec.key}",
            pose=self.matrix_to_pose_stamped(base_T_ee, self.base_frame, now_msg()),
            base_T_ee=base_T_ee,
            spec=spec,
        )


class SampleManager:
    def __init__(self, *, translation_delta_m: float, rotation_delta_deg: float,
                 rotation_distance_deg: Callable):
        self.translation_delta_m = float(translation_delta_m)
        self.rotation_delta_deg = float(rotation_delta_deg)
        self._rotation_distance_deg = rotation_distance_deg
        self._accepted: List[AcceptedSampleRecord] = []

    @property
    def accepted_samples(self) -> List[AcceptedSampleRecord]:
        return self._accepted

    def reset(self) -> None:
        self._accepted.clear()

    def diverse(self, pose) -> Tuple[bool, str]:
        for record in self._accepted:
            translation = float(np.linalg.norm(np.asarray(pose.translation) - np.asarray(record.robot_pose.translation)))
            rotation = self._rotation_distance_deg(record.robot_pose.rotation, pose.rotation)
            if translation < self.translation_delta_m and rotation < self.rotation_delta_deg:
                return False, (
                    f"duplicate SE(3): dt={translation:.4f}m < {self.translation_delta_m:.4f}m, "
                    f"dr={rotation:.2f}deg < {self.rotation_delta_deg:.2f}deg"
                )
        return True, "new SE(3) motion"

    def record(self, *, robot_pose, tracking_pose, spec: ToolDeltaSpec,
               quality: AcceptedSampleQuality, candidate_idx: int,
               image_stamp_ns: int) -> AcceptedSampleRecord:
        record = AcceptedSampleRecord(
            robot_pose=robot_pose,
            tracking_pose=tracking_pose,
            spec=spec,
            quality=quality,
            candidate_idx=int(candidate_idx),
            image_stamp_ns=int(image_stamp_ns),
        )
        self._accepted.append(record)
        return record
