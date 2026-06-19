"""Geometry helpers: TransformMatrix, CandidatePose, CollectorGeometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R

from .sample_types import CandidateSpec


@dataclass
class TransformMatrix:
    rotation: R
    translation: Tuple[float, float, float]

    def matrix(self):
        m = np.eye(4)
        m[:3, :3] = self.rotation.as_matrix()
        m[:3, 3] = self.translation
        return m


@dataclass
class CandidatePose:
    idx: int
    description: str
    pose: PoseStamped
    base_T_ee: TransformMatrix
    family: str
    removable: bool
    spec: Optional[CandidateSpec] = None


class CollectorGeometry:
    def __init__(
        self,
        *,
        base_frame: str,
        ee_frame: str,
        tracking_base_frame: str,
        tracking_marker_frame: str,
        max_candidate_attempts: int,
    ):
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.tracking_base_frame = tracking_base_frame
        self.tracking_marker_frame = tracking_marker_frame
        self.max_candidate_attempts = int(max_candidate_attempts)

    @staticmethod
    def tf_to_matrix(transform) -> TransformMatrix:
        q = transform.transform.rotation
        p = transform.transform.translation
        return TransformMatrix(
            rotation=R.from_quat([q.x, q.y, q.z, q.w]),
            translation=(float(p.x), float(p.y), float(p.z)),
        )

    @staticmethod
    def transform_to_matrix(transform) -> TransformMatrix:
        q = transform.rotation
        p = transform.translation
        return TransformMatrix(
            rotation=R.from_quat([q.x, q.y, q.z, q.w]),
            translation=(float(p.x), float(p.y), float(p.z)),
        )

    @staticmethod
    def transform_from_xyz_rpy(xyz, rpy_deg) -> TransformMatrix:
        return TransformMatrix(
            rotation=R.from_euler("xyz", [float(v) for v in rpy_deg], degrees=True),
            translation=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )

    @staticmethod
    def matrix_to_pose_stamped(transform: TransformMatrix, frame_id: str, stamp) -> PoseStamped:
        q = transform.rotation.as_quat()
        pose = Pose()
        pose.position = Point(
            x=float(transform.translation[0]),
            y=float(transform.translation[1]),
            z=float(transform.translation[2]),
        )
        pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.pose = pose
        return ps

    @staticmethod
    def compose(a: TransformMatrix, b: TransformMatrix) -> TransformMatrix:
        return CollectorGeometry.from_matrix(a.matrix() @ b.matrix())

    @staticmethod
    def from_matrix(m) -> TransformMatrix:
        return TransformMatrix(
            rotation=R.from_matrix(m[:3, :3]),
            translation=(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
        )

    @staticmethod
    def rotation_delta_deg(a: R, b: R) -> float:
        return math.degrees(float((a.inv() * b).magnitude()))

    def build_visibility_candidates(
        self,
        *,
        reference_base_T_ee: TransformMatrix,
        candidate_specs: List[CandidateSpec],
        workspace_status: Callable[[Tuple[float, float, float]], Tuple[bool, str]],
        now_msg: Callable[[], object],
    ) -> List[CandidatePose]:
        candidates = []
        for spec in candidate_specs:
            desired_translation = (
                float(reference_base_T_ee.translation[0] + spec.base_x),
                float(reference_base_T_ee.translation[1] + spec.base_y),
                float(reference_base_T_ee.translation[2] + spec.base_z),
            )
            desired_rotation = reference_base_T_ee.rotation * R.from_euler(
                "xyz",
                [spec.pitch, spec.yaw, spec.roll],
                degrees=True,
            )
            desired_base_T_ee = TransformMatrix(
                rotation=desired_rotation,
                translation=desired_translation,
            )
            workspace_ok, workspace_note = workspace_status(desired_base_T_ee.translation)
            if not workspace_ok:
                continue
            idx = len(candidates) + 1
            candidates.append(
                CandidatePose(
                    idx=idx,
                    description=(
                        f"{spec.family} {spec.source} base_offset x={spec.base_x:+.3f}m y={spec.base_y:+.3f}m "
                        f"z={spec.base_z:+.3f}m pitch={spec.pitch:+.1f}deg "
                        f"yaw={spec.yaw:+.1f}deg roll={spec.roll:+.1f}deg"
                    ),
                    pose=self.matrix_to_pose_stamped(desired_base_T_ee, self.base_frame, now_msg()),
                    base_T_ee=desired_base_T_ee,
                    family=spec.family,
                    removable=spec.removable,
                    spec=spec,
                )
            )
            if len(candidates) >= self.max_candidate_attempts:
                break
        return candidates
