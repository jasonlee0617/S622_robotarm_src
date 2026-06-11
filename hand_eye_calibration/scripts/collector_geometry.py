from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R, Slerp

from sample_manager import CandidateSpec


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
    base_T_cam: TransformMatrix
    prediction_note: str = ""
    spec: Optional[CandidateSpec] = None
    projected_u_px: float = float("nan")
    projected_v_px: float = float("nan")
    projected_margin_px: float = float("nan")
    projected_marker_px: float = float("nan")
    projected_center_error_px: float = float("nan")
    projected_distance_m: float = float("nan")
    segment_count: int = 1


class CollectorGeometry:
    def __init__(
        self,
        *,
        base_frame: str,
        ee_frame: str,
        tracking_base_frame: str,
        tracking_marker_frame: str,
        max_candidate_attempts: int,
        segment_step_m: float,
        segment_step_deg: float,
    ):
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.tracking_base_frame = tracking_base_frame
        self.tracking_marker_frame = tracking_marker_frame
        self.max_candidate_attempts = int(max_candidate_attempts)
        self.segment_step_m = float(segment_step_m)
        self.segment_step_deg = float(segment_step_deg)

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
    def inverse(a: TransformMatrix) -> TransformMatrix:
        return CollectorGeometry.from_matrix(np.linalg.inv(a.matrix()))

    @staticmethod
    def from_matrix(m) -> TransformMatrix:
        return TransformMatrix(
            rotation=R.from_matrix(m[:3, :3]),
            translation=(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
        )

    @staticmethod
    def rotation_delta_deg(a: R, b: R) -> float:
        return math.degrees(float((a.inv() * b).magnitude()))

    @staticmethod
    def normalize(v, fallback=None):
        arr = np.array(v, dtype=float)
        n = float(np.linalg.norm(arr))
        if n < 1.0e-9:
            if fallback is None:
                return arr
            return np.array(fallback, dtype=float)
        return arr / n

    def look_at_camera_pose(
        self,
        marker_base: np.ndarray,
        camera_base: np.ndarray,
        roll_deg: float,
        tilt_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0,
    ) -> TransformMatrix:
        z_axis = self.normalize(marker_base - camera_base, fallback=[1.0, 0.0, 0.0])
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(z_axis, world_up))) > 0.94:
            world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = self.normalize(np.cross(z_axis, world_up), fallback=[0.0, -1.0, 0.0])
        y_axis = self.normalize(np.cross(z_axis, x_axis), fallback=[0.0, 0.0, -1.0])
        rot = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        if abs(tilt_x_deg) > 1.0e-6 or abs(tilt_y_deg) > 1.0e-6:
            rot = rot * R.from_euler("xy", [math.radians(tilt_x_deg), math.radians(tilt_y_deg)])
        if abs(roll_deg) > 1.0e-6:
            rot = rot * R.from_euler("z", math.radians(roll_deg))
        return TransformMatrix(
            rotation=rot,
            translation=(float(camera_base[0]), float(camera_base[1]), float(camera_base[2])),
        )

    def project_marker_for_camera(
        self,
        base_T_cam: TransformMatrix,
        marker_base: np.ndarray,
        check_projected_marker: Callable[[np.ndarray], Tuple[bool, str]],
    ) -> Tuple[bool, str]:
        cam_T_base = self.inverse(base_T_cam)
        marker_h = np.array([marker_base[0], marker_base[1], marker_base[2], 1.0], dtype=float)
        marker_cam = (cam_T_base.matrix() @ marker_h)[:3]
        return check_projected_marker(marker_cam)

    def build_visibility_candidates(
        self,
        *,
        lookup_tf: Callable[[str, str, float], TransformMatrix],
        candidate_planner,
        workspace_status: Callable[[Tuple[float, float, float]], Tuple[bool, str]],
        projection_metrics: Callable[[np.ndarray], Tuple[bool, object]],
        check_projected_marker: Callable[[np.ndarray], Tuple[bool, str]],
        now_msg: Callable[[], object],
        logger_debug: Callable[[str], None],
    ) -> List[CandidatePose]:
        try:
            base_T_cam = lookup_tf(self.base_frame, self.tracking_base_frame, 2.0)
            ee_T_cam = lookup_tf(self.ee_frame, self.tracking_base_frame, 2.0)
            base_T_marker = lookup_tf(self.base_frame, self.tracking_marker_frame, 2.0)
        except Exception as exc:
            raise RuntimeError(
                "Cannot build marker-centric candidates. Required TF chain is missing: "
                f"{exc}"
            ) from exc

        cam_pos = np.array(base_T_cam.translation, dtype=float)
        marker_pos = np.array(base_T_marker.translation, dtype=float)
        camera_axes = base_T_cam.rotation.as_matrix()
        right_axis = self.normalize(camera_axes[:, 0], fallback=[0.0, -1.0, 0.0])
        up_axis = self.normalize(-camera_axes[:, 1], fallback=[0.0, 0.0, 1.0])
        forward_axis = self.normalize(marker_pos - cam_pos, fallback=camera_axes[:, 2])

        inv_ee_T_cam = self.inverse(ee_T_cam)
        candidates = []
        for spec in candidate_planner.build_specs():
            desired_cam_pos = (
                cam_pos
                + right_axis * spec.right
                + up_axis * spec.up
                - forward_axis * spec.dist
            )
            desired_base_T_cam = self.look_at_camera_pose(
                marker_pos,
                desired_cam_pos,
                spec.roll,
                tilt_x_deg=spec.tilt_x,
                tilt_y_deg=spec.tilt_y,
            )
            visible, reason = self.project_marker_for_camera(
                desired_base_T_cam,
                marker_pos,
                check_projected_marker,
            )
            if not visible:
                logger_debug(
                    f"Skip candidate {spec.source}: right={spec.right:.3f} up={spec.up:.3f} "
                    f"dist={spec.dist:.3f} roll={spec.roll:.1f}: {reason}"
                )
                continue
            cam_T_base = self.inverse(desired_base_T_cam)
            marker_cam = (cam_T_base.matrix() @ np.array([*marker_pos, 1.0], dtype=float))[:3]
            metrics_ok, metrics = projection_metrics(marker_cam)
            if not metrics_ok:
                logger_debug(
                    f"Skip candidate {spec.source}: projection metrics unavailable: {metrics}"
                )
                continue
            desired_base_T_ee = self.compose(desired_base_T_cam, inv_ee_T_cam)
            workspace_ok, workspace_note = workspace_status(desired_base_T_ee.translation)
            if not workspace_ok:
                logger_debug(
                    f"Skip candidate {spec.source}: right={spec.right:.3f} up={spec.up:.3f} "
                    f"dist={spec.dist:.3f} roll={spec.roll:.1f}: {workspace_note}"
                )
                continue
            segment_count = len(self.interpolated_transforms(base_T_cam, desired_base_T_cam))
            idx = len(candidates) + 1
            candidates.append(
                CandidatePose(
                    idx=idx,
                    description=(
                        f"{spec.source} look-at right={spec.right:+.3f}m up={spec.up:+.3f}m "
                        f"dist={spec.dist:+.3f}m roll={spec.roll:+.1f}deg "
                        f"tilt=({spec.tilt_x:+.1f},{spec.tilt_y:+.1f})deg"
                    ),
                    pose=self.matrix_to_pose_stamped(desired_base_T_ee, self.base_frame, now_msg()),
                    base_T_ee=desired_base_T_ee,
                    base_T_cam=desired_base_T_cam,
                    prediction_note=reason,
                    spec=spec,
                    projected_u_px=float(metrics.get("u", float("nan"))),
                    projected_v_px=float(metrics.get("v", float("nan"))),
                    projected_margin_px=float(metrics.get("margin", float("nan"))),
                    projected_marker_px=float(metrics.get("marker_px", float("nan"))),
                    projected_center_error_px=float(metrics.get("center_error_px", float("nan"))),
                    projected_distance_m=float(metrics.get("distance", float("nan"))),
                    segment_count=max(1, int(segment_count)),
                )
            )
            if len(candidates) >= self.max_candidate_attempts:
                break
        return candidates

    def interpolated_transforms(
        self,
        start: TransformMatrix,
        goal: TransformMatrix,
    ) -> List[TransformMatrix]:
        start_t = np.array(start.translation, dtype=float)
        goal_t = np.array(goal.translation, dtype=float)
        distance = float(np.linalg.norm(goal_t - start_t))
        rot_delta = (start.rotation.inv() * goal.rotation).magnitude()
        steps = max(
            1,
            int(math.ceil(distance / max(self.segment_step_m, 1.0e-4))),
            int(math.ceil(math.degrees(rot_delta) / max(self.segment_step_deg, 0.1))),
        )
        if steps == 1:
            return [goal]
        key_rots = R.from_quat([start.rotation.as_quat(), goal.rotation.as_quat()])
        slerp = Slerp([0.0, 1.0], key_rots)
        result = []
        for idx in range(1, steps + 1):
            ratio = float(idx) / float(steps)
            trans = start_t + (goal_t - start_t) * ratio
            result.append(
                TransformMatrix(
                    rotation=slerp([ratio])[0],
                    translation=(float(trans[0]), float(trans[1]), float(trans[2])),
                )
            )
        return result
