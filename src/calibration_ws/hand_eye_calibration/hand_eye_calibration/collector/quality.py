"""Direct PnP observations and per-sample image/robot quality gates."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .model import AcceptedSampleQuality, TransformMatrix


@dataclass
class CameraInfoState:
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k: Tuple[float, ...] = ()
    d: Tuple[float, ...] = ()
    frame_id: str = ""

    @property
    def ready(self) -> bool:
        return self.width > 0 and self.height > 0 and self.fx > 0.0 and self.fy > 0.0


@dataclass
class ArucoObservation:
    receipt_time: float
    center_px: Tuple[float, float]
    corners_px: Tuple[Tuple[float, float], ...]
    side_px: float
    margin_px: float
    tvec: Tuple[float, float, float]
    rvec: Tuple[float, float, float]
    image_stamp_ns: int = 0
    pnp_ambiguous: bool = False
    ippe_absolute_gap_px: float = float("inf")
    ippe_error_ratio: float = float("inf")

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.tvec))


@dataclass
class ImageFrameStatus:
    receipt_time: float
    detected: bool
    observation: Optional[ArucoObservation] = None
    reason: str = ""
    image_stamp_ns: int = 0


@dataclass
class StableWindowMetrics:
    latest_observation: ArucoObservation
    center_std_px: float
    depth_std_m: float
    note: str
    observations: Tuple[ArucoObservation, ...] = ()
    translation_mad_m: float = float("inf")
    rotation_mad_deg: float = float("inf")
    robot_translation_drift_m: float = float("inf")
    robot_rotation_drift_deg: float = float("inf")
    non_ambiguous_frame_count: int = 0


class VisionQualityGate:
    def __init__(self, *, marker_recent_timeout: float, min_marker_distance: float,
                 max_marker_distance: float, min_visible_border_px: float,
                 min_marker_side_px: float, stable_frame_count: int,
                 stable_min_valid_frames: int, max_pnp_translation_mad_m: float,
                 max_pnp_rotation_mad_deg: float, logger_warn: Callable[[str], None]):
        self.marker_recent_timeout = float(marker_recent_timeout)
        self.min_marker_distance = float(min_marker_distance)
        self.max_marker_distance = float(max_marker_distance)
        self.min_visible_border_px = float(min_visible_border_px)
        self.min_marker_side_px = float(min_marker_side_px)
        self.stable_frame_count = max(1, int(stable_frame_count))
        self.stable_min_valid_frames = max(1, int(stable_min_valid_frames))
        self.max_pnp_translation_mad_m = float(max_pnp_translation_mad_m)
        self.max_pnp_rotation_mad_deg = float(max_pnp_rotation_mad_deg)
        self._logger_warn = logger_warn
        self._camera_info = CameraInfoState()
        self._camera_lock = threading.Lock()
        self._lock = threading.Lock()
        self._last_observation: Optional[ArucoObservation] = None
        self._frames = deque(maxlen=80)

    def update_camera_info(self, info: CameraInfoState) -> None:
        with self._camera_lock:
            self._camera_info = info

    def camera_info_snapshot(self) -> CameraInfoState:
        with self._camera_lock:
            return CameraInfoState(**self._camera_info.__dict__)

    def latest_successful_observation(self) -> Optional[ArucoObservation]:
        with self._lock:
            return self._last_observation

    def last_failed_frame(self) -> Optional[ImageFrameStatus]:
        with self._lock:
            return next((frame for frame in reversed(self._frames) if not frame.detected), None)

    def record_frame_status(self, *, detected: bool, observation: Optional[ArucoObservation] = None,
                            reason: str = "", image_stamp_ns: int = 0,
                            receipt_time: Optional[float] = None) -> None:
        status = ImageFrameStatus(
            receipt_time=time.monotonic() if receipt_time is None else float(receipt_time),
            detected=detected,
            observation=observation,
            reason=reason,
            image_stamp_ns=int(image_stamp_ns),
        )
        with self._lock:
            self._frames.append(status)
            if detected and observation is not None:
                self._last_observation = observation

    def log_aruco_exception(self, context: str, exc: Exception) -> None:
        self._logger_warn(f"ArUco worker exception in {context}: {type(exc).__name__}: {exc}")

    def observation_quality(self, obs: Optional[ArucoObservation]) -> Tuple[bool, str]:
        if obs is None:
            failed = self.last_failed_frame()
            return False, f"no marker observation ({failed.reason if failed else 'no detected image'})"
        age = time.monotonic() - obs.receipt_time
        if age > self.marker_recent_timeout:
            return False, f"marker observation is stale ({age:.2f}s)"
        if len(obs.corners_px) != 4:
            return False, f"marker has {len(obs.corners_px)} corners, need four"
        if not self.min_marker_distance <= obs.distance_m <= self.max_marker_distance:
            return False, f"marker distance {obs.distance_m:.3f}m outside range"
        if obs.margin_px < self.min_visible_border_px:
            return False, f"marker is clipped (margin={obs.margin_px:.1f}px)"
        if obs.side_px < self.min_marker_side_px:
            return False, f"marker side {obs.side_px:.1f}px < {self.min_marker_side_px:.1f}px"
        if not self.camera_info_snapshot().ready:
            return False, "CameraInfo is not ready"
        return True, f"marker ok margin={obs.margin_px:.1f}px side={obs.side_px:.1f}px z={obs.distance_m:.3f}m"

    def post_motion_failure(self, min_receipt_time: float, min_stamp_ns: int, stable_note: str):
        with self._lock:
            fresh_receipts = [frame for frame in self._frames if frame.receipt_time > min_receipt_time]
        if not fresh_receipts:
            return "SESSION_FATAL", "no fresh image frame arrived after motion"
        fresh = [frame for frame in fresh_receipts if frame.image_stamp_ns > min_stamp_ns]
        if not fresh:
            return "SESSION_FATAL", "image timestamp did not advance after motion"
        mismatch = next((frame for frame in fresh if "CameraInfo/image" in frame.reason), None)
        if mismatch is not None:
            return "SESSION_FATAL", mismatch.reason
        last = fresh[-1]
        return "RETRYABLE", f"fresh frames but no valid stable marker: {stable_note}; last={last.reason or 'detected frame'}"

    def stable_window_metrics(self, *, min_receipt_time: float = 0.0, min_stamp_ns: int = 0):
        with self._lock:
            frames = list(self._frames)[-self.stable_frame_count:]
        if len(frames) < self.stable_frame_count:
            return None, f"need {self.stable_frame_count} image frames, have {len(frames)}"
        frames = [frame for frame in frames if frame.receipt_time > min_receipt_time and frame.image_stamp_ns > min_stamp_ns]
        if len(frames) < self.stable_frame_count:
            return None, f"need {self.stable_frame_count} fresh image frames after motion"
        observations = [frame.observation for frame in frames if frame.detected and frame.observation is not None]
        if len(observations) < self.stable_min_valid_frames:
            return None, f"need {self.stable_min_valid_frames} valid PnP frames, have {len(observations)}"
        if any(not self.observation_quality(obs)[0] for obs in observations):
            return None, "stable window contains an invalid marker observation"
        stamps = np.asarray([obs.image_stamp_ns for obs in observations], dtype=np.int64)
        if np.any(stamps <= 0) or np.any(np.diff(stamps) <= 0):
            return None, "stable window image timestamps are not strictly increasing"
        if len(stamps) > 2 and np.max(np.diff(stamps)) > 2.5 * np.median(np.diff(stamps)):
            return None, "stable window contains a timestamp gap"
        translations = np.asarray([obs.tvec for obs in observations], dtype=float)
        median_translation = np.median(translations, axis=0)
        translation_mad = float(np.median(np.linalg.norm(translations - median_translation, axis=1)))
        rotations = R.from_rotvec(np.asarray([obs.rvec for obs in observations], dtype=float))
        median_rotation = rotations.mean()
        rotation_deviation = np.asarray([math.degrees((median_rotation.inv() * rotation).magnitude()) for rotation in rotations])
        rotation_mad = float(np.median(rotation_deviation))
        if translation_mad > self.max_pnp_translation_mad_m:
            return None, f"PnP translation MAD {translation_mad:.6f}m exceeds limit"
        if rotation_mad > self.max_pnp_rotation_mad_deg:
            return None, f"PnP rotation MAD {rotation_mad:.3f}deg exceeds limit"
        medoid = int(np.argmin(np.linalg.norm(translations - median_translation, axis=1) + rotation_deviation))
        centers = np.asarray([obs.center_px for obs in observations])
        depths = np.linalg.norm(translations, axis=1)
        note = f"stable PnP {len(observations)}/{len(frames)} frames, translation_mad={translation_mad:.6f}m rotation_mad={rotation_mad:.3f}deg"
        return StableWindowMetrics(
            latest_observation=observations[medoid],
            center_std_px=float(np.max(np.std(centers, axis=0))),
            depth_std_m=float(np.std(depths)),
            note=note,
            observations=tuple(observations),
            translation_mad_m=translation_mad,
            rotation_mad_deg=rotation_mad,
            non_ambiguous_frame_count=sum(not bool(obs.pnp_ambiguous) for obs in observations),
        ), note


def camera_model_metrics(session, observation=None, *, reject_pnp_ambiguity=True, stable_metrics=None):
    observation = observation or session.vision_gate.latest_successful_observation()
    ok, note = session.vision_gate.observation_quality(observation)
    if not ok:
        return False, note, None
    if float(observation.tvec[2]) <= 0.0:
        return False, "PnP optical depth is non-positive", None
    if reject_pnp_ambiguity and bool(observation.pnp_ambiguous):
        clear_frames = int(getattr(stable_metrics, "non_ambiguous_frame_count", 0))
        required = int(getattr(session.sampling_cfg, "ippe_min_non_ambiguous_frames", 3))
        if clear_frames < required:
            return False, f"IPPE dual solution rejected: non-ambiguous frames {clear_frames} < {required}", None
    try:
        import cv2
        info = session.vision_gate.camera_info_snapshot()
        half = session.sampling_cfg.marker_size_m * 0.5
        object_points = np.asarray(((-half, half, 0.0), (half, half, 0.0), (half, -half, 0.0), (-half, -half, 0.0)), dtype=np.float32)
        projected, _ = cv2.projectPoints(object_points, np.asarray(observation.rvec, dtype=float), np.asarray(observation.tvec, dtype=float), np.asarray(info.k, dtype=float).reshape(3, 3), np.asarray(info.d, dtype=float) if info.d else np.zeros(5))
        errors = np.linalg.norm(projected.reshape(4, 2) - np.asarray(observation.corners_px, dtype=float), axis=1)
    except Exception as exc:
        return False, f"direct PnP reprojection failed: {exc}", None
    rms, maximum = float(np.sqrt(np.mean(errors ** 2))), float(np.max(errors))
    metrics = {"pixel_error_px": rms, "max_corner_error_px": maximum}
    if rms > session.sampling_cfg.pnp_reprojection_rms_max_px:
        return False, f"PnP RMS {rms:.2f}px exceeds limit", metrics
    if maximum > session.sampling_cfg.pnp_reprojection_max_corner_px:
        return False, f"PnP maximum corner {maximum:.2f}px exceeds limit", metrics
    suffix = f"; IPPE medoid allowed with {int(getattr(stable_metrics, 'non_ambiguous_frame_count', 0))} clear frames" if observation.pnp_ambiguous else ""
    return True, f"PnP RMS={rms:.2f}px max={maximum:.2f}px{suffix}", metrics


def wait_for_stable_marker(session, *, min_receipt_time=0.0, min_stamp_ns=0):
    deadline = time.monotonic() + session.sampling_cfg.stable_observation_timeout
    note = "no stable window"
    while time.monotonic() < deadline:
        if session.node._should_stop():
            return None, "stop requested"
        metrics, note = session.vision_gate.stable_window_metrics(min_receipt_time=min_receipt_time, min_stamp_ns=min_stamp_ns)
        if metrics is not None:
            return metrics, note
        time.sleep(0.04)
    return None, note


def robot_static_metrics(session, stable_metrics):
    poses = []
    for observation in stable_metrics.observations:
        if observation.image_stamp_ns <= 0:
            return None, "stable PnP window has an unstamped image"
        try:
            poses.append(session._lookup_tf_at_ns(session.frames.base_frame, session.frames.ee_frame, observation.image_stamp_ns))
        except Exception as exc:
            return None, f"tf_at_image_stamp_failed: {exc}"
    translations = np.asarray([pose.translation for pose in poses], dtype=float)
    translation_drift = float(np.max(np.linalg.norm(translations - np.median(translations, axis=0), axis=1)))
    rotations = [pose.rotation for pose in poses]
    median_rotation = R.from_matrix(np.asarray([rotation.as_matrix() for rotation in rotations])).mean()
    rotation_drift = float(max(math.degrees((median_rotation.inv() * rotation).magnitude()) for rotation in rotations))
    if translation_drift > session.sampling_cfg.max_ee_translation_drift_m:
        return None, f"EE translation drift {translation_drift:.6f}m exceeds limit"
    if rotation_drift > session.sampling_cfg.max_ee_rotation_drift_deg:
        return None, f"EE rotation drift {rotation_drift:.3f}deg exceeds limit"
    joint_state = getattr(getattr(session.motion, "arm", None), "joint_state", None)
    velocities = getattr(joint_state, "velocity", ()) if joint_state is not None else ()
    if velocities and float(np.max(np.abs(np.asarray(velocities, dtype=float)))) > session.sampling_cfg.max_joint_velocity_rad_s:
        return None, "joint velocity exceeds stationary limit"
    return replace(stable_metrics, robot_translation_drift_m=translation_drift, robot_rotation_drift_deg=rotation_drift), f"robot static t={translation_drift:.6f}m r={rotation_drift:.3f}deg"


def capture_direct_sample(session, stable_metrics):
    observation = stable_metrics.latest_observation
    if observation is None or observation.image_stamp_ns <= 0:
        return None, None, "no stamped stable PnP observation"
    try:
        robot = session._lookup_tf_at_ns(session.frames.base_frame, session.frames.ee_frame, observation.image_stamp_ns)
    except Exception as exc:
        return None, None, f"tf_at_image_stamp_failed: {exc}"
    tracking = TransformMatrix(R.from_rotvec(np.asarray(observation.rvec, dtype=float)), tuple(float(value) for value in observation.tvec))
    return robot, tracking, f"direct PnP/TF stamp={observation.image_stamp_ns}"


def candidate_quality_snapshot(session, observation, stable_metrics, camera_metrics, marker_note, model_note):
    info = session.vision_gate.camera_info_snapshot()
    return AcceptedSampleQuality(
        center_error_px=float(math.hypot(observation.center_px[0] - info.cx, observation.center_px[1] - info.cy)),
        margin_px=float(observation.margin_px),
        marker_side_px=float(observation.side_px),
        distance_m=float(observation.distance_m),
        camera_model_error_px=float(camera_metrics["pixel_error_px"]),
        center_std_px=float(stable_metrics.center_std_px),
        depth_std_m=float(stable_metrics.depth_std_m),
        marker_note=marker_note,
        model_note=model_note,
        stable_note=stable_metrics.note,
        translation_mad_m=float(stable_metrics.translation_mad_m),
        rotation_mad_deg=float(stable_metrics.rotation_mad_deg),
        robot_translation_drift_m=float(stable_metrics.robot_translation_drift_m),
        robot_rotation_drift_deg=float(stable_metrics.robot_rotation_drift_deg),
        ippe_absolute_gap_px=float(observation.ippe_absolute_gap_px),
        ippe_error_ratio=float(observation.ippe_error_ratio),
        ippe_non_ambiguous_frames=int(stable_metrics.non_ambiguous_frame_count),
    )
