from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


QUALITY_STARTUP = "startup_visibility"
QUALITY_CAMERA_MODEL = "camera_model_check"
QUALITY_SAMPLING = "sampling_quality"


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

    @property
    def ready(self) -> bool:
        return self.width > 0 and self.height > 0 and self.fx > 0.0 and self.fy > 0.0


@dataclass
class ArucoObservation:
    receipt_time: float
    center_px: Tuple[float, float]
    corners_px: Tuple[Tuple[float, float], ...]
    side_px: float
    area_px2: float
    margin_px: float
    tvec: Tuple[float, float, float]
    rvec: Tuple[float, float, float]
    image_stamp_ns: int = 0

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(np.array(self.tvec, dtype=float)))

    @property
    def angle_deg(self) -> float:
        return math.degrees(float(np.linalg.norm(np.array(self.rvec, dtype=float))))


@dataclass
class ImageFrameStatus:
    receipt_time: float
    detected: bool
    observation: Optional[ArucoObservation] = None
    reason: str = ""
    image_stamp_ns: int = 0


class VisionQualityGate:
    def __init__(
        self,
        *,
        marker_recent_timeout: float,
        min_marker_distance: float,
        max_marker_distance: float,
        startup_min_corner_margin_px: float,
        min_corner_margin_px: float,
        min_marker_side_px: float,
        max_center_error_px: float,
        stable_frame_count: int,
        max_center_std_px: float,
        max_depth_std_m: float,
        max_angle_std_deg: float,
        logger_warn: Callable[[str], None],
    ):
        self.marker_recent_timeout = float(marker_recent_timeout)
        self.min_marker_distance = float(min_marker_distance)
        self.max_marker_distance = float(max_marker_distance)
        self.startup_min_corner_margin_px = float(startup_min_corner_margin_px)
        self.min_corner_margin_px = float(min_corner_margin_px)
        self.min_marker_side_px = float(min_marker_side_px)
        self.max_center_error_px = float(max_center_error_px)
        self.stable_frame_count = max(1, int(stable_frame_count))
        self.max_center_std_px = float(max_center_std_px)
        self.max_depth_std_m = float(max_depth_std_m)
        self.max_angle_std_deg = float(max_angle_std_deg)
        self._logger_warn = logger_warn

        self._camera_info = CameraInfoState()
        self._camera_info_lock = threading.Lock()
        self._observation_lock = threading.Lock()
        self._last_observation: Optional[ArucoObservation] = None
        self._observation_history = deque(maxlen=40)
        self._frame_history = deque(maxlen=80)
        self._aruco_exception_counts = {}
        self._last_aruco_exception_log = 0.0

    def update_camera_info(self, info: CameraInfoState):
        with self._camera_info_lock:
            self._camera_info = info

    def camera_info_snapshot(self) -> CameraInfoState:
        with self._camera_info_lock:
            return CameraInfoState(
                width=self._camera_info.width,
                height=self._camera_info.height,
                fx=self._camera_info.fx,
                fy=self._camera_info.fy,
                cx=self._camera_info.cx,
                cy=self._camera_info.cy,
                k=tuple(self._camera_info.k),
                d=tuple(self._camera_info.d),
            )

    def latest_observation(self) -> Optional[ArucoObservation]:
        with self._observation_lock:
            return self._last_observation

    def latest_successful_observation(self) -> Optional[ArucoObservation]:
        return self.latest_observation()

    def latest_frame(self) -> Optional[ImageFrameStatus]:
        with self._observation_lock:
            if not self._frame_history:
                return None
            return self._frame_history[-1]

    def last_failed_frame(self) -> Optional[ImageFrameStatus]:
        with self._observation_lock:
            for frame in reversed(self._frame_history):
                if not frame.detected or frame.observation is None:
                    return frame
        return None

    def record_frame_status(
        self,
        *,
        detected: bool,
        observation: Optional[ArucoObservation] = None,
        reason: str = "",
        image_stamp_ns: int = 0,
    ):
        status = ImageFrameStatus(
            receipt_time=time.monotonic(),
            detected=detected,
            observation=observation,
            reason=reason,
            image_stamp_ns=int(image_stamp_ns),
        )
        with self._observation_lock:
            self._frame_history.append(status)
            if detected and observation is not None:
                self._last_observation = observation
                self._observation_history.append(observation)

    def log_aruco_exception(self, context: str, exc: Exception):
        exc_name = type(exc).__name__
        key = f"{context}:{exc_name}"
        self._aruco_exception_counts[key] = self._aruco_exception_counts.get(key, 0) + 1
        now = time.monotonic()
        if now - self._last_aruco_exception_log > 5.0:
            self._last_aruco_exception_log = now
            count = self._aruco_exception_counts[key]
            self._logger_warn(
                f"ArUco worker exception in {context}: {exc_name}: {exc} "
                f"(count={count}, throttled)"
            )

    def observation_quality(
        self,
        obs: Optional[ArucoObservation],
        *,
        quality_level: str,
        require_center: bool,
    ) -> Tuple[bool, str]:
        if obs is None:
            failed = self.last_failed_frame()
            if failed is not None and failed.reason:
                return False, f"{quality_level}: no successful image marker observation ({failed.reason})"
            return False, f"{quality_level}: image marker has not been observed"
        age = time.monotonic() - obs.receipt_time
        if age > self.marker_recent_timeout:
            latest_frame = self.latest_frame()
            if (
                latest_frame is not None
                and latest_frame.receipt_time > obs.receipt_time
                and (not latest_frame.detected or latest_frame.observation is None)
            ):
                reason = latest_frame.reason or "marker detection failed on fresh image frame"
                return False, f"{quality_level}: fresh image frames arrived but no marker detected ({reason})"
            return False, f"{quality_level}: image marker observation is stale ({age:.2f}s)"
        distance = obs.distance_m
        if distance < self.min_marker_distance or distance > self.max_marker_distance:
            return False, f"{quality_level}: image marker distance {distance:.3f}m outside range"

        if quality_level in (QUALITY_STARTUP, QUALITY_CAMERA_MODEL):
            min_corner_margin = self.startup_min_corner_margin_px
        else:
            min_corner_margin = self.min_corner_margin_px

        if min_corner_margin > 0.0 and obs.margin_px < min_corner_margin:
            return (
                False,
                f"{quality_level}: corner margin too small "
                f"({obs.margin_px:.1f}px < {min_corner_margin:.1f}px)",
            )
        if obs.side_px < self.min_marker_side_px:
            return (
                False,
                f"{quality_level}: marker side too small "
                f"({obs.side_px:.1f}px < {self.min_marker_side_px:.1f}px)",
            )

        info = self.camera_info_snapshot()
        if not info.ready:
            return False, f"{quality_level}: CameraInfo is not ready"

        center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
        if require_center and center_error > self.max_center_error_px:
            return (
                False,
                f"{quality_level}: marker center error too large "
                f"({center_error:.1f}px > {self.max_center_error_px:.1f}px)",
            )
        return (
            True,
            f"{quality_level}: image marker ok "
            f"center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
            f"err={center_error:.1f}px "
            f"margin={obs.margin_px:.1f}/{min_corner_margin:.1f}px "
            f"side={obs.side_px:.1f}/{self.min_marker_side_px:.1f}px "
            f"z={distance:.3f}m",
        )

    def image_marker_status(
        self,
        *,
        require_center: bool = False,
        quality_level: str = QUALITY_SAMPLING,
    ) -> Tuple[bool, str]:
        return self.observation_quality(
            self.latest_observation(),
            quality_level=quality_level,
            require_center=require_center,
        )

    def wait_for_new_frame(
        self,
        *,
        min_receipt_time: float,
        min_stamp_ns: int,
        timeout_sec: float,
        should_stop: Callable[[], bool],
    ) -> Tuple[bool, str]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            if should_stop():
                return False, "stop requested"
            frame = self.latest_frame()
            if frame is not None:
                if frame.receipt_time > min_receipt_time:
                    if min_stamp_ns <= 0 or frame.image_stamp_ns > min_stamp_ns:
                        return True, (
                            f"fresh image frame received: receipt={frame.receipt_time:.3f}, "
                            f"stamp_ns={frame.image_stamp_ns}"
                        )
            time.sleep(0.02)
        return False, "no fresh image frame arrived after motion"

    def wait_for_fresh_successful_observation(
        self,
        *,
        min_receipt_time: float,
        min_stamp_ns: int,
        timeout_sec: float,
        should_stop: Callable[[], bool],
    ) -> Tuple[bool, str]:
        t0 = time.monotonic()
        saw_fresh_frame = False
        last_failure_reason = ""
        while time.monotonic() - t0 < timeout_sec:
            if should_stop():
                return False, "stop requested"
            frame = self.latest_frame()
            if frame is not None and frame.receipt_time > min_receipt_time:
                if min_stamp_ns <= 0 or frame.image_stamp_ns > min_stamp_ns:
                    saw_fresh_frame = True
                    if frame.detected and frame.observation is not None:
                        return True, (
                            "fresh successful marker observation received: "
                            f"receipt={frame.receipt_time:.3f}, stamp_ns={frame.image_stamp_ns}"
                        )
                    last_failure_reason = frame.reason or "marker detection failed on fresh frame"
            time.sleep(0.02)
        if saw_fresh_frame:
            return False, f"fresh image frames arrived but no marker detected ({last_failure_reason or 'unknown reason'})"
        return False, "no fresh image frame arrived after motion"

    def stable_image_marker_status(
        self,
        *,
        require_center: bool,
        min_receipt_time: float = 0.0,
        min_stamp_ns: int = 0,
    ) -> Tuple[bool, str]:
        with self._observation_lock:
            recent_frames = list(self._frame_history)[-self.stable_frame_count :]
        if len(recent_frames) < self.stable_frame_count:
            return False, f"need {self.stable_frame_count} image frames, have {len(recent_frames)}"

        recent_frames = [
            frame for frame in recent_frames
            if frame.receipt_time >= min_receipt_time and frame.image_stamp_ns >= min_stamp_ns
        ]
        if len(recent_frames) < self.stable_frame_count:
            return (
                False,
                f"need {self.stable_frame_count} fresh image frames after motion, have {len(recent_frames)}"
            )

        failed = [frame for frame in recent_frames if not frame.detected or frame.observation is None]
        if failed:
            last_failed = failed[-1]
            return (
                False,
                "stable image window is not continuous: "
                f"{len(failed)}/{len(recent_frames)} recent frames failed "
                f"({last_failed.reason or 'unknown reason'})",
            )
        recent = [frame.observation for frame in recent_frames if frame.observation is not None]
        now = time.monotonic()
        if any(now - obs.receipt_time > self.marker_recent_timeout for obs in recent):
            return False, "stable image window contains stale marker frames"

        for obs in recent:
            ok, reason = self.observation_quality(
                obs,
                quality_level=QUALITY_SAMPLING,
                require_center=require_center,
            )
            if not ok:
                return False, reason

        centers = np.array([obs.center_px for obs in recent], dtype=float)
        depths = np.array([obs.distance_m for obs in recent], dtype=float)
        angles = np.array([obs.angle_deg for obs in recent], dtype=float)
        center_std = float(np.max(np.std(centers, axis=0)))
        depth_std = float(np.std(depths))
        angle_std = float(np.std(angles))
        if center_std > self.max_center_std_px:
            return False, f"center jitter too high ({center_std:.2f}px)"
        if depth_std > self.max_depth_std_m:
            return False, f"depth jitter too high ({depth_std:.4f}m)"
        if angle_std > self.max_angle_std_deg:
            return False, f"angle jitter too high ({angle_std:.2f}deg)"
        latest = recent[-1]
        ok, note = self.observation_quality(
            latest,
            quality_level=QUALITY_SAMPLING,
            require_center=require_center,
        )
        if not ok:
            return False, note
        return (
            True,
            f"stable image marker {len(recent)} frames: {note}, "
            f"std_center={center_std:.2f}px std_depth={depth_std:.4f}m "
            f"std_angle={angle_std:.2f}deg",
        )
