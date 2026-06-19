"""Session checks: marker/camera/service consistency/preflight helpers.

Each function takes `session: CollectorExecutionSession` as first parameter.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from easy_handeye2_msgs.srv import RemoveSample, TakeSample

from .sample_types import (
    FAMILY_EXECUTION_ORDER,
    AcceptedSampleQuality,
    CandidateFamily,
)
from .vision import QUALITY_CAMERA_MODEL, QUALITY_SAMPLING, QUALITY_STARTUP


# ------------------------------------------------------------------
# Post-move recenter requirement
# ------------------------------------------------------------------

def post_move_recenter_requirement(session) -> Tuple[bool, str]:
    """Check if a post-move recenter is needed."""
    sampling_ok, sampling_note = session._image_marker_status(
        require_center=True, quality_level=QUALITY_SAMPLING,
    )
    obs = session.vision_gate.latest_successful_observation()
    info = session.vision_gate.camera_info_snapshot()
    if obs is None or not info.ready:
        return True, f"sampling_quality_failed_after_move: {sampling_note}"
    center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
    if not sampling_ok:
        return True, f"sampling_quality_failed_after_move: {sampling_note}"
    if center_error > 75.0:
        return True, f"recenter_needed_center_error: center_error={center_error:.1f}px > 75.0px; {sampling_note}"
    if obs.margin_px < 120.0:
        return True, f"recenter_needed_margin: margin={obs.margin_px:.1f}px < 120.0px; {sampling_note}"
    return False, f"post-move sampling quality already good: {sampling_note}"


# ------------------------------------------------------------------
# XY coverage candidate check
# ------------------------------------------------------------------

def is_xy_coverage_candidate(candidate) -> bool:
    """纯 XY 平移覆盖候选：仅 base_x/base_y 非零，无旋转、无 Z 偏移。"""
    spec = candidate.spec
    return (
        candidate.family == CandidateFamily.SPHERE_SHELL
        and (abs(spec.base_x) > 1.0e-6 or abs(spec.base_y) > 1.0e-6)
        and abs(spec.base_z) < 1.0e-6
        and abs(spec.pitch) < 1.0e-6
        and abs(spec.yaw) < 1.0e-6
        and abs(spec.roll) < 1.0e-6
    )


# ------------------------------------------------------------------
# Projection / marker helpers
# ------------------------------------------------------------------

def projection_metrics(session, marker_in_camera: np.ndarray):
    z = float(marker_in_camera[2])
    distance = float(np.linalg.norm(marker_in_camera))
    if z <= 1.0e-4:
        return False, f"marker is behind camera optical frame (z={z:.3f})"
    if distance < session.sampling_cfg.min_marker_distance or distance > session.sampling_cfg.max_marker_distance:
        return False, f"marker distance {distance:.3f}m outside range"
    info = session.vision_gate.camera_info_snapshot()
    if not info.ready:
        return True, {"u": float("nan"), "v": float("nan"), "margin": float("inf"),
                      "marker_px": float("inf"), "distance": distance,
                      "note": f"visible, distance={distance:.3f}m, no CameraInfo yet"}
    u = info.fx * float(marker_in_camera[0]) / z + info.cx
    v = info.fy * float(marker_in_camera[1]) / z + info.cy
    marker_px = min(info.fx, info.fy) * session.sampling_cfg.marker_size_m / z
    margin = min(u, v, info.width - u, info.height - v)
    center_error_px = math.hypot(u - info.cx, v - info.cy)
    return True, {"u": float(u), "v": float(v), "margin": float(margin),
                  "marker_px": float(marker_px), "center_error_px": float(center_error_px),
                  "distance": distance}


def check_projected_marker(session, marker_in_camera: np.ndarray) -> Tuple[bool, str]:
    metrics_ok, metrics = projection_metrics(session, marker_in_camera)
    if not metrics_ok:
        return False, str(metrics)
    if metrics["margin"] < session.sampling_cfg.min_image_margin_px:
        return False, (
            f"marker projection too close to image border "
            f"(u={metrics['u']:.1f}, v={metrics['v']:.1f}, margin={metrics['margin']:.1f}px)"
        )
    if metrics["marker_px"] < session.sampling_cfg.min_projected_marker_px:
        return False, f"marker projection too small ({metrics['marker_px']:.1f}px)"
    return True, (
        f"visible, distance={metrics['distance']:.3f}m, "
        f"u={metrics['u']:.1f}, v={metrics['v']:.1f}, "
        f"size={metrics['marker_px']:.1f}px, margin={metrics['margin']:.1f}px"
    )


def marker_status(session, quality_level: str = QUALITY_STARTUP) -> Tuple[bool, str]:
    image_ok, image_note = session._image_marker_status(
        require_center=False, quality_level=quality_level,
    )
    if session._cv_ready() or image_ok:
        return image_ok, image_note
    with session.node._marker_lock:
        pose = session.node._last_marker_pose
        receipt_time = session.node._last_marker_receipt_time
    if pose is None or receipt_time is None:
        return False, f"marker id {session.frames.marker_id} has not been observed"
    age = time.monotonic() - receipt_time
    if age > session.sampling_cfg.marker_recent_timeout:
        return False, f"marker observation is stale ({age:.2f}s)"
    p = pose.position
    distance = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
    if distance < session.sampling_cfg.min_marker_distance or distance > session.sampling_cfg.max_marker_distance:
        return False, f"marker distance {distance:.3f}m outside range"
    projected_ok, projected_note = check_projected_marker(
        session, np.array([p.x, p.y, p.z], dtype=float),
    )
    if not projected_ok:
        return False, projected_note
    if session.motion_cfg.require_marker_tf:
        if not session.tf_buffer.can_transform(
            session.frames.tracking_base_frame, session.frames.tracking_marker_frame,
            Time(), timeout=Duration(seconds=0.1),
        ):
            return False, (
                f"TF {session.frames.tracking_base_frame}->{session.frames.tracking_marker_frame} "
                f"not available"
            )
    return True, projected_note


def camera_model_metrics(session) -> Tuple[bool, str, Optional[dict]]:
    obs = session.vision_gate.latest_successful_observation()
    ok, note = session.vision_gate.observation_quality(
        obs, quality_level=QUALITY_CAMERA_MODEL, require_center=False,
    )
    if not ok:
        return False, f"image observation unavailable for camera model check: {note}", None
    try:
        cam_T_marker = session._lookup_tf(
            session.frames.tracking_base_frame, session.frames.tracking_marker_frame, timeout_sec=1.0,
        )
    except Exception as exc:
        return False, (
            f"cannot lookup {session.frames.tracking_base_frame}->{session.frames.tracking_marker_frame}: {exc}"
        ), None
    marker_in_camera = np.array(cam_T_marker.translation, dtype=float)
    metrics_ok, metrics = projection_metrics(session, marker_in_camera)
    if not metrics_ok:
        return False, f"TF projection invalid: {metrics}", None
    if math.isnan(metrics["u"]) or math.isnan(metrics["v"]):
        return False, "CameraInfo is not ready; cannot compare TF projection to image corners", None
    pixel_error = math.hypot(obs.center_px[0] - metrics["u"], obs.center_px[1] - metrics["v"])
    result = {
        "pixel_error_px": float(pixel_error),
        "image_center_px": (float(obs.center_px[0]), float(obs.center_px[1])),
        "tf_projection_px": (float(metrics["u"]), float(metrics["v"])),
    }
    if pixel_error > session.sampling_cfg.camera_model_max_pixel_error:
        return False, (
            f"camera model mismatch: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
            f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) "
            f"error={pixel_error:.1f}px > {session.sampling_cfg.camera_model_max_pixel_error:.1f}px"
        ), result
    return True, (
        f"camera model check ok: image_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
        f"tf_projection=({metrics['u']:.1f},{metrics['v']:.1f}) error={pixel_error:.1f}px; {note}"
    ), result


def check_marker_visible(session, timeout: Optional[float] = None) -> Tuple[bool, str]:
    timeout = session.sampling_cfg.marker_timeout if timeout is None else timeout
    t0 = time.monotonic()
    last_reason = "not checked"
    while time.monotonic() - t0 < timeout:
        if session.node._should_stop():
            return False, "stop requested"
        ok, reason = marker_status(session)
        if ok:
            return True, reason
        last_reason = reason
        time.sleep(0.05)
    return False, last_reason


def wait_for_stable_marker(session, min_receipt_time: float = 0.0, min_stamp_ns: int = 0) -> Tuple[bool, str]:
    t0 = time.monotonic()
    stable = 0
    last_receipt = None
    last_reason = "not checked"
    while time.monotonic() - t0 < session.sampling_cfg.visibility_stable_timeout:
        if session.node._should_stop():
            return False, "stop requested"
        stable_metrics, image_reason = session.vision_gate.stable_window_metrics(
            require_center=True,
            min_receipt_time=min_receipt_time,
            min_stamp_ns=min_stamp_ns,
        )
        if stable_metrics is not None:
            return True, stable_metrics.note
        if session._cv_ready():
            last_reason = image_reason
            time.sleep(0.05)
            continue
        ok, reason = marker_status(session)
        if not ok:
            reason = image_reason if session._cv_ready() else reason
        last_reason = reason
        with session.node._marker_lock:
            receipt = session.node._last_marker_receipt_time
        if ok and receipt is not None and receipt != last_receipt:
            stable += 1
            last_receipt = receipt
            if stable >= session.sampling_cfg.visibility_stable_frames:
                return True, f"stable {stable} frames: {reason}"
        elif not ok:
            stable = 0
        time.sleep(0.05)
    return False, f"marker not stable: {last_reason}"


# ------------------------------------------------------------------
# Service helpers
# ------------------------------------------------------------------

def get_sample_count(session) -> Optional[int]:
    if not session.node.get_samples_cli.wait_for_service(
        timeout_sec=session.sampling_cfg.get_samples_service_wait_timeout,
    ):
        session._logger().warn(f"service {session.frames.get_sample_list_service} not available")
        return None
    future = session.node.get_samples_cli.call_async(TakeSample.Request())
    deadline = time.monotonic() + session.sampling_cfg.get_samples_call_timeout
    while not session.node._should_stop() and not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not future.done() or future.result() is None:
        return None
    return len(getattr(future.result().samples, "samples", []))


def clear_remote_samples(session) -> bool:
    if not session.node.remove_sample_cli.wait_for_service(
        timeout_sec=session.sampling_cfg.remove_samples_service_wait_timeout,
    ):
        session._logger().warn(f"service {session.frames.remove_sample_service} not available")
        return False
    while not session.node._should_stop():
        count = get_sample_count(session)
        if count is None:
            return False
        if count == 0:
            return True
        future = session.node.remove_sample_cli.call_async(
            RemoveSample.Request(sample_index=int(count - 1)),
        )
        deadline = time.monotonic() + session.sampling_cfg.remove_samples_call_timeout
        while not session.node._should_stop() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False
        result = future.result()
        if result is None:
            return False
    return False


def take_sample(session) -> Tuple[bool, str]:
    """Take sample with remote consistency gate."""
    if not session.node.sample_cli.wait_for_service(
        timeout_sec=session.sampling_cfg.take_sample_service_wait_timeout,
    ):
        return False, f"service {session.frames.take_sample_service} not available"

    # Preflight: get remote current transforms.
    preflight = None
    if session.node.get_current_transforms_cli.wait_for_service(timeout_sec=1.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < session.sampling_cfg.sample_consistency_timeout:
            pf = session.node.get_current_transforms_cli.call_async(TakeSample.Request())
            dl = time.monotonic() + 1.0
            while not pf.done() and time.monotonic() < dl:
                time.sleep(0.02)
            if pf.done() and pf.result():
                s = getattr(pf.result(), "samples", None)
                if s and len(s.samples) == 1:
                    preflight = s.samples[0]
                    break
            time.sleep(0.1)

    local_robot = session._current_transform(session.frames.base_frame, session.frames.ee_frame)
    local_tracking = session._current_transform(
        session.frames.tracking_base_frame, session.frames.tracking_marker_frame,
    )
    if local_robot is None or local_tracking is None:
        return False, "cannot capture local TF for consistency check"

    if preflight is not None:
        r_ok, r_note = transform_consistency(
            preflight.robot, local_robot, "robot",
            session.sampling_cfg.sample_consistency_max_translation_m,
            session.sampling_cfg.sample_consistency_max_rotation_deg,
        )
        t_ok, t_note = transform_consistency(
            preflight.tracking, local_tracking, "tracking",
            session.sampling_cfg.sample_consistency_max_translation_m,
            session.sampling_cfg.sample_consistency_max_rotation_deg,
        )
        if not r_ok or not t_ok:
            session._logger().warn(f"Sample consistency FAIL: {r_note} {t_note}")
            return False, f"preflight_consistency_fail: {r_note}; {t_note}"

    count_before = get_sample_count(session)
    if count_before is None:
        return False, "cannot verify sample count before take_sample"
    future = session.node.sample_cli.call_async(TakeSample.Request())
    deadline = time.monotonic() + session.sampling_cfg.take_sample_call_timeout
    while not session.node._should_stop() and not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not future.done():
        return False, "take_sample timed out"
    result = future.result()
    if result is None:
        return False, "take_sample returned no response"
    response_count = len(getattr(result.samples, "samples", []))
    count_after = get_sample_count(session)
    if count_after is None or count_after != count_before + 1:
        return False, (
            f"sample count did not increase by 1 "
            f"(before={count_before}, after={count_after}, response={response_count})"
        )
    return True, f"samples={count_after} (before={count_before})"


def transform_consistency(remote_sample, local_matrix, label, max_dt, max_dr):
    """Compare a remote Sample transform to a local TransformMatrix."""
    rp = remote_sample.translation
    lp = local_matrix.translation
    dt = math.sqrt((rp.x - lp[0])**2 + (rp.y - lp[1])**2 + (rp.z - lp[2])**2)
    rq = remote_sample.rotation
    remote_r = R.from_quat([rq.x, rq.y, rq.z, rq.w])
    dr = math.degrees(float((remote_r.inv() * local_matrix.rotation).magnitude()))
    ok = dt <= max_dt and dr <= max_dr
    note = f"{label} dt={dt:.4f}/{max_dt:.4f}m dr={dr:.2f}/{max_dr:.2f}deg {'PASS' if ok else 'FAIL'}"
    return ok, note


def call_empty_service(session, client, request, service_name: str, timeout_sec: float = 8.0):
    if not client.wait_for_service(timeout_sec=session.sampling_cfg.empty_service_wait_timeout):
        return None, f"service {service_name} not available"
    future = client.call_async(request)
    deadline = time.monotonic() + timeout_sec
    while not session.node._should_stop() and not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not future.done():
        return None, f"{service_name} timed out"
    result = future.result()
    if result is None:
        return None, f"{service_name} returned no response"
    return result, ""


def remove_remote_sample(session, sample_index: int) -> Tuple[bool, str]:
    if not session.node.remove_sample_cli.wait_for_service(
        timeout_sec=session.sampling_cfg.remove_samples_service_wait_timeout,
    ):
        return False, f"service {session.frames.remove_sample_service} not available"
    count_before = get_sample_count(session)
    if count_before is None:
        return False, "cannot verify sample count before remove_sample"
    future = session.node.remove_sample_cli.call_async(
        RemoveSample.Request(sample_index=int(sample_index)),
    )
    deadline = time.monotonic() + session.sampling_cfg.remove_samples_call_timeout
    while not session.node._should_stop() and not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not future.done():
        return False, "remove_sample timed out"
    result = future.result()
    if result is None:
        return False, "remove_sample returned no response"
    count_after = get_sample_count(session)
    if count_after is None or count_after != count_before - 1:
        return False, (
            f"sample count did not decrease by 1 "
            f"(before={count_before}, after={count_after})"
        )
    return True, f"removed sample index {sample_index}"


def apply_remote_removals(session, remove_indices) -> Tuple[bool, str]:
    if not remove_indices:
        return True, "no remote removals needed"
    applied = []
    for sample_index in sorted((int(idx) for idx in remove_indices), reverse=True):
        sample_ok, sample_note = remove_remote_sample(session, sample_index)
        if not sample_ok:
            return False, f"failed to remove sample {sample_index}: {sample_note}"
        session.sample_manager.remove_accepted_sample(sample_index)
        applied.append(f"{sample_index}:{sample_note}")
    return True, "; ".join(applied)


# ------------------------------------------------------------------
# Candidate quality snapshot
# ------------------------------------------------------------------

def candidate_quality_snapshot(
    session, *, marker_note, model_note, stable_note,
    camera_model_metrics, stable_window_metrics,
):
    obs = session.vision_gate.latest_successful_observation()
    info = session.vision_gate.camera_info_snapshot()
    if obs is None or not info.ready:
        return AcceptedSampleQuality(
            center_error_px=float("inf"),
            margin_px=float("-inf"),
            marker_side_px=float("-inf"),
            distance_m=float("inf"),
            camera_model_error_px=float("inf"),
            center_std_px=float("inf"),
            depth_std_m=float("inf"),
            angle_std_deg=float("inf"),
            marker_note=marker_note,
            model_note=model_note,
            stable_note=stable_note,
        )
    center_error = math.hypot(obs.center_px[0] - info.cx, obs.center_px[1] - info.cy)
    distance_m = float(np.linalg.norm(np.array(obs.tvec, dtype=float)))
    return AcceptedSampleQuality(
        center_error_px=float(center_error),
        margin_px=float(obs.margin_px),
        marker_side_px=float(obs.side_px),
        distance_m=distance_m,
        camera_model_error_px=float(
            camera_model_metrics["pixel_error_px"]
            if camera_model_metrics is not None else float("inf")
        ),
        center_std_px=float(
            stable_window_metrics.center_std_px
            if stable_window_metrics is not None else float("inf")
        ),
        depth_std_m=float(
            stable_window_metrics.depth_std_m
            if stable_window_metrics is not None else float("inf")
        ),
        angle_std_deg=float(
            stable_window_metrics.angle_std_deg
            if stable_window_metrics is not None else float("inf")
        ),
        marker_note=marker_note,
        model_note=model_note,
        stable_note=stable_note,
    )


# ------------------------------------------------------------------
# Precision sample status
# ------------------------------------------------------------------

def precision_sample_status(
    session, candidate, *, quality, recenter_attempted,
    recenter_strict_converged, center_error_limit_px=None,
) -> Tuple[bool, str]:
    if not session.sampling_cfg.precision_gate_enabled:
        return True, "precision gate disabled"

    failures = []
    if (
        session.sampling_cfg.precision_reject_non_strict_recenter_non_anchor
        and candidate.family != CandidateFamily.SPHERE_ANCHOR
        and recenter_attempted
        and not recenter_strict_converged
    ):
        failures.append("non-strict recenter rejected for non-anchor family")
    center_limit_px = (
        center_error_limit_px
        if center_error_limit_px is not None
        else session.sampling_cfg.precision_max_center_error_px
    )
    if quality.center_error_px > center_limit_px:
        failures.append(f"center_error={quality.center_error_px:.1f}px > {center_limit_px:.1f}px")
    if quality.camera_model_error_px > session.sampling_cfg.precision_max_camera_model_error_px:
        failures.append(
            f"camera_model_error={quality.camera_model_error_px:.1f}px > "
            f"{session.sampling_cfg.precision_max_camera_model_error_px:.1f}px"
        )
    if quality.center_std_px > session.sampling_cfg.precision_max_center_std_px:
        failures.append(
            f"center_std={quality.center_std_px:.2f}px > "
            f"{session.sampling_cfg.precision_max_center_std_px:.2f}px"
        )
    if quality.depth_std_m > session.sampling_cfg.precision_max_depth_std_m:
        failures.append(
            f"depth_std={quality.depth_std_m:.4f}m > "
            f"{session.sampling_cfg.precision_max_depth_std_m:.4f}m"
        )
    if quality.angle_std_deg > session.sampling_cfg.precision_max_angle_std_deg:
        failures.append(
            f"angle_std={quality.angle_std_deg:.2f}deg > "
            f"{session.sampling_cfg.precision_max_angle_std_deg:.2f}deg"
        )
    if failures:
        return False, "precision gate FAIL: " + "; ".join(failures)
    return True, (
        "precision gate PASS: "
        f"center_error={quality.center_error_px:.1f}/{center_limit_px:.1f}px, "
        f"camera_model_error={quality.camera_model_error_px:.1f}px, "
        f"center_std={quality.center_std_px:.2f}px, "
        f"depth_std={quality.depth_std_m:.4f}m, "
        f"angle_std={quality.angle_std_deg:.2f}deg"
    )


# ------------------------------------------------------------------
# Gate deficit critical check
# ------------------------------------------------------------------

def is_gate_deficit_critical(candidate, source: str, deficits: dict) -> bool:
    """Check whether a candidate addresses an active gate deficit."""
    spec = candidate.spec
    if source == "sphere_height" and (deficits.get("z") or deficits.get("height")):
        return True
    if source == "sphere_shell":
        if deficits.get("xy") and (abs(spec.base_x) > 1.0e-6 or abs(spec.base_y) > 1.0e-6):
            return True
        if deficits.get("z") and abs(spec.base_z) > 1.0e-6:
            return True
    if source == "sphere_shell" and deficits.get("shell"):
        return True
    if source == "sphere_roll_coverage" and deficits.get("rot"):
        return True
    if source == "sphere_anchor" and (deficits.get("pitch") or deficits.get("yaw") or deficits.get("roll")):
        return True
    return False


# ------------------------------------------------------------------
# Spec family map builder
# ------------------------------------------------------------------

def build_spec_family_map(base_offsets: dict) -> dict:
    """Determine candidate source (family-name key) for each candidate."""
    spec_family_map = {}
    for fam_name in FAMILY_EXECUTION_ORDER:
        offsets = base_offsets.get(fam_name, [])
        for off in offsets:
            spec_family_map[off.label] = fam_name
    return spec_family_map
