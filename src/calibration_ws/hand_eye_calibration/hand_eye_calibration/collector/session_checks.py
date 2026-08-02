"""Session checks: direct marker/camera quality and preflight helpers.

本模块提供采集会话期间所需的各类检查辅助函数：
- 移动后的重新居中需求判断
- XY 覆盖候选识别
- 标记在图像上的投影计算与可见性检查
- 相机模型一致性验证（图像中心与 TF 投影对比）
- 标记稳定性等待与直接 PnP 采样
- 精度门控与赤字严重性判断
- 候选规范到家族名称的映射构建

每个函数的第一个参数均为 `session: CollectorExecutionSession`，用于访问配置、状态和服务。
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .sample_types import (
    FAMILY_EXECUTION_ORDER,
    AcceptedSampleQuality,
    CandidateFamily,
)
from .vision import QUALITY_CAMERA_MODEL, QUALITY_SAMPLING, QUALITY_STARTUP


# ------------------------------------------------------------------
# 移动后的重新居中需求检查
# ------------------------------------------------------------------

def post_move_recenter_requirement(session) -> Tuple[bool, str]:
    """
    判断机械臂移动到候选位姿后是否需要执行重新居中（recenter）。
    需要重新居中的条件：
    1. 当前图像采样质量不满足要求（通常要求标记位于图像中心附近且大小/边缘合适）
    2. 标记中心误差超过 75 像素
    3. 标记与图像边界的距离小于 120 像素
    返回 (是否需要重新居中, 原因描述)。
    """
    # 以 SAMPLING 质量级别检查图像标记状态，要求中心附近
    sampling_ok, sampling_note = session._image_marker_status(
        require_center=True, quality_level=QUALITY_SAMPLING,
    )
    obs = session.vision_gate.latest_successful_observation()
    info = session.vision_gate.camera_info_snapshot()
    if obs is None or not info.ready:
        # 没有有效观测或相机内参未就绪，必须尝试重新居中
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
# XY 覆盖候选识别
# ------------------------------------------------------------------

def is_xy_coverage_candidate(candidate) -> bool:
    """
    判断一个候选位姿是否属于纯 XY 平移覆盖候选。
    条件：
    - 家族为 SPHERE_SHELL
    - 仅 base_x 或 base_y 非零
    - base_z、pitch、yaw、roll 均接近零（无旋转、无 Z 偏移）
    """
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
# 投影与标记可见性辅助函数
# ------------------------------------------------------------------

def projection_metrics(session, marker_in_camera: np.ndarray):
    """
    计算标记在相机坐标系下的投影度量。
    输入：marker_in_camera 为标记在相机光心坐标系中的三维坐标 (x,y,z)
    返回：(是否有效, 度量字典)
    度量字典包含像素坐标 u,v、边缘距离 margin、像素边长 marker_px、实际距离 distance 等。
    若 z 值过小或超出距离范围则视为无效。
    """
    z = float(marker_in_camera[2])
    distance = float(np.linalg.norm(marker_in_camera))
    if z <= 1.0e-4:
        return False, f"marker is behind camera optical frame (z={z:.3f})"
    if distance < session.sampling_cfg.min_marker_distance or distance > session.sampling_cfg.max_marker_distance:
        return False, f"marker distance {distance:.3f}m outside range"
    info = session.vision_gate.camera_info_snapshot()
    if not info.ready:
        # 没有相机内参时仍认为可见，但无法计算像素坐标
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
    """
    检查投影后的标记是否满足可视性要求（边缘距离和最小像素尺寸）。
    依赖 projection_metrics 计算，然后在有相机内参时检查 margin 和 marker_px。
    """
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
    """
    获取直接图像测量链的当前标记状态。
    """
    image_ok, image_note = session._image_marker_status(
        require_center=False, quality_level=quality_level,
    )
    return image_ok, image_note


def camera_model_metrics(session) -> Tuple[bool, str, Optional[dict]]:
    """
    计算直接 PnP 的角点重投影误差。该检查不再依赖另一个 ArUco 节点或其 TF。
    返回 (是否通过, 描述, 包含 pixel_error_px 等信息的字典或 None)
    """
    obs = session.vision_gate.latest_successful_observation()
    ok, note = session.vision_gate.observation_quality(
        obs, quality_level=QUALITY_CAMERA_MODEL, require_center=False,
    )
    if not ok:
        return False, f"image observation unavailable for camera model check: {note}", None
    try:
        import cv2
        half = session.sampling_cfg.marker_size_m * 0.5
        object_points = np.asarray(
            ((-half, half, 0.0), (half, half, 0.0),
             (half, -half, 0.0), (-half, -half, 0.0)), dtype=np.float32,
        )
        info = session.vision_gate.camera_info_snapshot()
        projected, _ = cv2.projectPoints(
            object_points, np.asarray(obs.rvec, dtype=float), np.asarray(obs.tvec, dtype=float),
            np.asarray(info.k, dtype=float).reshape(3, 3),
            np.asarray(info.d, dtype=float) if info.d else np.zeros(5, dtype=float),
        )
    except Exception as exc:
        return False, f"direct PnP reprojection failed: {exc}", None
    pixel_error = float(np.sqrt(np.mean(np.sum((
        projected.reshape(4, 2) - np.asarray(obs.corners_px, dtype=float)
    ) ** 2, axis=1))))
    result = {
        "pixel_error_px": float(pixel_error),
        "image_center_px": (float(obs.center_px[0]), float(obs.center_px[1])),
        "direct_pnp": True,
    }
    if pixel_error > session.sampling_cfg.camera_model_max_pixel_error:
        return False, (
            f"direct PnP reprojection error={pixel_error:.2f}px > "
            f"{session.sampling_cfg.camera_model_max_pixel_error:.2f}px"
        ), result
    return True, (
        f"direct PnP reprojection={pixel_error:.2f}px; {note}"
    ), result


def capture_direct_sample(session, stable_metrics):
    """Capture robot TF and PnP at the same image timestamp without easy_handeye2."""
    observation = session.node.refine_stable_observation(stable_metrics)
    if observation is None or observation.image_stamp_ns <= 0:
        return None, None, "direct sample has no stamped PnP observation"
    try:
        robot = session._lookup_tf_at_ns(
            session.frames.base_frame, session.frames.ee_frame, observation.image_stamp_ns,
        )
    except Exception as exc:
        return None, None, f"cannot capture robot TF at image stamp: {exc}"
    tracking = session.geometry.from_matrix(np.eye(4))
    tracking.rotation = R.from_rotvec(np.asarray(observation.rvec, dtype=float))
    tracking.translation = tuple(float(value) for value in observation.tvec)
    return robot, tracking, f"direct_pnp stamp_ns={observation.image_stamp_ns}"


def check_marker_visible(session, timeout: Optional[float] = None) -> Tuple[bool, str]:
    """
    在指定超时内循环检查标记是否可见（调用 marker_status）。
    一旦可见立即返回 True，超时返回 False。
    """
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
    """
    等待标记观测稳定。使用视觉门控的稳定窗口度量，或在无图像检测时连续观察标记话题更新。
    返回 (稳定是否达成, 描述)。
    """
    t0 = time.monotonic()
    stable = 0
    last_receipt = None
    last_reason = "not checked"
    while time.monotonic() - t0 < session.sampling_cfg.visibility_stable_timeout:
        if session.node._should_stop():
            return False, "stop requested"
        # 优先使用视觉质量门的稳定度量（基于连续帧统计）
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
        # 回退：基于话题接收计数判断
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
# 候选质量快照
# ------------------------------------------------------------------

def candidate_quality_snapshot(
    session, *, marker_note, model_note, stable_note,
    camera_model_metrics, stable_window_metrics,
):
    """
    根据当前视觉门控状态生成一个 AcceptedSampleQuality 实例。
    如果图像观测或相机信息不可用，填充无限大/无限小的占位值。
    """
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
# 精度采样状态检查
# ------------------------------------------------------------------

def precision_sample_status(
    session, candidate, *, quality, recenter_attempted,
    recenter_strict_converged, center_error_limit_px=None,
) -> Tuple[bool, str]:
    """
    精度门控：根据质量指标判断样本是否足够精确，以决定是否接受。
    可配置是否拒绝未严格收敛的非锚点样本，并逐项检查中心误差、相机模型误差、
    中心抖动、深度抖动和角度抖动。
    返回 (是否通过, 详细描述)。
    """
    if not session.sampling_cfg.precision_gate_enabled:
        return True, "precision gate disabled"

    failures = []
    # 非锚点样本如果重新居中但未严格收敛，可能精度不足
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
# 门控赤字严重性检查
# ------------------------------------------------------------------

def is_gate_deficit_critical(candidate, source: str, deficits: dict) -> bool:
    """
    判断某个候选是否能解决当前覆盖度/可观测性的不足（赤字）。
    根据候选的来源家族和偏移方向与 deficits 字典中的布尔值比较，
    如果对应赤字为 True 且候选包含相关运动分量，则认为该候选关键。
    """
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
    if source == "sphere_roll_coverage" and (deficits.get("rot") or deficits.get("motion_rank") or deficits.get("motion_condition")):
        return True
    if source == "sphere_anchor" and (
        deficits.get("pitch") or deficits.get("yaw") or deficits.get("roll")
        or deficits.get("motion_rank") or deficits.get("motion_condition")
    ):
        return True
    return False


# ------------------------------------------------------------------
# 候选规范家族映射构建
# ------------------------------------------------------------------

def build_spec_family_map(base_offsets: dict) -> dict:
    """
    根据 base_offsets 配置构建一个映射表：label -> family_name。
    遍历所有家族及其偏移配置，提取每个偏移的 label 作为键，家族名作为值。
    便于通过 label 快速查询候选来源。
    """
    spec_family_map = {}
    for fam_name in FAMILY_EXECUTION_ORDER:
        offsets = base_offsets.get(fam_name, [])
        for off in offsets:
            spec_family_map[off.label] = fam_name
    return spec_family_map
