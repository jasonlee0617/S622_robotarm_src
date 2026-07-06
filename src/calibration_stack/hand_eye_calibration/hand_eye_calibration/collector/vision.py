"""Vision quality gate: camera info, ArUco observations, stability windows.

本模块实现手眼标定采集流程中的视觉质量门控系统。
核心功能：
- 管理相机内参（CameraInfoState）
- 封装 ArUco 标记的单帧观测结果（ArucoObservation）
- 记录每帧图像的检测状态（ImageFrameStatus）
- 通过滑动窗口统计多帧的稳定性（StableWindowMetrics）
- 提供多种质量等级的观测检查（启动、相机模型验证、采样）
- 等待运动后出现新的成功观测
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# 三个预定义的质量等级名称，用于区分不同场景下的门控严格程度
QUALITY_STARTUP = "startup_visibility"          # 启动时的可见性检查（宽松）
QUALITY_CAMERA_MODEL = "camera_model_check"     # 相机模型一致性检查
QUALITY_SAMPLING = "sampling_quality"           # 正式采样时的严格质量检查


@dataclass
class CameraInfoState:
    """相机内参的快照，线程安全地由 VisionQualityGate 管理和更新。

    包含图像尺寸、焦距、主点坐标以及完整的畸变系数。
    """
    width: int = 0               # 图像宽度（像素）
    height: int = 0              # 图像高度（像素）
    fx: float = 0.0              # 焦距 fx（像素单位）
    fy: float = 0.0              # 焦距 fy（像素单位）
    cx: float = 0.0              # 主点 cx（像素单位）
    cy: float = 0.0              # 主点 cy（像素单位）
    k: Tuple[float, ...] = ()    # 相机矩阵内参（9 个元素的一维元组，对应 3x3 矩阵展开）
    d: Tuple[float, ...] = ()    # 畸变系数 (k1, k2, p1, p2, k3...)

    @property
    def ready(self) -> bool:
        """判断相机内参是否已就绪：必须包含有效的图像尺寸和焦距。"""
        return self.width > 0 and self.height > 0 and self.fx > 0.0 and self.fy > 0.0


@dataclass
class ArucoObservation:
    """单个 ArUco 标记的观测结果。

    由 ArUco 检测器在图像中识别后生成，包含角点坐标、位姿估计以及时间戳等信息。
    """
    receipt_time: float          # 该观测被记录时的系统单调时间（秒）
    center_px: Tuple[float, float]   # 标记中心在图像中的像素坐标 (u, v)
    corners_px: Tuple[Tuple[float, float], ...]  # 四个角点的像素坐标，顺序与 OpenCV 一致
    side_px: float               # 标记最小边长（像素），用于评估尺寸是否足够
    area_px2: float              # 标记在图像中的面积（平方像素）
    margin_px: float             # 标记整体距离图像边缘的最短像素距离
    tvec: Tuple[float, float, float]   # 相机坐标系下的平移向量 (x, y, z)，单位米
    rvec: Tuple[float, float, float]   # 相机坐标系下的旋转向量（Rodrigues），单位弧度
    image_stamp_ns: int = 0      # 原始图像消息的时间戳（纳秒），用于判断帧的新鲜度

    @property
    def distance_m(self) -> float:
        """标记到相机光心的欧氏距离（米）。"""
        return float(np.linalg.norm(np.array(self.tvec, dtype=float)))

    @property
    def angle_deg(self) -> float:
        """标记相对于相机的旋转角度（度），由旋转向量的模长转换得到。"""
        return math.degrees(float(np.linalg.norm(np.array(self.rvec, dtype=float))))


@dataclass
class ImageFrameStatus:
    """每一帧图像的处理状态记录。

    无论是否成功检测到目标标记，都会产生一条记录，包含检测结果和失败原因。
    """
    receipt_time: float          # 该帧被处理的时间
    detected: bool               # 是否成功检测到目标标记
    observation: Optional[ArucoObservation] = None  # 如果检测成功，包含观测数据
    reason: str = ""             # 失败原因或状态描述
    image_stamp_ns: int = 0      # 原始图像的时间戳（纳秒）


@dataclass
class StableWindowMetrics:
    """基于滑动窗口的稳定度量统计。

    对最近 N 帧成功观测计算各项标准差，用于评估当前视觉信号的稳定性。
    """
    latest_observation: ArucoObservation  # 窗口中最新的观测
    center_std_px: float                  # 中心点在连续帧中的标准差（像素）
    depth_std_m: float                    # 深度值的标准差（米）
    angle_std_deg: float                  # 姿态角度的标准差（度）
    window_count: int                     # 实际参与统计的帧数
    note: str                             # 人类可读的总结字符串


class VisionQualityGate:
    """视觉质量门控器。

    负责实时跟踪最新的相机内参和 ArUco 观测，并提供多级质量检查、
    帧状态记录、稳定性评估以及新鲜观测等待等功能。
    """

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
        """初始化质量门控参数及内部状态容器。

        所有阈值由外部配置注入，保证模块可重用。
        """
        self.marker_recent_timeout = float(marker_recent_timeout)     # 标记观测的有效期（秒）
        self.min_marker_distance = float(min_marker_distance)         # 最小允许距离（米）
        self.max_marker_distance = float(max_marker_distance)         # 最大允许距离（米）
        self.startup_min_corner_margin_px = float(startup_min_corner_margin_px)  # 启动阶段的最小角点边缘距离（像素）
        self.min_corner_margin_px = float(min_corner_margin_px)       # 正常采样时的最小角点边缘距离（像素）
        self.min_marker_side_px = float(min_marker_side_px)           # 标记最小边长（像素）
        self.max_center_error_px = float(max_center_error_px)         # 中心误差上限（像素）
        self.stable_frame_count = max(1, int(stable_frame_count))     # 稳定性窗口所需的连续成功帧数
        self.max_center_std_px = float(max_center_std_px)             # 中心标准差上限（像素）
        self.max_depth_std_m = float(max_depth_std_m)                 # 深度标准差上限（米）
        self.max_angle_std_deg = float(max_angle_std_deg)             # 角度标准差上限（度）
        self._logger_warn = logger_warn                               # 外部日志警告函数

        # 相机内参，受锁保护
        self._camera_info = CameraInfoState()
        self._camera_info_lock = threading.Lock()

        # 观测相关的内部状态，受锁保护
        self._observation_lock = threading.Lock()
        self._last_observation: Optional[ArucoObservation] = None       # 最近一次成功观测
        self._observation_history = deque(maxlen=40)                    # 成功观测历史（最大保留40条）
        self._frame_history = deque(maxlen=80)                          # 所有帧的处理状态历史
        self._aruco_exception_counts = {}                               # 异常类型计数字典
        self._last_aruco_exception_log = 0.0                            # 上次记录异常日志的时间（秒）

    # ----------------------------------------------------------------
    # 相机内参管理
    # ----------------------------------------------------------------

    def update_camera_info(self, info: CameraInfoState):
        """线程安全地更新相机内参。"""
        with self._camera_info_lock:
            self._camera_info = info

    def camera_info_snapshot(self) -> CameraInfoState:
        """线程安全地获取当前相机内参的快照（拷贝）。"""
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

    # ----------------------------------------------------------------
    # 观测访问
    # ----------------------------------------------------------------

    def latest_observation(self) -> Optional[ArucoObservation]:
        """返回最近一次成功观测（不区分失败帧）。"""
        with self._observation_lock:
            return self._last_observation

    def latest_successful_observation(self) -> Optional[ArucoObservation]:
        """latest_observation 的显式别名，语义更明确。"""
        return self.latest_observation()

    def latest_frame(self) -> Optional[ImageFrameStatus]:
        """返回最近处理的一帧状态（无论成功与否）。"""
        with self._observation_lock:
            if not self._frame_history:
                return None
            return self._frame_history[-1]

    def last_failed_frame(self) -> Optional[ImageFrameStatus]:
        """从最近帧往前查找，返回第一个失败帧的状态。"""
        with self._observation_lock:
            for frame in reversed(self._frame_history):
                if not frame.detected or frame.observation is None:
                    return frame
        return None

    # ----------------------------------------------------------------
    # 帧状态记录
    # ----------------------------------------------------------------

    def record_frame_status(
        self,
        *,
        detected: bool,
        observation: Optional[ArucoObservation] = None,
        reason: str = "",
        image_stamp_ns: int = 0,
    ):
        """记录一帧图像的处理结果。

        无论是检测成功还是失败，都将生成 ImageFrameStatus 并存入历史。
        成功后同时更新 _last_observation 和 _observation_history。
        """
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

    # ----------------------------------------------------------------
    # 异常日志（带限频）
    # ----------------------------------------------------------------

    def log_aruco_exception(self, context: str, exc: Exception):
        """记录 ArUco 处理中发生的异常，并按类型累积计数，5 秒内只输出一次警告。"""
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

    # ----------------------------------------------------------------
    # 观测质量检查（单帧）
    # ----------------------------------------------------------------

    def observation_quality(
        self,
        obs: Optional[ArucoObservation],
        *,
        quality_level: str,
        require_center: bool,
        center_error_limit_px: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """对单个观测进行多级质量评估。

        检查项：
        - 观测是否为空
        - 观测是否过时（超过 marker_recent_timeout）
        - 标记距离是否在允许范围内
        - 角点距离图像边缘是否足够（根据质量等级使用不同阈值）
        - 标记边长是否足够
        - 中心误差是否过大（如果需要居中且超出阈值）

        返回 (是否合格, 描述字符串)。
        """
        if obs is None:
            failed = self.last_failed_frame()
            if failed is not None and failed.reason:
                return False, f"{quality_level}: no successful image marker observation ({failed.reason})"
            return False, f"{quality_level}: image marker has not been observed"

        age = time.monotonic() - obs.receipt_time
        if age > self.marker_recent_timeout:
            # 若观测已过期，进一步检查是否有更新但失败的帧
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

        # 根据质量等级选择不同的边缘阈值
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
        limit = center_error_limit_px if center_error_limit_px is not None else self.max_center_error_px
        if require_center and center_error > limit:
            return (
                False,
                f"{quality_level}: marker center error too large "
                f"({center_error:.1f}px > {limit:.1f}px)",
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

    # ----------------------------------------------------------------
    # 对最新观测的质量检查（便捷方法）
    # ----------------------------------------------------------------

    def image_marker_status(
        self,
        *,
        require_center: bool = False,
        quality_level: str = QUALITY_SAMPLING,
        center_error_limit_px: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """直接对最近一次观测执行质量检查。"""
        return self.observation_quality(
            self.latest_observation(),
            quality_level=quality_level,
            require_center=require_center,
            center_error_limit_px=center_error_limit_px,
        )

    # ----------------------------------------------------------------
    # 等待新鲜成功观测（用于运动后确认标记可见）
    # ----------------------------------------------------------------

    def wait_for_fresh_successful_observation(
        self,
        *,
        min_receipt_time: float,
        min_stamp_ns: int,
        timeout_sec: float,
        should_stop: Callable[[], bool],
    ) -> Tuple[bool, str]:
        """在超时时间内等待一个时间戳晚于给定值的成功观测。

        用于确保机械臂移动后，视觉系统确实收到了新的、检测到标记的图像帧。
        min_receipt_time: 帧的接收时间必须大于此值
        min_stamp_ns: 图像原始时间戳必须大于此值（如果 <=0 则忽略）
        返回 (是否等到, 描述)。
        """
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

    # ----------------------------------------------------------------
    # 稳定窗口指标
    # ----------------------------------------------------------------

    def stable_window_metrics(
        self,
        *,
        require_center: bool,
        min_receipt_time: float = 0.0,
        min_stamp_ns: int = 0,
        center_error_limit_px: Optional[float] = None,
    ) -> Tuple[Optional[StableWindowMetrics], str]:
        """基于最近 N 帧成功观测计算稳定性指标。

        要求：
        - 必须有至少 stable_frame_count 帧成功、新鲜的观测
        - 所有帧通过采样质量检查
        - 中心点、深度、角度的标准差在允许范围内

        返回 (StableWindowMetrics 对象, 描述) 或 (None, 失败原因)。
        """
        with self._observation_lock:
            recent_frames = list(self._frame_history)[-self.stable_frame_count :]
        if len(recent_frames) < self.stable_frame_count:
            return None, f"need {self.stable_frame_count} image frames, have {len(recent_frames)}"

        # 仅保留满足新鲜度要求的帧
        recent_frames = [
            frame for frame in recent_frames
            if frame.receipt_time >= min_receipt_time and frame.image_stamp_ns >= min_stamp_ns
        ]
        if len(recent_frames) < self.stable_frame_count:
            return (
                None,
                f"need {self.stable_frame_count} fresh image frames after motion, have {len(recent_frames)}"
            )

        # 检查是否有失败帧混入
        failed = [frame for frame in recent_frames if not frame.detected or frame.observation is None]
        if failed:
            last_failed = failed[-1]
            return (
                None,
                "stable image window is not continuous: "
                f"{len(failed)}/{len(recent_frames)} recent frames failed "
                f"({last_failed.reason or 'unknown reason'})",
            )
        recent = [frame.observation for frame in recent_frames if frame.observation is not None]
        now = time.monotonic()
        if any(now - obs.receipt_time > self.marker_recent_timeout for obs in recent):
            return None, "stable image window contains stale marker frames"

        # 对每一帧执行采样质量检查
        for obs in recent:
            ok, reason = self.observation_quality(
                obs,
                quality_level=QUALITY_SAMPLING,
                require_center=require_center,
                center_error_limit_px=center_error_limit_px,
            )
            if not ok:
                return None, reason

        # 计算标准差
        centers = np.array([obs.center_px for obs in recent], dtype=float)
        depths = np.array([obs.distance_m for obs in recent], dtype=float)
        angles = np.array([obs.angle_deg for obs in recent], dtype=float)
        center_std = float(np.max(np.std(centers, axis=0)))  # 取 x、y 标准差的最大值
        depth_std = float(np.std(depths))
        angle_std = float(np.std(angles))

        if center_std > self.max_center_std_px:
            return None, f"center jitter too high ({center_std:.2f}px)"
        if depth_std > self.max_depth_std_m:
            return None, f"depth jitter too high ({depth_std:.4f}m)"
        if angle_std > self.max_angle_std_deg:
            return None, f"angle jitter too high ({angle_std:.2f}deg)"

        latest = recent[-1]
        ok, note = self.observation_quality(
            latest,
            quality_level=QUALITY_SAMPLING,
            require_center=require_center,
            center_error_limit_px=center_error_limit_px,
        )
        if not ok:
            return None, note

        metrics_note = (
            f"stable image marker {len(recent)} frames: {note}, "
            f"std_center={center_std:.2f}px std_depth={depth_std:.4f}m "
            f"std_angle={angle_std:.2f}deg"
        )
        return (
            StableWindowMetrics(
                latest_observation=latest,
                center_std_px=center_std,
                depth_std_m=depth_std,
                angle_std_deg=angle_std,
                window_count=len(recent),
                note=metrics_note,
            ),
            metrics_note,
        )