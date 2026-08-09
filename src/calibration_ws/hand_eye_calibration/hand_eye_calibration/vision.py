"""Pure image-side gates shared by Fairino simulation and real collection.

该模块（vision.py）是手眼标定流程中的“视觉层”，独立于 ROS 节点，主要提供：
  1. 相机内参的抽象与验证（CameraInfoState）；
  2. ArUco 观测数据的数据类（ArucoObservation）；
  3. 基于 OpenCV 的 ArUco 检测器创建与 IPPE 位姿估计；
  4. 核心类 VisionQualityGate：根据多帧稳定性（位置、深度、角度）对观测进行门控，
     确保只有稳定、可靠的观测才能进入样本队列。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
import time
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraInfoState:
    """相机内参状态的数据类，用于存储从 CameraInfo 消息中提取的关键信息。
    
    此类主要关注：
      - 图像宽高（用于判断边缘距离）
      - 投影矩阵 P（用于提取内参矩阵，并验证其合法性）
      - 畸变系数 D（用于 PnP 解算）
      - frame_id（用于匹配图像话题）
    """
    width: int = 0
    height: int = 0
    p: Tuple[float, ...] = ()       # 投影矩阵 P (3x4 的 12 个元素)
    d: Tuple[float, ...] = ()       # 畸变系数 (通常为 5 个元素)
    frame_id: str = ""

    @property
    def ready(self) -> bool:
        """检查内参是否完整且有效：
           - 宽高 > 0
           - P 矩阵至少有 12 个元素，且关键元素（fx, fy, cx, cy）为有限正数
           - frame_id 非空
        """
        matrix_indices = (0, 1, 2, 4, 5, 6, 8, 9, 10)  # 内参矩阵在 P 中的 9 个关键位置
        return (
            self.width > 0
            and self.height > 0
            and len(self.p) >= 12
            and all(math.isfinite(float(self.p[index])) for index in matrix_indices)
            and self.p[0] > 0.0       # fx > 0
            and self.p[5] > 0.0       # fy > 0
            and bool(self.frame_id)
        )

    def camera_matrix(self) -> np.ndarray:
        """从 P 矩阵中提取 3x3 内参矩阵：
           K = [[fx,  s, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]]
           P 矩阵为 P = K * [R|t]，此处仅取左上角 3x3。
        """
        if not self.ready:
            raise ValueError("CameraInfo projection matrix is not ready")
        return np.asarray(((self.p[0], self.p[1], self.p[2]), 
                          (self.p[4], self.p[5], self.p[6]), 
                          (self.p[8], self.p[9], self.p[10])), dtype=float)


@dataclass(frozen=True)
class ArucoObservation:
    """单帧 ArUco 观测的完整数据结构。
    
    包含：
      - 时间戳 (用于稳定窗口的时间对齐)
      - 像素信息 (中心点、四个角点、边长、距图像边缘距离)
      - 3D 位姿信息 (平移向量 tvec、旋转向量 rvec)
    """
    receipt_time: float                      # 接收时间戳 (秒，单调时钟)
    center_px: Tuple[float, float]           # 标记中心在图像中的像素坐标
    corners_px: Tuple[Tuple[float, float], ...]  # 四个角点的像素坐标 (顺序与 OpenCV 一致)
    side_px: float                           # 标记平均边长 (像素)
    margin_px: float                         # 距图像边缘的最小距离 (像素)
    tvec: Tuple[float, float, float]         # 标记相对相机的平移向量 (米)
    rvec: Tuple[float, float, float]         # 标记相对相机的旋转向量 (弧度)

    @property
    def distance_m(self) -> float:
        """计算标记中心到相机光心的欧氏距离 (米)。"""
        return float(np.linalg.norm(np.asarray(self.tvec, dtype=float)))


def create_aruco_detector(dictionary_name: str):
    """创建 OpenCV ArUco 检测器。

    注意：
      - cv2.setNumThreads(0) 用于防止 OpenCV 内部多线程与 ROS 2 的 executor 冲突。
      - 使用 getPredefinedDictionary 兼容不同 OpenCV 版本。
      - 开启 cornerRefinementMethod = CORNER_REFINE_SUBPIX，以获得亚像素精度的角点。
    """
    import cv2

    cv2.setNumThreads(0)   # 禁用 OpenCV 内部多线程，避免与 ROS 2 冲突
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    # 兼容不同 OpenCV 版本的 API
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id) if hasattr(cv2.aruco, "getPredefinedDictionary") else cv2.aruco.Dictionary_get(dictionary_id)
    parameters = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 亚像素精炼
    return dictionary, parameters


def estimate_marker_pose(corners: Sequence[Sequence[float]], marker_size_m: float, info: CameraInfoState):
    """使用 IPPE (Infinitely many Perspective-n-Point) 算法估计 ArUco 码的 3D 位姿。

    IPPE 算法能够快速求解正方形标记的位姿，但会返回多个候选解。
    本函数根据以下规则选择最优解：
      1. 只保留 tvec[2] > 0 (标记在相机前方) 的解；
      2. 选择重投影误差最小的解。

    Args:
      corners: 4个角点的像素坐标 (2D)
      marker_size_m: 标记的物理边长 (米)
      info: 相机内参状态

    Returns:
      (rvec, tvec): 旋转向量和平移向量 (均为 tuple[float, float, float])
    """
    import cv2

    if not math.isfinite(marker_size_m) or marker_size_m <= 0.0:
        raise ValueError("marker_size_m must be finite and positive")
    
    # 定义标记在世界坐标系中的 3D 坐标 (单位：米)，Z=0 平面
    half = marker_size_m * 0.5
    objects = np.asarray(((-half, half, 0.0), (half, half, 0.0), (half, -half, 0.0), (-half, -half, 0.0)), dtype=np.float32)
    image = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    
    # 畸变系数：若没有畸变则设为 5 个 0
    distortion = np.asarray(info.d, dtype=float) if info.d else np.zeros(5, dtype=float)
    if not np.all(np.isfinite(distortion)):
        raise ValueError("CameraInfo distortion coefficients are non-finite")

    # 调用 solvePnPGeneric 返回所有候选解（IPPE 通常返回两个解）
    solved, rotations, translations, errors = cv2.solvePnPGeneric(
        objects, image, info.camera_matrix(), distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not solved or rotations is None or translations is None:
        raise RuntimeError("IPPE returned no pose")
    
    # 过滤出位姿在相机前方 (tvec[2] > 0) 且有限的有效解
    costs = np.asarray(errors if errors is not None else np.full(len(rotations), math.inf), dtype=float).reshape(-1)
    candidates = []
    for index, (rotation, translation) in enumerate(zip(rotations, translations)):
        rvec = np.asarray(rotation, dtype=float).reshape(3)
        tvec = np.asarray(translation, dtype=float).reshape(3)
        if np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)) and tvec[2] > 0.0:
            candidates.append((float(costs[index]) if index < len(costs) else math.inf, rvec, tvec))
    if not candidates:
        raise RuntimeError("IPPE returned no finite positive-depth pose")
    
    # 选择重投影误差最小的候选解
    _, rvec, tvec = min(candidates, key=lambda item: item[0])
    return tuple(float(value) for value in rvec), tuple(float(value) for value in tvec)


def mean_marker_side_px(corners: Sequence[Sequence[float]]) -> float:
    """计算 ArUco 码四条边的平均长度 (像素)。"""
    points = np.asarray(corners, dtype=float).reshape(4, 2)
    return float(statistics.fmean(float(np.linalg.norm(points[(index + 1) % 4] - points[index])) for index in range(4)))


def make_observation(corners: Sequence[Sequence[float]], rvec, tvec, info: CameraInfoState, *, receipt_time: Optional[float] = None) -> ArucoObservation:
    """根据检测到的角点、位姿和相机内参，构造完整的 ArucoObservation 对象。
    
    自动计算：
      - 中心点
      - 图像边缘余量 (margin)
      - 平均边长
    """
    points = np.asarray(corners, dtype=float).reshape(4, 2)
    center = np.mean(points, axis=0)
    # 计算四个方向到图像边界的距离，取最小值
    margin = min(np.min(points[:, 0]), np.min(points[:, 1]), info.width - np.max(points[:, 0]), info.height - np.max(points[:, 1]))
    return ArucoObservation(
        receipt_time=time.monotonic() if receipt_time is None else float(receipt_time),
        center_px=(float(center[0]), float(center[1])),
        corners_px=tuple((float(point[0]), float(point[1])) for point in points),
        side_px=mean_marker_side_px(points),
        margin_px=float(margin),
        tvec=tuple(float(value) for value in tvec),
        rvec=tuple(float(value) for value in rvec),
    )


def median_marker_corners(observations: Sequence[ArucoObservation]) -> Tuple[Tuple[float, float], ...]:
    """对稳定窗口内的多帧角点求中值，用于去除离群帧的影响。
    
    中值滤波比均值滤波对噪声更鲁棒，能有效抑制瞬间抖动。
    """
    observations = tuple(observations)
    if not observations or any(len(observation.corners_px) != 4 for observation in observations):
        raise ValueError("stable marker window has invalid corners")
    # 取所有帧的角点数据，沿第 0 维（帧）取中值
    median = np.median(np.asarray([observation.corners_px for observation in observations], dtype=float), axis=0)
    return tuple((float(point[0]), float(point[1])) for point in median)


class VisionQualityGate:
    """视觉质量门控核心类。
    
    该类维护一个长度为 stable_frames 的队列，存储连续的合格观测。
    只有满足以下条件的观测才会被加入队列：
      1. 单帧质量检查通过（距离在有效范围内、边缘充足、尺寸足够）
      2. 在窗口期内，帧间稳定性满足要求（中心点标准差、深度标准差、角度标准差均低于阈值）

    当任何一帧不合格时，队列会被清空，需要重新积累 stable_frames 帧。
    这样可以确保最终用于解算的观测是高度稳定、噪声极低的。
    """

    def __init__(
        self,
        *,
        marker_distance_min_m: float,
        marker_distance_max_m: float,
        minimum_corner_margin_px: float,
        minimum_marker_side_px: float,
        stable_frames: int,
        maximum_center_std_px: float,
        maximum_marker_depth_std_m: float,
        maximum_marker_angle_std_deg: float,
        logger_warn: Callable[[str], None],
    ):
        """初始化视觉门控参数。

        参数均来自 YAML 配置中的 sampling_config，与采集器中的配置一一对应。
        """
        self.marker_distance_min_m = float(marker_distance_min_m)
        self.marker_distance_max_m = float(marker_distance_max_m)
        self.minimum_corner_margin_px = float(minimum_corner_margin_px)
        self.minimum_marker_side_px = float(minimum_marker_side_px)
        self.stable_frames = int(stable_frames)
        self.maximum_center_std_px = float(maximum_center_std_px)
        self.maximum_marker_depth_std_m = float(maximum_marker_depth_std_m)
        self.maximum_marker_angle_std_deg = float(maximum_marker_angle_std_deg)
        self._logger_warn = logger_warn

        # 相机内参快照 (由 _on_camera_info 回调更新)
        self._camera_info = CameraInfoState()
        self._camera_lock = threading.Lock()   # 保护内参的线程安全

        # 观测队列和保护锁
        self._lock = threading.Lock()
        self._observations: deque[ArucoObservation] = deque(maxlen=self.stable_frames)
        self._latest_observation: Optional[ArucoObservation] = None
        
        # 窗口起始时间：只有在 begin_window() 之后到达的观测才会被记录
        self._minimum_receipt_time = -math.inf
        self._last_failure = "waiting for marker"

        # 异常日志抑制 (避免每帧都打印，减少日志噪音)
        self._exception_counts: dict[str, int] = {}
        self._last_exception_log = 0.0

    def update_camera_info(self, info: CameraInfoState) -> None:
        """线程安全地更新相机内参快照。"""
        with self._camera_lock:
            self._camera_info = info

    def camera_info_snapshot(self) -> CameraInfoState:
        """线程安全地获取当前相机内参的快照。"""
        with self._camera_lock:
            return self._camera_info

    def reset_window(self) -> None:
        """重置整个队列（用于新会话开始前）。"""
        with self._lock:
            self._observations.clear()
            self._latest_observation = None
            self._last_failure = "waiting for marker"

    reset_session = reset_window

    def begin_window(self) -> None:
        """开始一个新的稳定窗口（用于单次样本采集）。
        
        清空队列，并记录当前时间戳。只有在此时间戳之后到达的观测才会被接受。
        这样可以避免在机器人运动过程中积攒的旧帧干扰新窗口的稳定性统计。
        """
        with self._lock:
            self._observations.clear()
            self._latest_observation = None
            self._last_failure = "waiting for marker"
            self._minimum_receipt_time = time.monotonic()

    def record_failure(self, reason: str, *, receipt_time: Optional[float] = None) -> None:
        """记录一次失败观测，并清空队列。
        
        若 receipt_time 早于窗口开始时间，则直接忽略（该帧属于上一个窗口）。
        """
        with self._lock:
            if receipt_time is not None and receipt_time < self._minimum_receipt_time:
                return
            self._observations.clear()
            self._latest_observation = None
            self._last_failure = str(reason)

    def record_success(self, observation: ArucoObservation) -> bool:
        """记录一次成功观测，若满足单帧质量要求则加入队列。

        Returns:
          True: 观测被接受并加入队列
          False: 观测被拒绝，队列已被清空
        """
        ok, reason = self.observation_quality(observation)
        with self._lock:
            self._latest_observation = observation
            if observation.receipt_time < self._minimum_receipt_time:
                return False   # 属于旧窗口，直接丢弃
            if not ok:
                self._observations.clear()   # 任何一帧不合格，整个窗口清零
                self._last_failure = reason
                return False
            self._observations.append(observation)
        return True

    def latest_observation(self) -> Tuple[Optional[ArucoObservation], bool, str]:
        """Return the most recent frame for live guidance without changing its gate state."""
        with self._lock:
            observation = self._latest_observation
            failure = self._last_failure
        if observation is None:
            return None, False, failure
        accepted, reason = self.observation_quality(observation)
        return observation, accepted, reason

    def observation_quality(self, observation: ArucoObservation) -> Tuple[bool, str]:
        """单帧观测的质量检查（距离、边缘、大小、数值有效性）。"""
        values = (*observation.tvec, *observation.rvec, observation.margin_px, observation.side_px)
        if not all(math.isfinite(float(value)) for value in values) or observation.tvec[2] <= 0.0:
            return False, "non-finite observation or non-positive depth"
        if not self.camera_info_snapshot().ready:
            return False, "CameraInfo projection matrix is not ready"
        if not self.marker_distance_min_m <= observation.distance_m <= self.marker_distance_max_m:
            return False, "marker distance is outside the calibration range"
        if observation.margin_px < self.minimum_corner_margin_px:
            return False, "marker is too close to the image edge"
        if observation.side_px < self.minimum_marker_side_px:
            return False, "marker is too small in the image"
        return True, "marker frame accepted"

    def stable_window(self) -> Tuple[Optional[Tuple[ArucoObservation, ...]], str]:
        """检查当前队列是否满足稳定窗口条件。

        要求：
          1. 队列长度达到 stable_frames
          2. 中心点像素标准差 <= maximum_center_std_px
          3. 深度 (tvec[2]) 标准差 <= maximum_marker_depth_std_m
          4. 姿态角 (rvec) 标准差 <= maximum_marker_angle_std_deg

        Returns:
          (frames, reason): 若满足，frames 为元组；否则 frames=None，reason 为失败原因
        """
        with self._lock:
            frames = tuple(self._observations)
            last_reason = self._last_failure
        if len(frames) < self.stable_frames:
            return None, last_reason if last_reason != "waiting for marker" else "insufficient stable marker frames"
        
        # 中心点标准差 (取 x 和 y 方向的最大值)
        center_std = max(statistics.pstdev(frame.center_px[0] for frame in frames), 
                         statistics.pstdev(frame.center_px[1] for frame in frames))
        if center_std > self.maximum_center_std_px:
            return None, f"marker centre is unstable: {center_std:.3f}px"
        
        # 深度 (Z轴) 标准差
        depth_std = statistics.pstdev(frame.tvec[2] for frame in frames)
        if depth_std > self.maximum_marker_depth_std_m:
            return None, f"marker depth is unstable: {depth_std:.6f}m"
        
        # 角度标准差：计算每个 rvec 相对于均值 rvec 的旋转向量差值范数，再求均方根
        mean_rvec = np.mean(np.asarray([frame.rvec for frame in frames], dtype=float), axis=0)
        angle_std = math.degrees(math.sqrt(statistics.fmean(float(np.linalg.norm(np.asarray(frame.rvec) - mean_rvec)) ** 2 for frame in frames)))
        if angle_std > self.maximum_marker_angle_std_deg:
            return None, f"marker angle is unstable: {angle_std:.3f}deg"
        
        return frames, "marker observation is stable"

    def log_exception(self, context: str, exc: Exception) -> None:
        """记录并抑制频繁的异常日志（防止日志刷屏）。
        
        相同类型的异常每 5 秒仅打印一次，并附带累计计数。
        """
        key = f"{context}:{type(exc).__name__}"
        self._exception_counts[key] = self._exception_counts.get(key, 0) + 1
        now = time.monotonic()
        if now - self._last_exception_log >= 5.0:
            self._last_exception_log = now
            self._logger_warn(f"ArUco {key}: {exc} (count={self._exception_counts[key]})")
