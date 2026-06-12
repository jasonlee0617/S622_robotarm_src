#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# 导入所需的所有ROS2和Python库
# =============================================================================

from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import rclpy                     # ROS2 Python客户端库
from rclpy.node import Node      # ROS2节点基类
from rclpy.qos import (          # 服务质量(QoS)设置
    qos_profile_sensor_data,     # 传感器数据的默认QoS配置（适合高频率、可能丢失数据）
    ReliabilityPolicy,           # 可靠性策略：可靠或尽力而为
    HistoryPolicy,               # 历史策略：保留最近N条或全部
    QoSProfile,                  # QoS配置类
    DurabilityPolicy,            # 持久性策略：易失或瞬态本地
)
from sensor_msgs.msg import Image, CameraInfo          # ROS图像和相机信息消息
from geometry_msgs.msg import PointStamped             # 带时间戳的3D点消息
from std_msgs.msg import Header, Float32MultiArray     # 标准消息头和多维浮点数组
import torch                     # PyTorch，用于深度学习设备管理
from cv_bridge import CvBridge, CvBridgeError          # ROS图像与OpenCV图像互转
from ultralytics import YOLO     # YOLOv8模型接口
import time                      # 时间相关函数
import cv2                       # OpenCV图像处理
import numpy as np               # 数值计算库
import threading                 # 线程锁，用于保护共享数据
import math                      # 数学函数
from collections import deque
from message_filters import Subscriber, ApproximateTimeSynchronizer  # 时间同步多个话题
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup     # 互斥回调组
from rclpy.executors import MultiThreadedExecutor        # 多线程执行器


# =============================================================================
# 角度处理辅助函数
# =============================================================================

def wrap_to_pi(a: float) -> float:
    """
    将任意角度 a 归一化到 (-π, π] 范围内。
    数学原理: 使用取模运算和调整，等价于 atan2(sin(a), cos(a))
    """
    # 将 a 加上 π 后对 2π 取模，再减去 π，确保结果在 (-π, π]
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def angle_diff(a: float, b: float) -> float:
    """
    计算两个角度的最小差值，结果范围 (-π, π]。
    表示从 b 到 a 需要旋转的最短有向角度。
    """
    return wrap_to_pi(a - b)


def choose_equivalent_angle(cur: float, prev: float, period: float) -> float:
    """
    在角度周期对称性下，选择与历史角度 prev 最接近的等效角度。
    参数:
        cur: 当前观测角度（在 [0,π) 或 [0,π/2) 等基本周期内）
        prev: 上一时刻的连续角度值（已unwrap）
        period: 角度周期（π 或 π/2），取决于物体的对称性
    返回: 与 prev 最接近的等效角度，并 wrap 到 (-π, π]
    """
    best = cur
    best_err = abs(angle_diff(cur, prev))
    # 尝试前后各4个周期，找到与 prev 角度差最小的候选
    for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        cand = cur + k * period       # 候选角度 = 基本角度 + k * 周期
        err = abs(angle_diff(cand, prev))
        if err < best_err:
            best_err = err
            best = cand
    # 最终将最佳候选包装回 (-π, π] 范围
    return wrap_to_pi(best)


def yaw_0_to_pi_right0_left180(corners_2d: np.ndarray) -> float:
    """
    从定向边界框(OBB)的四个角点计算物体的偏航角 yaw。
    方法：找到四边形中最长的边，将其方向作为主方向。
    结果限制在 [0, π] 范围内，0 表示指向图像右侧，π 表示指向左侧。
    参数:
        corners_2d: 形状 (4,2) 的 numpy 数组，四个角点坐标 (x,y)
    返回:
        yaw 角度（弧度），范围 [0, π]
    """
    c = corners_2d.astype(np.float32)   # 转为浮点数
    best_v = None
    best_len = -1.0

    # 遍历四条边，找到最长边
    for i in range(4):
        v = c[(i + 1) % 4] - c[i]       # 边的向量
        L = float(np.linalg.norm(v))    # 边长
        if L > best_len:
            best_len = L
            best_v = v                  # 最长边的向量

    if best_v is None or best_len < 1e-6:
        return 0.0                      # 安全默认值

    dx, dy = float(best_v[0]), float(best_v[1])
    # 使用 atan2 计算绝对方向，再取绝对值，将 yaw 限制在第一象限 [0, π/2]？不，代码是 atan2(|dy|, dx)
    # 实际公式: yaw = arctan(|dy| / dx)，当 dx>0 时 yaw 在 [0, π/2]；若 dx<0，则 yaw 在 [π/2, π] 之间
    # 最终范围是 [0, π]
    yaw = math.atan2(abs(dy), dx)
    yaw = max(0.0, min(math.pi, yaw))   # 确保在 [0, π] 内
    return float(yaw)


# =============================================================================
# 从 YOLO 结果中提取 OBB 角点
# =============================================================================

def try_extract_obb_corners(result, i_det):
    """
    尝试从 YOLO 检测结果中提取第 i_det 个定向边界框的四个角点。
    参数:
        result: YOLO 预测结果对象
        i_det:  检测框索引
    返回:
        形状 (4,2) 的 numpy 数组，若失败则返回 None
    """
    obb = getattr(result, 'obb', None)        # 获取定向框属性
    if obb is None:
        return None
    # 检查是否存在 xyxyxyxy 属性（四个角点坐标）
    if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > i_det:
        try:
            # 取出第 i_det 个框，从 GPU 转到 CPU，转为 numpy，再重塑为 (4,2)
            arr = obb.xyxyxyxy[i_det].detach().cpu().numpy().reshape(-1, 2)
            if arr.shape[0] >= 4:
                return arr[:4]                # 返回前4个角点
        except Exception:
            pass
    return None


# =============================================================================
# 3D 位置卡尔曼滤波器 (常速度模型)
# =============================================================================


def resolve_yolo_model_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(get_package_share_directory("yolo_model")) / path)


class CVKalmanFilter3D:
    """
    三维空间常速度模型卡尔曼滤波器。
    状态向量: [x, y, z, vx, vy, vz]^T
    观测向量: [x, y, z]^T
    支持预测、更新和门控抗野值。
    """

    def __init__(
        self,
        q_pos=1e-4,        # 位置过程噪声方差 (单位: m^2/s^2? 实际是位置随机游走强度)
        q_vel=5e-3,        # 速度过程噪声方差 (单位: m^2/s^4? 加速度噪声)
        r_meas=2e-3,       # 观测噪声方差 (单位: m^2)
        max_jump=0.20,     # 最大允许跳跃距离 (米)，超过则视为野值
    ):
        # 保存参数
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.max_jump = float(max_jump)

        # 滤波器状态标志
        self.initialized = False
        # 状态向量 (6x1): [x, y, z, vx, vy, vz]^T
        self.x = np.zeros((6, 1), dtype=np.float64)
        # 状态协方差矩阵 (6x6)，初始值较大表示不确定
        self.P = np.eye(6, dtype=np.float64) * 1.0
        # 上次更新的时间戳（秒）
        self.last_t = None

        # 观测矩阵 H (3x6): 将状态映射到观测空间
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # 观测噪声协方差矩阵 R (3x3)
        self.R = np.eye(3, dtype=np.float64) * self.r_meas

    def reset(self):
        """重置滤波器，清除所有历史信息"""
        self.initialized = False
        self.x[:] = 0.0
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.last_t = None

    def init_state(self, xyz: np.ndarray, t_sec: float):
        """
        用第一次观测初始化滤波器。
        xyz: 观测位置 [x,y,z]
        t_sec: 观测时间戳（秒）
        """
        self.x[:] = 0.0
        self.x[0:3, 0] = xyz.reshape(3,)        # 位置设为观测值
        # 协方差: 位置部分较小 (0.1)，速度部分较大 (10) 表示速度未知
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self.P[3:, 3:] *= 10.0
        self.initialized = True
        self.last_t = float(t_sec)

    def _build_F_Q(self, dt: float):
        """
        根据时间间隔 dt 构建状态转移矩阵 F 和过程噪声协方差矩阵 Q。
        F = [[1,0,0,dt,0,0],
             [0,1,0,0,dt,0],
             [0,0,1,0,0,dt],
             [0,0,0,1,0,0],
             [0,0,0,0,1,0],
             [0,0,0,0,0,1]]
        Q 为对角阵，位置噪声方差 q_pos，速度噪声方差 q_vel。
        注意：更精确的连续模型离散化应包含 dt^3/3 等项，此处简化。
        """
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0, 0] = self.q_pos
        Q[1, 1] = self.q_pos
        Q[2, 2] = self.q_pos
        Q[3, 3] = self.q_vel
        Q[4, 4] = self.q_vel
        Q[5, 5] = self.q_vel
        return F, Q

    def predict_to(self, t_sec: float):
        """
        将滤波器状态预测到指定时刻 t_sec（仅预测，不更新）。
        返回预测后的位置，若未初始化则返回 None。
        """
        if not self.initialized or self.last_t is None:
            return None

        # 计算时间差，限制在 [1e-4, 0.2] 秒内防止数值问题
        dt = float(np.clip(t_sec - self.last_t, 1e-4, 0.2))
        F, Q = self._build_F_Q(dt)

        # 状态预测: x = F * x
        self.x = F @ self.x
        # 协方差预测: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + Q
        self.last_t = float(t_sec)
        return self.get_pos()

    def update(self, xyz_meas: np.ndarray, t_sec: float):
        """
        卡尔曼更新步骤：用观测 xyz_meas 更新状态。
        如果滤波器未初始化，则调用 init_state。
        如果观测跳跃过大（超过 max_jump），则临时增大观测噪声 R 以减弱该观测的影响。
        返回更新后的位置。
        """
        xyz_meas = np.asarray(xyz_meas, dtype=np.float64).reshape(3,)

        if not self.initialized:
            self.init_state(xyz_meas, t_sec)
            return self.get_pos()

        # 先预测到观测时刻
        self.predict_to(t_sec)

        # 门控: 计算观测与预测位置之间的欧氏距离
        pred = self.get_pos()
        jump = float(np.linalg.norm(xyz_meas - pred))
        if jump > self.max_jump:
            # 野值: 将观测噪声放大10倍，降低此观测的权重
            R_use = self.R * 10.0
        else:
            R_use = self.R

        # 观测向量 z (3x1)
        z = xyz_meas.reshape(3, 1)
        # 残差 y = z - H*x
        y = z - self.H @ self.x
        # 新息协方差 S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + R_use
        # 卡尔曼增益 K = P*H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新: x = x + K*y
        self.x = self.x + K @ y
        # 协方差更新: P = (I - K*H) * P
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.get_pos()

    def get_pos(self):
        """返回当前估计的位置 (x,y,z)"""
        return self.x[0:3, 0].copy()

    def get_vel(self):
        """返回当前估计的速度 (vx,vy,vz)"""
        return self.x[3:6, 0].copy()


# =============================================================================
# 角度卡尔曼滤波器 (常角速度模型)
# =============================================================================

class AngleKalmanFilter:
    """
    一维角度卡尔曼滤波器，状态为 [yaw, yaw_rate]。
    内部使用连续（未缠绕）的角度值，避免角度跳变问题。
    """

    def __init__(
        self,
        q_yaw=1e-3,      # 角度过程噪声方差 (rad^2/s^2? 实际是角度随机游走)
        q_rate=5e-2,     # 角速度过程噪声方差 (rad^2/s^4? 角加速度噪声)
        r_yaw=3e-3,      # 角度观测噪声方差 (rad^2)
    ):
        self.q_yaw = float(q_yaw)
        self.q_rate = float(q_rate)
        self.r_yaw = float(r_yaw)

        self.initialized = False
        # 状态向量 [theta, theta_dot]^T (theta 为连续角度)
        self.x = np.zeros((2, 1), dtype=np.float64)
        # 协方差矩阵
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.last_t = None

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.x[:] = 0.0
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.last_t = None

    def init_state(self, yaw: float, t_sec: float):
        """
        用第一次观测角度初始化滤波器。
        注意: 输入的 yaw 应为连续等效角度（已通过 choose_equivalent_angle 处理）
        """
        self.x[:] = 0.0
        self.x[0, 0] = float(yaw)          # 角度状态
        # 协方差: 角度部分较小 (0.1)，角速度部分较大 (10)
        self.P = np.eye(2, dtype=np.float64) * 0.1
        self.P[1, 1] = 10.0
        self.initialized = True
        self.last_t = float(t_sec)

    def predict_to(self, t_sec: float):
        """
        预测到指定时刻，返回预测后的角度（包装到 (-π,π]）。
        """
        if not self.initialized or self.last_t is None:
            return None

        dt = float(np.clip(t_sec - self.last_t, 1e-4, 0.2))
        # 状态转移矩阵 F = [[1, dt], [0, 1]]
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)
        # 过程噪声协方差 Q = [[q_yaw, 0], [0, q_rate]]
        Q = np.array([[self.q_yaw, 0.0],
                      [0.0, self.q_rate]], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_t = float(t_sec)
        return self.get_yaw_wrapped()

    def update(self, yaw_meas_equiv: float, t_sec: float):
        """
        卡尔曼更新步骤。
        yaw_meas_equiv: 经过 choose_equivalent_angle 处理后的连续等效角度
        """
        yaw_meas_equiv = float(yaw_meas_equiv)

        if not self.initialized:
            self.init_state(yaw_meas_equiv, t_sec)
            return self.get_yaw_wrapped()

        self.predict_to(t_sec)

        # 观测矩阵 H = [[1, 0]]
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        # 观测噪声 R = [[r_yaw]]
        R = np.array([[self.r_yaw]], dtype=np.float64)
        z = np.array([[yaw_meas_equiv]], dtype=np.float64)

        # 残差
        y = z - H @ self.x
        # 新息协方差
        S = H @ self.P @ H.T + R
        # 卡尔曼增益
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ H) @ self.P
        return self.get_yaw_wrapped()

    def get_yaw_wrapped(self):
        """返回包装到 (-π, π] 的角度"""
        return wrap_to_pi(float(self.x[0, 0]))

    def get_yaw_rate(self):
        """返回角速度估计值 (rad/s)"""
        return float(self.x[1, 0])


# =============================================================================
# 单个物体的轨迹容器
# =============================================================================

class ObjectTrack:
    """
    管理一个物体的卡尔曼滤波器、观测缓存和丢失计数。
    """
    def __init__(self, cls_id: int, name: str):
        self.cls_id = int(cls_id)      # 类别ID: 0=pen, 1=box, 2=cube
        self.name = str(name)          # 类别名称

        # 创建两个独立的卡尔曼滤波器
        self.xyz_kf = CVKalmanFilter3D()   # 3D位置滤波器
        self.yaw_kf = AngleKalmanFilter()  # 角度滤波器

        # 最近一次的观测值（用于调试）
        self.last_meas_xyz = None
        self.last_meas_yaw = None
        # 最近一次收到观测的系统时间（墙上时间，秒）
        self.last_update_wall = 0.0
        # 最近一次观测的ROS消息头
        self.last_header = None
        # 连续未检测到的帧数
        self.missed_count = 0

    def reset(self):
        """重置该物体的所有状态和滤波器"""
        self.xyz_kf.reset()
        self.yaw_kf.reset()
        self.last_meas_xyz = None
        self.last_meas_yaw = None
        self.last_update_wall = 0.0
        self.last_header = None
        self.missed_count = 0


# =============================================================================
# 主节点类 YoloDetectorNode
# =============================================================================

class YoloDetectorNode(Node):
    def __init__(self):
        # 初始化ROS节点，节点名 'yolov8_Kalman_detector_node'
        super().__init__('yolov8_Kalman_detector_node')

        # ------------------------------
        # 声明所有ROS参数，允许通过launch文件或yaml文件配置
        # ------------------------------

        # YOLO模型相关
        self.declare_parameter('backend', 'torch')        # 'torch' or 'tensorrt'
        self.declare_parameter('model_path', 'yolo-obb-gazebo.pt')
        self.declare_parameter('engine_path', 'yolo-obb-gazebo.engine')
        self.declare_parameter('device', 'auto')          # 'auto', 'cuda:0', 'cpu'
        self.declare_parameter('conf', 0.2)               # 检测置信度阈值
        self.declare_parameter('imgsz', 640)              # 输入图像尺寸

        # 深度相机相关
        self.declare_parameter('depth_max_range', 10.0)   # 深度有效最大距离(m)
        self.declare_parameter('publish_rpy', True)       # 是否发布RPY（仅有yaw）

        # 点云采样参数（从OBB区域提取3D点时）
        self.declare_parameter('stride_pen', 5)           # pen类的采样步长
        self.declare_parameter('stride_box', 10)          # box类的采样步长
        self.declare_parameter('stride_cube', 1)          # cube类的采样步长
        self.declare_parameter('max_points', 5000)        # 最大采样点数
        self.declare_parameter('min_points_pen', 20)      # pen类最少点数要求
        self.declare_parameter('min_points_box', 200)     # box类最少点数要求
        self.declare_parameter('min_points_cube', 50)     # cube类最少点数要求

        # 处理周期和发布频率
        self.declare_parameter('inference_period', 0.0166)     # 检测周期（秒），约30Hz
        self.declare_parameter('pose_publish_rate', 60.0)     # 位置发布频率（Hz）
        self.declare_parameter('hold_last_seconds', 0.13)     # 无新观测时保持输出的最大时间

        # 旧版平滑参数（兼容，但不再使用）
        self.declare_parameter('yaw_smoothing_alpha', 0.3)
        self.declare_parameter('xyz_smoothing_alpha', 0.8)

        # 卡尔曼滤波开关及参数
        self.declare_parameter('use_kalman_filter', True)     # 是否启用卡尔曼滤波
        self.declare_parameter('kf_q_pos', 0.0001)            # 位置过程噪声
        self.declare_parameter('kf_q_vel', 0.003)              # 速度过程噪声
        self.declare_parameter('kf_r_xyz', 0.02)               # 位置观测噪声
        self.declare_parameter('kf_max_xyz_jump', 0.025)       # 最大跳跃距离(m)
        self.declare_parameter('kf_q_yaw', 1e-3)              # 角度过程噪声
        self.declare_parameter('kf_q_yaw_rate', 5e-2)         # 角速度过程噪声
        self.declare_parameter('kf_r_yaw', 3e-3)              # 角度观测噪声
        self.declare_parameter('max_predict_seconds', 0.04)    # 最大预测外推时间(s)
        self.declare_parameter('max_missed_frames', 6)        # 最大连续丢失帧数，超过则重置

        # ROS话题名称
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')

        # 时间同步参数
        self.declare_parameter('sync_queue_size', 10)         # 同步队列大小
        self.declare_parameter('sync_slop', 0.015)             # 最大时间差(s)

        # 读取参数值并存储到成员变量
        self.backend = self.get_parameter('backend').get_parameter_value().string_value.lower()
        engine_path = resolve_yolo_model_path(self.get_parameter('engine_path').get_parameter_value().string_value)
        model_path = resolve_yolo_model_path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.depth_max_range = float(self.get_parameter('depth_max_range').value)
        self.publish_rpy = bool(self.get_parameter('publish_rpy').value)

        self.stride_pen = int(self.get_parameter('stride_pen').value)
        self.stride_box = int(self.get_parameter('stride_box').value)
        self.stride_cube = int(self.get_parameter('stride_cube').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.min_points_pen = int(self.get_parameter('min_points_pen').value)
        self.min_points_box = int(self.get_parameter('min_points_box').value)
        self.min_points_cube = int(self.get_parameter('min_points_cube').value)

        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        self.inference_period = float(self.get_parameter('inference_period').value)
        self.pose_publish_rate = float(self.get_parameter('pose_publish_rate').value)
        self.hold_last_seconds = float(self.get_parameter('hold_last_seconds').value)

        self.use_kf = bool(self.get_parameter('use_kalman_filter').value)
        self.kf_q_pos = float(self.get_parameter('kf_q_pos').value)
        self.kf_q_vel = float(self.get_parameter('kf_q_vel').value)
        self.kf_r_xyz = float(self.get_parameter('kf_r_xyz').value)
        self.kf_max_xyz_jump = float(self.get_parameter('kf_max_xyz_jump').value)

        self.kf_q_yaw = float(self.get_parameter('kf_q_yaw').value)
        self.kf_q_yaw_rate = float(self.get_parameter('kf_q_yaw_rate').value)
        self.kf_r_yaw = float(self.get_parameter('kf_r_yaw').value)

        self.max_predict_seconds = float(self.get_parameter('max_predict_seconds').value)
        self.max_missed_frames = int(self.get_parameter('max_missed_frames').value)

        # ------------------------------
        # 确定计算设备 (CPU/GPU)
        # ------------------------------
        if self.device.lower() == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.get_logger().info('CUDA is available, using GPU.')
            else:
                self.device = 'cpu'
                self.get_logger().info('CUDA not available, using CPU.')

        # 加载YOLO模型
        # self.get_logger().info(f'Loading YOLOv8 model: {model_path}, device={self.device}')
        # self.model = YOLO(model_path)
        # self.model.fuse()
        # try:
        #     self.model.to(self.device)      # 将模型移动到指定设备
        # except Exception:
        #     pass
        if self.backend == 'tensorrt':
            self.get_logger().info(f'Loading YOLOv8 TensorRT engine: {engine_path}')
            self.model = YOLO(engine_path)

        elif self.backend == 'torch':
            self.get_logger().info(f'Loading YOLOv8 PyTorch model: {model_path}, device={self.device}')
            self.model = YOLO(model_path)
            self.model.fuse()
            try:
                self.model.to(self.device)
            except Exception:
                pass

        else:
            raise ValueError(f"Unsupported backend: {self.backend}, use 'torch' or 'tensorrt'")



        # 创建ROS图像与OpenCV图像转换器
        self.bridge = CvBridge()

        # 相机内参，从CameraInfo消息中获取
        self.camera_intrinsics = None
        # 最近接收的RGB图像、深度图像和消息头（使用锁保护）
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_header = None
        self.lock = threading.Lock()

        # 类别名称和对应的可视化颜色
        self.class_names = {0: 'pen', 1: 'box', 2: 'cube'}
        self.class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255)}
        self.default_color = (0, 255, 255)

        # 创建三个物体的轨迹对象
        self.tracks = {
            0: ObjectTrack(0, 'pen'),
            1: ObjectTrack(1, 'box'),
            2: ObjectTrack(2, 'cube'),
        }
        # 根据参数配置每个轨迹内的卡尔曼滤波器参数
        self._configure_tracks()

        # 创建回调组，用于将检测和发布任务分配到不同线程，避免互锁
        self.cb_infer = MutuallyExclusiveCallbackGroup()  # 检测回调组
        self.cb_pub = MutuallyExclusiveCallbackGroup()    # 发布回调组

        # 订阅相机内参话题（只订阅一次，获取后自动销毁）
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # 使用 message_filters 进行时间同步的RGB和深度订阅器
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        # 近似时间同步器：允许最大 slop 秒的时间差
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False
        )
        # 注册同步回调函数
        self.sync.registerCallback(self.synced_rgb_depth_callback)

        # 创建可靠传输、只保留最新的QoS配置（用于发布结果）
        qos_reliable_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # 发布可视化图像
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', qos_reliable_latest)

        # 发布三个物体的3D位置
        self.pub_pen_position = self.create_publisher(PointStamped, '/pen_position_3d', qos_reliable_latest)
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', qos_reliable_latest)
        self.pub_cube_position = self.create_publisher(PointStamped, '/cube_position_3d', qos_reliable_latest)

        # 可选发布RPY（仅yaw）
        self.pub_pen_rpy = self.create_publisher(Float32MultiArray, '/pen_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_box_rpy = self.create_publisher(Float32MultiArray, '/box_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_cube_rpy = self.create_publisher(Float32MultiArray, '/cube_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_vision_latency_trace = self.create_publisher(Float32MultiArray, '/vision_latency_trace', qos_reliable_latest)

        # 标志，防止重入 process_images
        self._busy = False
        self._last_dt = 0.0
        self._vision_lat_hist = deque(maxlen=300)
        # 定时器：按 inference_period 执行 YOLO 检测和滤波更新
        self.detection_timer = self.create_timer(self.inference_period, self.process_images, callback_group=self.cb_infer)
        # 定时器：按 pose_publish_rate 发布滤波/预测后的结果
        self.publish_timer = self.create_timer(1.0 / max(1.0, self.pose_publish_rate), self.publish_cached_outputs, callback_group=self.cb_pub)

    def _configure_tracks(self):
        """
        根据参数配置每个物体的卡尔曼滤波器参数。
        """
        for _, trk in self.tracks.items():
            # 重新创建3D位置滤波器，传入参数
            trk.xyz_kf = CVKalmanFilter3D(
                q_pos=self.kf_q_pos,
                q_vel=self.kf_q_vel,
                r_meas=self.kf_r_xyz,
                max_jump=self.kf_max_xyz_jump,
            )
            # 重新创建角度滤波器，传入参数
            trk.yaw_kf = AngleKalmanFilter(
                q_yaw=self.kf_q_yaw,
                q_rate=self.kf_q_yaw_rate,
                r_yaw=self.kf_r_yaw,
            )

    def camera_info_callback(self, msg: CameraInfo):
        """
        接收相机内参消息，提取 fx, fy, cx, cy 并保存。
        获取后立即销毁订阅器，避免重复处理。
        """
        if self.camera_intrinsics is None:
            # 从 CameraInfo 的 K 矩阵中提取内参: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info(
                f'Camera intrinsics: fx={self.camera_intrinsics["fx"]:.1f}, '
                f'fy={self.camera_intrinsics["fy"]:.1f}, '
                f'cx={self.camera_intrinsics["cx"]:.1f}, '
                f'cy={self.camera_intrinsics["cy"]:.1f}'
            )
            # 内参已获取，不再需要订阅相机信息话题
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def synced_rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        """
        时间同步后的RGB和深度图像回调。
        将ROS图像转为OpenCV格式，并进行深度预处理，然后存储到 latest_* 变量中。
        """
        # 转换RGB图像 (bgr8格式)
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB convert error: {e}")
            return

        # 转换深度图像
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            # 如果深度编码是16UC1（毫米），则转换为米（除以1000）
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_image = depth_image.astype(np.float32) / 1000.0
            else:
                depth_image = depth_image.astype(np.float32)

            # 将深度值限制在20米以内，避免无效值
            depth_image[depth_image > 20.0] = 20.0
            # 将NaN或无穷替换为0
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=20.0, neginf=0.0)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth convert error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")
            return

        # 使用线程锁保护，更新最新数据
        with self.lock:
            self.latest_rgb = rgb
            self.latest_depth = depth_image
            self.latest_header = rgb_msg.header

    def _msg_time_to_sec(self, header):
        """将ROS消息头中的时间戳转换为浮点数秒"""
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
    
    def _now_ros_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9
    
    def _publish_vision_latency_trace(self, t_img_sec: float, t_det_sec: float, det_cost_sec: float, infer_cost_sec:float, n_det: int):
        try:
            m = Float32MultiArray()
            m.data = [
                float(t_img_sec),
                float(t_det_sec),
                float(t_det_sec - t_img_sec),
                float(det_cost_sec),
                float(infer_cost_sec),
                float(n_det),
            ] 
            self.pub_vision_latency_trace.publish(m)
            self._vision_lat_hist.append(float(t_det_sec - t_img_sec))
            if len(self._vision_lat_hist) >= 30:
                arr = np.asarray(self._vision_lat_hist, dtype=np.float64)
                if np.random.rand() < 0.02:
                    self.get_logger().info(
                        f"vision latency img->det mean={np.mean(arr)*1000.0:.1f}ms p95={np.percentile(arr,95)*1000.0:.1f}ms max={np.max(arr)*1000.0:.1f}ms"
                    )
        except Exception as e:
            self.get_logger().warn(f'publish vision latency failed: {e}')

    def _center3d_from_obb_depth(self, poly_2d: np.ndarray, depth: np.ndarray, cls: int):
        """
        给定OBB的多边形角点和深度图，计算物体在相机坐标系下的3D质心。
        步骤：
        1. 根据多边形生成掩膜，并腐蚀以去除边界。
        2. 在掩膜区域内按类别指定步长采样有效深度点。
        3. 使用相机内参将像素(u,v)和深度Z转换为3D点(X,Y,Z)。
        4. 返回所有有效点的均值作为质心。
        """
        H, W = depth.shape[:2]
        # 创建掩膜并填充多边形区域
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_2d.astype(np.int32)], 255)
        # 腐蚀操作：去掉边缘像素，减少深度噪声
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        # 获取掩膜内所有像素坐标
        ys, xs = np.where(mask > 0)
        if xs.size < 100:          # 如果区域太小，提前返回
            return None

        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']

        # 根据类别选择采样步长和最少点数要求
        if cls == 0:   # pen
            stride = max(1, self.stride_pen)
            min_points = self.min_points_pen
        elif cls == 1: # box
            stride = max(1, self.stride_box)
            min_points = self.min_points_box
        elif cls == 2: # cube
            stride = max(1, self.stride_cube)
            min_points = self.min_points_cube
        else:
            stride = 1
            min_points = 50

        pts = []
        count = 0
        # 按步长遍历有效像素
        for u, v in zip(xs[::stride], ys[::stride]):
            Z = float(depth[v, u])
            # 检查深度有效且在范围内
            if np.isfinite(Z) and 0.0 < Z <= self.depth_max_range:
                # 相机投影公式: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
                X = (float(u) - cx) * Z / fx
                Y = (float(v) - cy) * Z / fy
                pts.append([X, Y, Z])
                count += 1
                if count >= self.max_points:   # 达到最大采样点数，提前停止
                    break

        if len(pts) < min_points:
            return None

        P = np.asarray(pts, dtype=np.float32)
        return np.mean(P, axis=0)   # 返回质心

    def _measurement_yaw_equiv(self, cls: int, yaw_meas_0_pi: float, track: ObjectTrack) -> float:
        """
        将观测到的 yaw（范围 [0,π]）转换为与轨迹历史最接近的等效连续角度。
        考虑物体对称性：pen 周期 π，box/cube 周期 π/2。
        """
        # 获取上一时刻滤波器的角度（包装后的，用于比较）
        prev = None
        if track.yaw_kf.initialized:
            prev = track.yaw_kf.get_yaw_wrapped()

        # 根据类别确定对称周期
        if cls in (1, 2):   # box 或 cube 有90度对称性
            period = math.pi / 2.0
        else:               # pen 或其它，周期为 π
            period = math.pi

        # 将观测角度映射到 (-π, π] 范围内，方便处理负角度
        meas_rep = yaw_meas_0_pi
        if meas_rep > (math.pi / 2.0):
            meas_rep = meas_rep - math.pi   # 例如 150° -> -30°

        if prev is None:
            return float(meas_rep)

        # 调用 choose_equivalent_angle 找到最接近历史角度的等效值
        return choose_equivalent_angle(meas_rep, prev, period=period)

    def _yaw_wrapped_to_0_pi(self, yaw_wrapped: float) -> float:
        """
        将包装在 (-π, π] 的角度转换为 [0, π] 范围。
        用于输出和显示。
        """
        y = float(yaw_wrapped)
        if y < 0.0:
            y += math.pi
        return max(0.0, min(math.pi, y))

    def _predict_track_outputs(self, cls_id: int, header):
        """
        对指定物体的轨迹进行预测（外推）到当前时刻。
        返回 (xyz, yaw) ，其中 yaw 在 [0,π] 范围内。
        如果滤波器未初始化或超出最大预测时间，返回 (None, None)。
        """
        trk = self.tracks[cls_id]
        if not trk.xyz_kf.initialized:
            return None, None

        # 预测时间点：不能超过 max_predict_seconds
        now_t = self._msg_time_to_sec(header)
        if trk.xyz_kf.last_t is not None:
            dt_pred = float(np.clip(now_t - trk.xyz_kf.last_t, 0.0, self.max_predict_seconds))
            pred_t = float(trk.xyz_kf.last_t + dt_pred)
        else:
            pred_t = now_t

        # 预测位置
        xyz = trk.xyz_kf.predict_to(pred_t)
        if xyz is None:
            return None, None

        # 预测角度
        yaw = None
        if trk.yaw_kf.initialized:
            trk.yaw_kf.predict_to(pred_t)
            yaw = self._yaw_wrapped_to_0_pi(trk.yaw_kf.get_yaw_wrapped())

        return xyz, yaw

    def publish_cached_outputs(self):
        """
        定时器回调：发布每个物体的最新滤波/预测结果。
        如果物体在 hold_last_seconds 内没有新观测，则不再发布（视为丢失）。
        """
        # 如果还没有相机内参，无法计算3D坐标，直接返回
        if self.camera_intrinsics is None:
            return

        # 构建消息头：使用最新图像的时间戳和坐标系
        header = Header()
        if self.latest_header is not None:
            header.stamp = self.latest_header.stamp
            header.frame_id = self.latest_header.frame_id
        else:
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_color_optical_frame"

        now_wall = time.time()

        # 定义一个内部函数，发布单个物体的位置和可选RPY
        def pub_one(cls_id: int, pub_point, pub_rpy=None):
            trk = self.tracks[cls_id]
            # 如果上次更新距今超过 hold_last_seconds，则不再发布
            if (now_wall - trk.last_update_wall) > float(self.hold_last_seconds):
                return

            # 预测到当前时刻
            xyz, yaw = self._predict_track_outputs(cls_id, header)
            if xyz is None:
                return

            # 发布 PointStamped
            ps = PointStamped()
            ps.header = header
            ps.point.x = float(xyz[0])
            ps.point.y = float(xyz[1])
            ps.point.z = float(xyz[2])
            pub_point.publish(ps)

            # 如果需要发布RPY且yaw有效
            if self.publish_rpy and pub_rpy is not None and yaw is not None:
                m = Float32MultiArray()
                m.data = [0.0, 0.0, float(yaw)]   # roll=0, pitch=0, yaw
                pub_rpy.publish(m)

        # 发布三个物体
        pub_one(0, self.pub_pen_position, self.pub_pen_rpy)
        pub_one(1, self.pub_box_position, self.pub_box_rpy)
        pub_one(2, self.pub_cube_position, self.pub_cube_rpy)

    def process_images(self):
        """
        主检测循环，由定时器调用。
        执行YOLO检测、3D点提取、卡尔曼更新，并生成可视化图像。
        """
        # 防止重入（如果上一次调用还未完成，则跳过）
        if self._busy:
            return
        self._busy = True
        t0 = time.monotonic()
        infer_cost_sec = 0.0    # 推理时间初始化

        try:
            # 必须已有相机内参
            if self.camera_intrinsics is None:
                return
           
            # 从锁保护的变量中拷贝最新的图像和头信息
            with self.lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                header_src = self.latest_header

            # 无有效图像则返回
            if rgb is None or depth is None:
                return

            # 获取图像时间戳（秒）
            if header_src is not None:
                frame_t = self._msg_time_to_sec(header_src)
            else:
                frame_t = time.time()

            # ---------- YOLO 推理 ----------
            try:
                # results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, half=True,verbose=False)

                if self.device.startswith("cuda") or self.backend == "tensorrt":
                    torch.cuda.synchronize()
                t_infer0 = time.perf_counter()

                if self.backend == 'tensorrt':
                    results = self.model.predict(
                        rgb,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        verbose=False,
                        device=0,
                    )
                else:
                    results = self.model.predict(
                        rgb,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        half=True,
                        verbose=False,
                        device=self.device,
                    )

                if self.device.startswith("cuda") or self.backend == "tensorrt":
                    torch.cuda.synchronize()
                infer_cost_sec = time.perf_counter() - t_infer0     # 计算推理所花费的时间

            except Exception as e:
                self.get_logger().error(f'YOLO inference error: {e}')
                return

            # 可视化图像（在原图上绘制）
            vis = rgb.copy()

            r = results[0]
            detected_classes = set()   # 用于记录本次检测到的类别

            # 检查是否有 OBB 输出
            if not hasattr(r, 'obb') or r.obb is None or r.obb.xyxyxyxy is None:
                # 没有检测到任何定向框，发布原图并返回
                try:
                    self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
                except Exception:
                    pass
                return

            n_obb = len(r.obb.xyxyxyxy)

            # 为每个类别保留最高置信度的检测结果（因为一张图中同类物体可能只有一个，但YOLO可能输出多个重叠框）
            best_by_class = {
                0: {'conf': 0.0, 'xyz': None, 'yaw': None, 'corners': None, 'label_xy': None},
                1: {'conf': 0.0, 'xyz': None, 'yaw': None, 'corners': None, 'label_xy': None},
                2: {'conf': 0.0, 'xyz': None, 'yaw': None, 'corners': None, 'label_xy': None},
            }

            # 遍历每个检测框
            for i in range(n_obb):
                corners = try_extract_obb_corners(r, i)   # (4,2) 角点
                if corners is None:
                    continue

                cls = int(r.obb.cls[i].item())            # 类别ID
                conf = float(r.obb.conf[i].item())        # 置信度
                label = self.class_names.get(cls, f'cls{cls}')
                color = self.class_colors.get(cls, self.default_color)

                # 计算角点的中心像素坐标（用于显示标签）
                cx_pix = int(np.clip(np.mean(corners[:, 0]), 0, rgb.shape[1] - 1))
                cy_pix = int(np.clip(np.mean(corners[:, 1]), 0, rgb.shape[0] - 1))

                # 在图像上绘制 OBB 多边形和角点
                poly = corners.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(vis, [poly], True, color, 2)
                for p in corners:
                    cv2.circle(vis, tuple(map(int, p)), 2, color, -1)

                # 绘制标签和置信度
                cv2.putText(
                    vis, f'{label}:{conf:.2f}',
                    (cx_pix, max(0, cy_pix - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

                # 从深度图计算3D质心
                center3d = self._center3d_from_obb_depth(corners, depth, cls)
                if center3d is None:
                    continue

                # 计算观测 yaw (范围 [0, π])
                yaw_meas_0_pi = yaw_0_to_pi_right0_left180(corners)

                # 保留每个类别中置信度最高的检测
                if cls in best_by_class and conf > best_by_class[cls]['conf']:
                    best_by_class[cls]['conf'] = conf
                    best_by_class[cls]['xyz'] = center3d.astype(np.float64)
                    best_by_class[cls]['yaw'] = float(yaw_meas_0_pi)
                    best_by_class[cls]['corners'] = corners.copy()
                    best_by_class[cls]['label_xy'] = (cx_pix, cy_pix)

            # ---------- 更新每个物体的卡尔曼滤波器 ----------
            for cls_id, best in best_by_class.items():
                trk = self.tracks[cls_id]
                color = self.class_colors.get(cls_id, self.default_color)

                # 如果没有观测到该类物体
                if best['xyz'] is None:
                    trk.missed_count += 1
                    continue

                # 有观测，重置丢失计数
                detected_classes.add(cls_id)
                trk.missed_count = 0
                trk.last_header = header_src
                trk.last_update_wall = time.time()   # 墙上时间，用于发布超时判断

                xyz_meas = best['xyz']
                yaw_meas_0_pi = best['yaw']

                if self.use_kf:
                    # ---- 使用卡尔曼滤波 ----
                    # 更新3D位置滤波器
                    xyz_f = trk.xyz_kf.update(xyz_meas, frame_t)
                    # 将观测角度转换为连续等效角度
                    yaw_meas_equiv = self._measurement_yaw_equiv(cls_id, yaw_meas_0_pi, trk)
                    # 更新角度滤波器
                    yaw_f_wrapped = trk.yaw_kf.update(yaw_meas_equiv, frame_t)
                    # 将角度转回 [0, π] 用于显示和输出
                    yaw_f = self._yaw_wrapped_to_0_pi(yaw_f_wrapped)
                    vel_f = trk.xyz_kf.get_vel()
                else:
                    # ---- 不使用滤波，直接使用观测值（同时初始化滤波器以保持代码统一）----
                    if not trk.xyz_kf.initialized:
                        trk.xyz_kf.init_state(xyz_meas, frame_t)
                    else:
                        # 直接覆盖状态位置，保持速度不变
                        trk.xyz_kf.x[0:3, 0] = xyz_meas
                        trk.xyz_kf.last_t = frame_t
                    xyz_f = xyz_meas
                    vel_f = np.zeros(3, dtype=np.float64)
                    yaw_f = yaw_meas_0_pi

                # 保存最近观测（仅用于调试）
                trk.last_meas_xyz = xyz_meas
                trk.last_meas_yaw = yaw_meas_0_pi

                # 提取数值用于可视化
                X, Y, Z = float(xyz_f[0]), float(xyz_f[1]), float(xyz_f[2])
                yaw_deg = float(np.degrees(yaw_f))
                vx, vy, vz = float(vel_f[0]), float(vel_f[1]), float(vel_f[2])

                # 在图像上显示3D位置、yaw和速度
                cx_pix, cy_pix = best['label_xy']
                cv2.putText(
                    vis, f'X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f} m',
                    (cx_pix, min(rgb.shape[0] - 5, cy_pix + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1
                )
                cv2.putText(
                    vis, f'Yaw:{yaw_deg:.1f} deg',
                    (cx_pix, min(rgb.shape[0] - 5, cy_pix + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1
                )
                cv2.putText(
                    vis, f'Vx:{vx:.2f} Vy:{vy:.2f} Vz:{vz:.2f}',
                    (cx_pix, min(rgb.shape[0] - 5, cy_pix + 45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 120), 1
                )

            # 对于连续多帧未检测到的物体，重置其滤波器，避免旧轨迹干扰
            for cls_id, trk in self.tracks.items():
                if trk.missed_count > self.max_missed_frames:
                    trk.reset()

            # 发布可视化图像
            try:
                self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            except Exception as e:
                self.get_logger().warn(f'publish vis failed: {e}')

        finally:
            # 记录本次处理耗时，并清除 busy 标志
            det_cost_sec = time.monotonic() - t0    #该节点内部 process_images 函数执行一次所消耗的时间（秒），即纯算法计算耗时
            self._last_dt = det_cost_sec
            try:
                t_img_sec = frame_t if 'frame_t' in locals() else self._now_ros_sec()#输入图像（RGB/Depth）在传感器端产生的时间（秒）。这是延迟计算的起点
                t_det_sec = self._now_ros_sec()#当前代码执行到发布延迟统计时的时间（秒）。代表视觉处理结束的时刻
                n_det = len(detected_classes) if 'detected_classes' in locals() else 0      #本次处理周期内检测到的有效物体数量
                self._publish_vision_latency_trace(t_img_sec, t_det_sec, det_cost_sec, infer_cost_sec, n_det)
            except Exception:
                pass
            self._busy = False


# =============================================================================
# 主函数入口
# =============================================================================

def main(args=None):
    # 初始化ROS2客户端库
    rclpy.init(args=args)
    # 创建节点实例
    node = YoloDetectorNode()
    # 使用多线程执行器，允许同时处理多个回调
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        # 循环运行，直到节点被关闭
        ex.spin()
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
