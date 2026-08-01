#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GraspNet 推理节点 (GraspnetInferenceNode)
#
# 功能：
#   - 订阅 RGB 图像、深度图像、相机内参
#   - 使用 GraspNet-baseline 模型生成抓取候选
#   - 支持自动/手动/服务触发推理
#   - 支持 Open3D 可视化确认（用户选择最优抓取或全部候选）
#   - 发布抓取姿态 PoseArray、得分 Float32MultiArray、元数据 Float32MultiArray
#   - 提供预览最佳抓取姿态的话题
#
# 依赖：
#   - ROS 2 (rclpy)
#   - cv_bridge（ROS 图像转 OpenCV）
#   - torch（PyTorch）
#   - open3d（可选，用于可视化）
#   - graspnet-baseline 代码库
# ---------------------------------------------------------------------------
import os
import sys
import threading
from typing import List, Optional, Tuple
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def _prepend_path(path: str) -> None:
    """将路径添加到 sys.path 的最前面，用于导入外部模块。"""
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _load_graspnet_modules(baseline_dir: str):
    """
    动态加载 GraspNet 相关模块（来自 graspnet-baseline 目录）。
    返回：torch, GraspNet, pred_decode, CameraInfo, create_point_cloud_from_depth_image, GraspGroup
    """
    # 将 baseline 目录及其子目录加入 Python 搜索路径
    _prepend_path(baseline_dir)
    _prepend_path(os.path.join(baseline_dir, "models"))
    _prepend_path(os.path.join(baseline_dir, "dataset"))
    _prepend_path(os.path.join(baseline_dir, "utils"))
    _prepend_path(os.path.join(baseline_dir, "graspnetAPI"))
    _prepend_path(os.path.join(baseline_dir, "pointnet2"))
    _prepend_path(os.path.join(baseline_dir, "knn"))

    import torch
    from data_utils import CameraInfo as GNCameraInfo
    from data_utils import create_point_cloud_from_depth_image
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup

    return torch, GraspNet, pred_decode, GNCameraInfo, create_point_cloud_from_depth_image, GraspGroup


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> Tuple[float, float, float, float]:
    """
    将 3x3 旋转矩阵转换为四元数 (x, y, z, w)。
    适用于 GraspNet 输出的旋转矩阵。
    """
    r = rot.astype(np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r[2, 1] - r[1, 2]) / scale
        qy = (r[0, 2] - r[2, 0]) / scale
        qz = (r[1, 0] - r[0, 1]) / scale
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / scale
        qx = 0.25 * scale
        qy = (r[0, 1] + r[1, 0]) / scale
        qz = (r[0, 2] + r[2, 0]) / scale
    elif r[1, 1] > r[2, 2]:
        scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / scale
        qx = (r[0, 1] + r[1, 0]) / scale
        qy = 0.25 * scale
        qz = (r[1, 2] + r[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / scale
        qx = (r[0, 2] + r[2, 0]) / scale
        qy = (r[1, 2] + r[2, 1]) / scale
        qz = 0.25 * scale
    return float(qx), float(qy), float(qz), float(qw)


def _vector(values, count: int, fallback: float) -> np.ndarray:
    """
    将输入转换为长度为 count 的 float32 向量，不足部分用 fallback 填充。
    用于处理 scores/widths/depths 可能缺失的情况。
    """
    out = np.full((count,), fallback, dtype=np.float32)
    if values is None:
        return out
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out[: min(count, arr.shape[0])] = arr[:count]
    return out


def _graspgroup_to_pose_metadata(grasp_group) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    """
    将 GraspGroup 对象或 numpy 数组转换为：
        - poses_np: (N, 7) 数组，每一行为 [x, y, z, qx, qy, qz, qw]
        - metadata: List of (score, width, depth) 元组
    支持 graspnetAPI 的 GraspGroup 对象和原始 numpy 数组两种格式。
    """
    # 情况 1：graspnetAPI 的 GraspGroup 对象（具有 translations 和 rotation_matrices 属性）
    if hasattr(grasp_group, "translations") and hasattr(grasp_group, "rotation_matrices"):
        translations = np.asarray(grasp_group.translations)
        rotations = np.asarray(grasp_group.rotation_matrices)
        count = int(translations.shape[0])
        scores = _vector(getattr(grasp_group, "scores", None), count, 1.0)
        widths = _vector(getattr(grasp_group, "widths", None), count, np.nan)
        depths = _vector(getattr(grasp_group, "depths", None), count, np.nan)
        poses = []
        for i in range(count):
            qx, qy, qz, qw = _rotmat_to_quat_xyzw(rotations[i])
            poses.append(
                [
                    translations[i, 0],
                    translations[i, 1],
                    translations[i, 2],
                    qx,
                    qy,
                    qz,
                    qw,
                ]
            )
        metadata = [
            (float(scores[i]), float(widths[i]), float(depths[i]))
            for i in range(count)
        ]
        return np.asarray(poses, dtype=np.float32), metadata

    # 情况 2：原始 numpy 数组（17 列格式，来自 pred_decode 输出）
    grasp_array = None
    for name in ("grasp_group_array", "grasp_group", "gg_array"):
        if hasattr(grasp_group, name):
            grasp_array = np.asarray(getattr(grasp_group, name))
            break
    if grasp_array is None:
        grasp_array = np.asarray(grasp_group)
    if grasp_array.ndim != 2 or grasp_array.shape[1] < 17:
        raise RuntimeError(f"Unexpected GraspGroup shape: {grasp_array.shape}")

    # 解析 GraspNet 标准输出格式：前 4 列为 score, width, depth, ?，接着 9 列旋转矩阵，3 列平移
    scores = grasp_array[:, 0].astype(np.float32)
    widths = grasp_array[:, 1].astype(np.float32)
    depths = grasp_array[:, 3].astype(np.float32)          # 第 2 列可能为 1，depth 在第 3 列
    rotations = grasp_array[:, 4:13].reshape(-1, 3, 3).astype(np.float32)
    translations = grasp_array[:, 13:16].astype(np.float32)
    poses = []
    for i in range(grasp_array.shape[0]):
        qx, qy, qz, qw = _rotmat_to_quat_xyzw(rotations[i])
        poses.append(
            [
                translations[i, 0],
                translations[i, 1],
                translations[i, 2],
                qx,
                qy,
                qz,
                qw,
            ]
        )
    metadata = [
        (float(scores[i]), float(widths[i]), float(depths[i]))
        for i in range(grasp_array.shape[0])
    ]
    return np.asarray(poses, dtype=np.float32), metadata


def _float_list(value, fallback: List[float]) -> List[float]:
    """安全解析逗号分隔的浮点数列表，若为空则返回 fallback。"""
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",")]
        values = [float(item) for item in parts if item]
    else:
        values = [float(item) for item in value]
    return values if values else list(fallback)


# ═══════════════════════════════════════════════════════════
#  GraspNet 推理节点
# ═══════════════════════════════════════════════════════════

class GraspnetInferenceNode(Node):
    """
    ROS2 节点：使用 GraspNet-baseline 进行 6-DOF 抓取姿态推理。

    订阅：
        - 彩色图像 (RGB)
        - 对齐后的深度图像
        - 相机内参 (CameraInfo)
    通过时间同步器（近似同步）触发推理，或通过 /grasp/compute 服务手动触发。

    发布：
        - 抓取姿态 PoseArray（/grasp/poses）
        - 得分 Float32MultiArray（/grasp/scores）
        - 元数据 Float32MultiArray（/grasp/metadata）
        - 预览最佳抓取姿态 PoseStamped
        - 预览最佳抓取得分 Float32
    """

    def __init__(self):
        super().__init__("graspnet_inference", automatically_declare_parameters_from_overrides=True)

        # 声明所有参数的默认值（如果尚未声明）
        self._declare_defaults()

        # 读取 ROS 参数
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.poses_topic = str(self.get_parameter("poses_topic").value)
        self.scores_topic = str(self.get_parameter("scores_topic").value)
        self.metadata_topic = str(self.get_parameter("metadata_topic").value)
        self.preview_best_pose_topic = str(self.get_parameter("preview_best_pose_topic").value)
        self.preview_best_score_topic = str(self.get_parameter("preview_best_score_topic").value)

        # GraspNet 模型路径与配置
        self.baseline_dir = str(self.get_parameter("baseline_dir").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.num_point = int(self.get_parameter("num_point").value)
        self.top_k_publish = max(1, int(self.get_parameter("top_k_publish").value))
        self.roi_norm = _float_list(self.get_parameter("roi_norm").value, [0.2, 0.2, 0.9, 0.85])
        self.min_valid_points = int(self.get_parameter("min_valid_points").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

        # 运行模式
        self.auto_once = bool(self.get_parameter("auto_once").value)           # 收到第一帧后自动推理一次
        self.auto_visualize = bool(self.get_parameter("auto_visualize").value) # 自动推理后是否可视化
        self.confirm_before_publish = bool(self.get_parameter("confirm_before_publish").value) # 发布前需人工确认
        self.confirm_visual_top_k = max(1, int(self.get_parameter("confirm_visual_top_k").value)) # 确认时可视化的候选数
        self.confirm_window_name = str(self.get_parameter("confirm_window_name").value)

        # 时间同步参数
        self.sync_queue = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop_s").value)

        # 随机数生成器
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))

        # 动态加载 GraspNet 模块
        modules = _load_graspnet_modules(self.baseline_dir)
        (
            self.torch,
            self.GraspNet,
            self.pred_decode,
            self.GNCameraInfo,
            self.create_point_cloud_from_depth_image,
            self.GraspGroup,
        ) = modules

        # 设置设备（优先 GPU）
        self.device = self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")
        self.net = self._load_net()

        # CV Bridge（ROS ↔ OpenCV 图像转换）
        self.bridge = CvBridge()

        # 线程锁：保护共享数据
        self._lock = threading.Lock()
        self._compute_lock = threading.Lock()  # 防止并发推理
        self._latest: Optional[Tuple[Image, Image, CameraInfo]] = None  # 最近一次同步数据
        self._auto_started = False  # 自动推理是否已启动

        # 发布者 QoS 配置（可靠，保留最新 5 条）
        out_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pose_pub = self.create_publisher(PoseArray, self.poses_topic, out_qos)
        self.score_pub = self.create_publisher(Float32MultiArray, self.scores_topic, out_qos)
        self.metadata_pub = self.create_publisher(Float32MultiArray, self.metadata_topic, out_qos)
        self.preview_pose_pub = self.create_publisher(PoseStamped, self.preview_best_pose_topic, out_qos)
        self.preview_score_pub = self.create_publisher(Float32, self.preview_best_score_topic, out_qos)

        # 服务：/grasp/compute
        self.create_service(Trigger, "/grasp/compute", self.on_compute)

        # 订阅 RGB、深度、相机内参，并创建近似时间同步器
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.info_sub = Subscriber(self, CameraInfo, self.info_topic, qos_profile=qos_profile_sensor_data)
        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
        )
        self.ats.registerCallback(self.on_synced)

        # 输出初始化信息
        self.get_logger().info(f"GraspNet checkpoint: {self.checkpoint_path}")
        self.get_logger().info(f"RGB/Depth/Info: {self.rgb_topic}, {self.depth_topic}, {self.info_topic}")
        self.get_logger().info(f"ROI norm: {self.roi_norm}, top_k_publish={self.top_k_publish}")
        self.get_logger().info(
            f"Confirm before publish={self.confirm_before_publish}, "
            f"confirm_visual_top_k={self.confirm_visual_top_k}"
        )

    def _declare_defaults(self):
        """声明所有可配置参数的默认值（仅当参数尚未声明时）。"""
        home = os.path.expanduser("~")
        defaults = {
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "camera_frame": "camera_color_optical_frame",
            "poses_topic": "/grasp/poses",
            "scores_topic": "/grasp/scores",
            "metadata_topic": "/grasp/metadata",
            "preview_best_pose_topic": "/graspnet_grasping/preview_best_pose",
            "preview_best_score_topic": "/graspnet_grasping/preview_best_score",
            "baseline_dir": os.path.join(home, "manipulator_grasp", "graspnet-baseline"),
            "checkpoint_path": os.path.join(home, "manipulator_grasp", "logs", "log_rs", "checkpoint-rs.tar"),
            "num_point": 20000,
            "top_k_publish": 5,
            "roi_norm": [0.2, 0.2, 0.9, 0.85],  # 归一化 ROI 区域 [x_min, y_min, x_max, y_max]
            "min_valid_points": 2000,
            "depth_min_m": 0.05,
            "depth_max_m": 5.0,
            "sync_queue_size": 10,
            "sync_slop_s": 0.05,
            "auto_once": False,
            "auto_visualize": False,
            "confirm_before_publish": False,
            "confirm_visual_top_k": 50,
            "confirm_window_name": "GraspNet candidates: SPACE=execute, S=best, ESC/Q=cancel",
            "random_seed": 0,
        }
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

    def _load_net(self):
        """加载 GraspNet 模型权重并设置为 eval 模式。"""
        net = self.GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(self.device)
        checkpoint = self.torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    # ═══════════════════════════════════════════════════════
    #  消息同步回调
    # ═══════════════════════════════════════════════════════

    def on_synced(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        """时间同步回调：缓存最新的同步数据；若 auto_once 启用且未开始，则后台启动推理。"""
        with self._lock:
            self._latest = (rgb_msg, depth_msg, info_msg)
        if self.auto_once and not self._auto_started:
            self._auto_started = True
            threading.Thread(target=self._run_once, daemon=True).start()

    # ═══════════════════════════════════════════════════════
    #  服务回调
    # ═══════════════════════════════════════════════════════

    def on_compute(self, _req: Trigger.Request, resp: Trigger.Response):
        """
        服务 /grasp/compute 的回调：执行一次推理并发布结果。
        若已有推理在运行，返回失败。
        """
        if not self._compute_lock.acquire(blocking=False):
            resp.success = False
            resp.message = "GraspNet inference is already running."
            return resp
        try:
            with self._lock:
                latest = self._latest
            if latest is None:
                resp.success = False
                resp.message = "No synchronized RGB/Depth/CameraInfo received yet."
                return resp
            # 执行推理并发布（不进行可视化）
            count = self._infer_and_publish(*latest, visual=False)
            resp.success = True
            resp.message = f"Published {count} GraspNet grasp candidates."
            return resp
        except Exception as exc:
            resp.success = False
            resp.message = f"Inference failed: {exc}"
            self.get_logger().error(resp.message)
            return resp
        finally:
            self._compute_lock.release()

    def _run_once(self):
        """自动推理一次（由 auto_once 参数触发）。"""
        try:
            with self._lock:
                latest = self._latest
            if latest is not None:
                self._infer_and_publish(*latest, visual=self.auto_visualize)
        except Exception as exc:
            self.get_logger().error(f"Auto inference failed: {exc}")

    # ═══════════════════════════════════════════════════════
    #  核心推理与发布流程
    # ═══════════════════════════════════════════════════════

    def _infer_and_publish(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo, visual: bool) -> int:
        """
        执行完整的抓取推理流水线：
            1. 将 ROS 图像转为 numpy
            2. 调用 _generate_grasps 获得所有抓取和采样点云
            3. 根据配置进行可视化确认
            4. 转换数据并发布
        返回发布的抓取数量。
        """
        rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        grasp_all, grasp_pub, cloud_points, cloud_colors = self._generate_grasps(
            np.asarray(rgb_bgr),
            np.asarray(depth_raw),
            info_msg,
        )
        # 确定坐标系和时间戳
        frame_id = info_msg.header.frame_id or depth_msg.header.frame_id or self.camera_frame
        stamp = rgb_msg.header.stamp

        if self.confirm_before_publish:
            # 需要用户确认（仅当确认通过才继续）
            if not self._confirm_grasps(grasp_all, cloud_points, cloud_colors, frame_id, stamp):
                raise RuntimeError("Grasp confirmation canceled by user.")
        elif visual:
            # 直接可视化（不阻塞发布）
            self._visualize(grasp_pub, cloud_points, cloud_colors)

        # 将 GraspGroup 转换为姿态和元数据
        poses_np, metadata = _graspgroup_to_pose_metadata(grasp_pub)
        self._publish_results(poses_np, metadata, frame_id, stamp)
        return int(poses_np.shape[0])

    def _generate_grasps(self, rgb_bgr: np.ndarray, depth_raw: np.ndarray, info: CameraInfo):
        """
        GraspNet 抓取生成核心逻辑：
            - 深度预处理（单位转换、ROI 过滤）
            - 创建点云
            - 随机采样
            - 前向推理
            - 解码、NMS、排序
        返回: (全部抓取 GraspGroup, 发布的 top-k GraspGroup, ROI 点云, 点云颜色)
        """
        # 深度图转换为米（假设原始为毫米的 uint16）
        depth_m = depth_raw.astype(np.float32) / 1000.0 if depth_raw.dtype == np.uint16 else depth_raw.astype(np.float32)
        height, width = depth_m.shape[:2]
        # RGB 转为 [0,1] 浮点数
        color = rgb_bgr[..., ::-1].astype(np.float32) / 255.0

        # 解析相机内参
        fx, fy, cx, cy = float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError("Invalid camera intrinsics: fx/fy is zero.")
        camera = self.GNCameraInfo(width, height, fx, fy, cx, cy, 1.0)
        # 生成有组织的点云
        cloud_org = self.create_point_cloud_from_depth_image(depth_m, camera, organized=True)

        # ROI 与有效深度范围掩码
        mask = self._roi_mask(depth_m)
        valid = int(mask.sum())
        self.get_logger().info(f"GraspNet valid ROI points={valid}, total={mask.size}")
        if valid < self.min_valid_points:
            raise RuntimeError(f"Too few valid points in ROI: {valid}")

        # 提取 ROI 内的点云和颜色
        cloud_masked = cloud_org[mask]
        color_masked = color[mask]
        # 随机采样固定数量的点
        if len(cloud_masked) >= self.num_point:
            indices = self.rng.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            base = np.arange(len(cloud_masked))
            extra = self.rng.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            indices = np.concatenate([base, extra], axis=0)

        cloud_sampled = cloud_masked[indices]
        color_sampled = color_masked[indices]

        # 构建输入 batch
        end_points = {
            "point_clouds": self.torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device),
            "cloud_colors": color_sampled,
        }
        # 前向推理
        with self.torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self.pred_decode(end_points)

        grasp_array = grasp_preds[0].detach().cpu().numpy()
        if grasp_array.size == 0:
            raise RuntimeError("pred_decode produced no grasps.")
        # 创建 GraspGroup，NMS 过滤并按得分排序
        grasp_group = self.GraspGroup(grasp_array)
        grasp_group.nms()
        grasp_group.sort_by_score()
        # 取前 top_k 个抓取用于发布
        grasp_pub = grasp_group[: self.top_k_publish]
        return grasp_group, grasp_pub, cloud_masked, color_masked

    def _roi_mask(self, depth_m: np.ndarray) -> np.ndarray:
        """
        根据归一化 ROI 范围 [x_min, y_min, x_max, y_max] 和深度范围
        生成二值掩码。
        """
        height, width = depth_m.shape[:2]
        x_min, y_min, x_max, y_max = self.roi_norm
        # 归一化坐标转像素坐标
        x0 = int(round(np.clip(x_min, 0.0, 1.0) * width))
        y0 = int(round(np.clip(y_min, 0.0, 1.0) * height))
        x1 = int(round(np.clip(x_max, 0.0, 1.0) * width))
        y1 = int(round(np.clip(y_max, 0.0, 1.0) * height))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"Invalid ROI bounds: {self.roi_norm}")
        roi = np.zeros_like(depth_m, dtype=bool)
        roi[y0:y1, x0:x1] = True
        # 深度范围过滤
        depth_mask = (depth_m > self.depth_min_m) & (depth_m < self.depth_max_m)
        return roi & depth_mask

    # ═══════════════════════════════════════════════════════
    #  可视化（Open3D）
    # ═══════════════════════════════════════════════════════

    def _visualize(self, grasp_group, cloud_points: np.ndarray, cloud_colors: np.ndarray):
        """使用 Open3D 显示点云和抓取候选。"""
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(cloud_colors.astype(np.float32))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        grippers = grasp_group.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, frame, *grippers])

    def _confirm_grasps(self, grasp_group, cloud_points: np.ndarray, cloud_colors: np.ndarray, frame_id: str, stamp) -> bool:
        """
        交互式确认窗口：
            - 显示点云、坐标系和若干抓取候选
            - 按键操作：
                空格: 确认当前所有候选并发布
                S: 只显示最佳抓取，并发布预览姿态
                ESC/Q/关闭窗口: 取消
        返回 True 表示用户确认发布，False 表示取消。
        """
        if len(grasp_group) == 0:
            raise RuntimeError("No GraspNet grasps available for confirmation.")

        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError(f"Open3D is required for grasp confirmation: {exc}") from exc

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(cloud_colors.astype(np.float32))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # 显示前 confirm_visual_top_k 个抓取，最佳抓取用绿色高亮
        grasp_vis = grasp_group[: self.confirm_visual_top_k]
        candidate_grippers = grasp_vis[1:].to_open3d_geometry_list() if len(grasp_vis) > 1 else []
        best_gripper = grasp_vis[0].to_open3d_geometry(color=(0.0, 1.0, 0.0))

        accepted = {"value": None}  # 用字典存储用户决定（True/False/None）

        # 按键回调
        def accept(vis):
            accepted["value"] = True
            vis.close()
            return False

        def cancel(vis):
            accepted["value"] = False
            vis.close()
            return False

        def show_best_only(vis):
            # 移除除最佳以外的所有抓取模型
            for gripper in candidate_grippers:
                vis.remove_geometry(gripper, reset_bounding_box=False)
            vis.update_renderer()
            # 发布最佳抓取的预览姿态
            self._publish_preview_best_pose(grasp_group[:1], frame_id, stamp)
            self.get_logger().info(
                "Showing best GraspNet grasp only and published preview pose. "
                "Press SPACE to execute or ESC/Q to cancel."
            )
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name=self.confirm_window_name, width=1280, height=720):
            raise RuntimeError("Failed to create Open3D confirmation window. Check DISPLAY/GUI access.")

        try:
            vis.add_geometry(cloud)
            vis.add_geometry(frame)
            for gripper in candidate_grippers:
                vis.add_geometry(gripper)
            vis.add_geometry(best_gripper)
            # 注册按键
            vis.register_key_callback(ord(" "), accept)   # 空格
            vis.register_key_callback(ord("S"), show_best_only)
            vis.register_key_callback(ord("s"), show_best_only)
            vis.register_key_callback(ord("Q"), cancel)
            vis.register_key_callback(ord("q"), cancel)
            vis.register_key_callback(256, cancel)        # ESC
            self.get_logger().info(
                "Grasp confirmation window opened. Press S to show best grasp only; "
                "press SPACE to execute; press ESC/Q or close window to cancel."
            )
            vis.run()
        finally:
            vis.destroy_window()

        return bool(accepted["value"])

    # ═══════════════════════════════════════════════════════
    #  发布函数
    # ═══════════════════════════════════════════════════════

    def _publish_preview_best_pose(self, grasp_group, frame_id: str, stamp):
        """发布最佳抓取姿态的预览（单个 PoseStamped 和得分）。"""
        poses_np, metadata = _graspgroup_to_pose_metadata(grasp_group)
        if poses_np.shape[0] == 0:
            return
        score = float(metadata[0][0]) if metadata else float("nan")
        score_msg = Float32()
        score_msg.data = score
        self.preview_score_pub.publish(score_msg)

        x, y, z, qx, qy, qz, qw = [float(value) for value in poses_np[0].tolist()]
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.preview_pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published best GraspNet preview pose score={score:.4f} frame={frame_id}")

    def _publish_results(
        self,
        poses_np: np.ndarray,
        metadata: List[Tuple[float, float, float]],
        frame_id: str,
        stamp,
    ):
        """发布抓取姿态、得分和元数据（宽度、深度）。"""
        # 构建 PoseArray
        poses_msg = PoseArray()
        poses_msg.header.frame_id = frame_id
        poses_msg.header.stamp = stamp
        for row in poses_np:
            x, y, z, qx, qy, qz, qw = [float(value) for value in row.tolist()]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            poses_msg.poses.append(pose)

        count = len(metadata)
        # 得分
        scores_msg = Float32MultiArray()
        scores_msg.data = [float(row[0]) for row in metadata]

        # 元数据（每行 3 个值：score, width, depth）
        metadata_msg = Float32MultiArray()
        metadata_msg.layout.dim = [
            MultiArrayDimension(label="grasp", size=count, stride=count * 3),
            MultiArrayDimension(label="field", size=3, stride=3),
        ]
        metadata_msg.data = [float(value) for row in metadata for value in row]

        self.score_pub.publish(scores_msg)
        self.metadata_pub.publish(metadata_msg)
        self.pose_pub.publish(poses_msg)


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = GraspnetInferenceNode()
    try:
        rclpy.spin(node)  # 阻塞直到节点退出
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
