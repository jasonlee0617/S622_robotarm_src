#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, ReliabilityPolicy, HistoryPolicy, QoSProfile,DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped,PoseStamped
from std_msgs.msg import Header, Float32MultiArray
import torch
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import time
import cv2
import numpy as np
import threading
import math
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
# -----------------------------
# Angle helpers
# -----------------------------

def wrap_to_pi(a: float) -> float:  # 包装后的角度，范围[-π, π]
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def angle_diff(a: float, b: float) -> float:  #计算两个角度之间的最小差值
    return wrap_to_pi(a - b)


def choose_equivalent_angle(cur: float, prev: float, period: float) -> float:
    """
    参数：
    cur: 当前测量角度（已包装到[-π, π]）
    prev: 前一个角度
    period: 周期（pen: π, box/cube: π/2）
    
    返回：
    最接近prev的等价角度
    """

    best = cur
    best_err = abs(angle_diff(cur, prev))
    # 尝试多个周期偏移（-4到4个周期）
    for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        cand = cur + k * period # 生成候选角度
        err = abs(angle_diff(cand, prev)) # 计算与prev的差值
        if err < best_err:
            best_err = err
            best = cand
    return wrap_to_pi(best)


def yaw_0_to_pi_right0_left180(corners_2d: np.ndarray) -> float:
    """
    从OBB角点计算偏航角（范围：0到π）
    
    物理意义：
    0°: 物体右侧朝右（最长的边水平向右）
    90°: 物体右侧朝下
    180°: 物体右侧朝左
    
    参数：
    corners_2d: 4×2数组，OBB的四个角点坐标
    
    返回：
    偏航角（弧度），范围[0, π]
    """
    c = corners_2d.astype(np.float32)
    best_v = None
    best_len = -1.0
    # 遍历四条边，找出最长的边
    for i in range(4):
        v = c[(i + 1) % 4] - c[i]  # 计算第i条边向量：从角点i到角点(i+1)%4
        L = float(np.linalg.norm(v)) # 计算边的长度（L2范数）
        # 更新最长边
        if L > best_len:
            best_len = L
            best_v = v

    if best_v is None or best_len < 1e-6:
        return 0.0

    dx, dy = float(best_v[0]), float(best_v[1])    # 提取边向量的x,y分量
    yaw = math.atan2(abs(dy), dx)  # [0, pi]， 计算角度：使用atan2(|dy|, dx)将角度限制在[0, π]
    yaw = max(0.0, min(math.pi, yaw)) # 确保角度在[0, π]范围内
    return float(yaw)


# -----------------------------
# OBB extraction
# -----------------------------

def try_extract_obb_corners(result, i_det):
    obb = getattr(result, 'obb', None)
    if obb is None:
        return None
    if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > i_det:
        try:
            arr = obb.xyxyxyxy[i_det].detach().cpu().numpy().reshape(-1, 2)
            if arr.shape[0] >= 4:
                return arr[:4]
        except Exception:
            pass
    return None


# -----------------------------
# Node
# -----------------------------

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector_yaw_0_180')

        # Params
        self.declare_parameter('model_path', '/home/robot/S622_robotarm/yolo-obb-gazebo.pt')
        self.declare_parameter('device', 'auto')  
        self.declare_parameter('conf', 0.2)
        self.declare_parameter('imgsz', 640)
        
        self.declare_parameter('depth_max_range', 10.0)
        self.declare_parameter('publish_rpy', True)

        self.declare_parameter('stride_pen', 5)
        self.declare_parameter('stride_box', 10)
        self.declare_parameter('stride_cube', 1)
        self.declare_parameter('max_points', 5000)
        self.declare_parameter('min_points_pen', 20)
        self.declare_parameter('min_points_box', 200)
        self.declare_parameter('min_points_cube', 50)

        self.declare_parameter('inference_period', 0.033)    # YOLO 推理周期
        self.declare_parameter('pose_publish_rate', 30.0)   # 输出发布频率
        self.declare_parameter('hold_last_seconds', 0.15)   # 超时不再发布

        self.declare_parameter('yaw_smoothing_alpha', 0.3)
        self.declare_parameter('xyz_smoothing_alpha', 0.8)

        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')


        self.declare_parameter('sync_queue_size', 10)  #RGB+Depth 对齐匹配缓存长度
        self.declare_parameter('sync_slop', 0.02)  # RGB 和 Depth容忍时间差

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
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
        self.alpha = float(self.get_parameter('yaw_smoothing_alpha').value)
        self.xyz_alpha = float(self.get_parameter('xyz_smoothing_alpha').value)

        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        self.inference_period = float(self.get_parameter('inference_period').value)
        self.pose_publish_rate = float(self.get_parameter('pose_publish_rate').value)
        self.hold_last_seconds = float(self.get_parameter('hold_last_seconds').value)
        # Load model
        # 自动选择设备
        if self.device.lower() == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda:0'  # 或 'cuda'
                self.get_logger().info('CUDA is available, using GPU.')
            else:
                self.device = 'cpu'
                self.get_logger().info('CUDA not available, using CPU.')
        else:
            self.device = self.device  # 使用用户指定的值

        # 加载模型
        self.get_logger().info(f'Loading YOLOv8 model: {model_path}, device={self.device}')
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.bridge = CvBridge()

        self.camera_intrinsics = None
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()
        
        # prev yaw per class (store in [0,pi])
        self.prev_yaw = {0: None, 1: None, 2: None}
        self.latest_header = None
        self.prev_xyz = {0: None, 1: None, 2: None}     
        self.last_best_xyz = {0: None, 1: None, 2: None}
        self.last_best_rpy = {0: None, 1: None, 2: None}
        self.last_update_wall = {0: 0.0, 1: 0.0, 2: 0.0}

        self.cb_infer = MutuallyExclusiveCallbackGroup()
        self.cb_pub = MutuallyExclusiveCallbackGroup()

        # Subscribers
        self.camera_info_sub = self.create_subscription(CameraInfo,self.camera_info_topic,self.camera_info_callback,10)

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],queue_size=self.sync_queue_size,slop=self.sync_slop,allow_headerless=False)
        self.sync.registerCallback(self.synced_rgb_depth_callback)
        qos_reliable_latest = QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)#控制发布者/订阅者的通信策略
        #history=HistoryPolicy.KEEP_LAST(历史策略：告诉 DDS（ROS2 底层通信）如何保存消息历史)
        #队列深度（缓存条数,depth=1 表示：只保留最新消息
        #durability=DurabilityPolicy.VOLATILE,订阅者刚启动时，只会收到启动之后发布的消息
        # Publishers
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', qos_reliable_latest)

        self.pub_pen_position = self.create_publisher(PointStamped, '/pen_position_3d', qos_reliable_latest)
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', qos_reliable_latest)
        self.pub_cube_position = self.create_publisher(PointStamped, '/cube_position_3d', qos_reliable_latest)

        self.pub_pen_rpy = self.create_publisher(Float32MultiArray, '/pen_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_box_rpy = self.create_publisher(Float32MultiArray, '/box_rpy', qos_reliable_latest) if self.publish_rpy else None
        self.pub_cube_rpy = self.create_publisher(Float32MultiArray, '/cube_rpy', qos_reliable_latest) if self.publish_rpy else None

        self.class_names = {0: 'pen', 1: 'box', 2: 'cube'}
        self.class_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255)}
        self.default_color = (0, 255, 255)

        self._busy = False
        self._last_dt = 0.0
        # self.detection_timer = self.create_timer(0.1, self.process_images)
        # self.detection_timer = self.create_timer(self.inference_period, self.process_images)
        # self.publish_timer = self.create_timer(1.0/max(1.0, self.pose_publish_rate), self.publish_cached_outputs)
        self.detection_timer = self.create_timer(
            self.inference_period, self.process_images, callback_group=self.cb_infer
        )
        self.publish_timer = self.create_timer(1.0/max(1.0, self.pose_publish_rate), self.publish_cached_outputs, callback_group=self.cb_pub)


    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
            self.get_logger().info(f'Camera intrinsics: fx={self.camera_intrinsics["fx"]:.1f}, 'f'fy={self.camera_intrinsics["fy"]:.1f}, 'f'cx={self.camera_intrinsics["cx"]:.1f}, 'f'cy={self.camera_intrinsics["cy"]:.1f}')
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def synced_rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        """
        只处理同步后的 RGB+Depth：
        - 只做：转换 -> 同一把锁内同时写 latest_rgb/latest_depth
        """
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB convert error: {e}")
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_image = depth_image.astype(np.float32) / 1000.0
            else:
                depth_image = depth_image.astype(np.float32)


            # 深度处理逻辑：仍然 /1000 转米 + 截断 + nan_to_num
            depth_image[depth_image > 20.0] = 20.0
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=20.0, neginf=0.0)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth convert error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Depth processing error: {e}")
            return

        # 原子更新：同一把锁同时写入，确保 RGB/Depth 永远成对
        with self.lock:
            self.latest_rgb = rgb
            self.latest_depth = depth_image
            self.latest_header = rgb_msg.header   # ✅ 用图像时间戳
    
    #   从OBB多边形和深度图像计算3D中心点
    def _center3d_from_obb_depth(self, poly_2d: np.ndarray, depth: np.ndarray,cls: int):
        # 获取图像尺寸
        H, W = depth.shape[:2]
        # ========================
        # 1. 创建OBB掩膜
        # ========================
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_2d.astype(np.int32)], 255) # 将多边形填充为白色（255）
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # ========================
        # 2. 获取多边形内的像素坐标
        # ========================
        ys, xs = np.where(mask > 0)# 掩膜内所有像素的坐标
        if xs.size < 100:# 检查是否有足够多的像素
            return None
        # ========================
        # 3. 提取相机内参
        # ========================
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        # ========================
        # 4. 采样深度点并转换为3D点
        # ========================
        # stride = max(1, self.stride) # 采样步长，避免处理过多点
        if cls == 0:      # pen
            stride = max(1, self.stride_pen)
            min_points = self.min_points_pen
        elif cls == 1:    # box
            stride = max(1, self.stride_box)
            min_points = self.min_points_box
        elif cls == 2:    # cube
            stride = max(1, self.stride_cube)
            min_points = self.min_points_cube
        else:
            stride = 1
            min_points = 50

        pts = []# 存储3D点
        count = 0
        for u, v in zip(xs[::stride], ys[::stride]):  # 遍历掩膜内的像素（按步长采样）
            Z = float(depth[v, u]) # 获取深度值
            if np.isfinite(Z) and 0.0 < Z <= self.depth_max_range: # 检查深度值的有效性
                # 从像素坐标和深度计算3D坐标
                X = (float(u) - cx) * Z / fx
                Y = (float(v) - cy) * Z / fy
                pts.append([X, Y, Z])
                count += 1
                # 达到最大点数限制时停止
                if count >= self.max_points:
                    break
        # ========================
        # 5. 检查是否有足够多的有效点
        # ========================
        if len(pts) < min_points:
            return None
        # ========================
        # 6. 计算所有3D点的质心
        # ========================
        P = np.asarray(pts, dtype=np.float32) # N×3数组
        return np.mean(P, axis=0) # 计算均值

    # ===== XYZ时序滤波 =====
    def smooth_xyz(self, cls: int, xyz_raw: np.ndarray) -> np.ndarray:
        """
        对XYZ坐标进行指数平滑滤波
        
        公式：xyz_smooth = xyz_prev + α * (xyz_raw - xyz_prev)
        α越小，滤波越强（但响应越慢）
        """
        prev = self.prev_xyz.get(cls, None)
        
        if prev is None:
            # 第一帧直接使用
            xyz_out = xyz_raw
        else:
            # 指数平滑
            xyz_out = prev + self.xyz_alpha * (xyz_raw - prev)
        
        # 更新历史
        self.prev_xyz[cls] = xyz_out
        return xyz_out
    
    def _estimate_yaw_0_pi(self, cls: int, corners: np.ndarray) -> float:
        """
        估计偏航角（0到π范围）并进行平滑处理
        """
        # 1. 从角点计算偏航角
        yaw_meas = yaw_0_to_pi_right0_left180(corners)  # [0, pi]
        # 2. 获取前一个角度
        prev = self.prev_yaw.get(cls, None)
        # 3. 确定角度周期
        # pen: 周期π（180°），因为笔有方向性
        # box/cube: 周期π/2（90°），因为正方形旋转90°后看起来相同
        if cls in (1, 2):   # box, cube
            period = (math.pi / 2.0)
        else:               # pen
            period = math.pi
        # 4. 处理第一个测量值
        if prev is None:
            yaw_out = yaw_meas
        else:

            # 5. 将当前测量值转换到等价表示范围
            # 由于yaw_meas在[0,π]，而choose_equivalent_angle期望[-π,π]
            meas_rep = yaw_meas
            if meas_rep > (math.pi / 2.0):
                meas_rep = meas_rep - math.pi  # (-pi/2, pi/2]

            prev_rep = prev
            if prev_rep > (math.pi / 2.0):
                prev_rep = prev_rep - math.pi
            # 6. 选择最接近前一个角度的等价角度
            yaw_eq = choose_equivalent_angle(meas_rep, prev_rep, period=period)
            # 7. 计算角度差并应用低通滤波
            diff = angle_diff(yaw_eq, prev_rep)
            yaw_smooth_rep = wrap_to_pi(prev_rep + self.alpha * diff)

            # 8. 转换回[0,π]范围
            yaw_out = yaw_smooth_rep
            if yaw_out < 0.0:
                yaw_out += math.pi
        # 9. 确保角度在[0,π]范围内
        yaw_out = max(0.0, min(math.pi, float(yaw_out)))
        # 10. 存储当前角度供下次使用
        self.prev_yaw[cls] = yaw_out
        return yaw_out
    def publish_cached_outputs(self):
        if self.camera_intrinsics is None:
            return

        # 用最新同步图像的 header（stamp + frame_id）保证 TF 对齐
        header = Header()
        if self.latest_header is not None:
            header.stamp = self.latest_header.stamp
            header.frame_id = self.latest_header.frame_id
        else:
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_color_optical_frame"

        now_wall = time.time()

        def pub_one(cls_id: int, pub_point, pub_rpy=None):
            xyz = self.last_best_xyz.get(cls_id, None)
            rpy = self.last_best_rpy.get(cls_id, None)
            t_upd = float(self.last_update_wall.get(cls_id, 0.0))
            if xyz is None:
                return
            if (now_wall - t_upd) > float(self.hold_last_seconds):
                return

            ps = PointStamped()
            ps.header = header
            ps.point.x, ps.point.y, ps.point.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            pub_point.publish(ps)

            if self.publish_rpy and pub_rpy is not None:
                m = Float32MultiArray()
                m.data = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
                pub_rpy.publish(m)

        pub_one(0, self.pub_pen_position, self.pub_pen_rpy)
        pub_one(1, self.pub_box_position, self.pub_box_rpy)
        pub_one(2, self.pub_cube_position, self.pub_cube_rpy)
    
    def process_images(self):
        """
        主处理函数
        执行检测、计算3D位置和姿态、发布结果
        """
        if self._busy:
            return
        self._busy = True
        t0 = time.monotonic()
        try:
            # self.get_logger().info(f"rgb={rgb.shape}, depth={depth.shape}")
            # ========================
            # 1. 检查相机内参是否就绪
            # ========================
            if self.camera_intrinsics is None:
                return
            # ========================
            # 2. 获取最新图像（线程安全）
            # ========================
            with self.lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
            # 检查图像是否有效
            if rgb is None or depth is None:
                return
            # ========================
            # 3. 运行YOLO推理
            # ========================
            try:
                # 使用YOLO模型进行预测
                # conf: 置信度阈值
                # imgsz: 输入图像尺寸（YOLO会自动resize）
                # verbose: 是否显示详细输出
                results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
            except Exception as e:
                self.get_logger().error(f'YOLO inference error: {e}')
                return
            # ========================
            # 4. 准备可视化图像和消息头
            # ========================
            vis = rgb.copy()# 用于可视化的图像副本
            header = Header()
            # header.stamp = self.get_clock().now().to_msg()# 当前时间戳
            # header.frame_id = "camera_color_optical_frame"# 坐标系
            if self.latest_header is not None:
                header.stamp = self.latest_header.stamp
                header.frame_id = self.latest_header.frame_id
            else:
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "camera_color_optical_frame"
            # ========================
            # 5. 初始化最佳检测结果
            # ========================
            # 每个类别只发布最高置信度的检测结果
            best_pen = None; best_pen_conf = 0.0; best_pen_rpy = None
            best_box = None; best_box_conf = 0.0; best_box_rpy = None
            best_cube = None; best_cube_conf = 0.0; best_cube_rpy = None

            r = results[0]  # 获取第一个（也是唯一一个）结果
            if not hasattr(r, 'obb') or r.obb is None or r.obb.xyxyxyxy is None:      # 检查是否有OBB检测结果
                try:
                    self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))   # 没有检测到物体，只发布可视化图像
                except Exception:
                    pass
                return
            # 获取OBB数量
            n_obb = len(r.obb.xyxyxyxy)
            # 遍历所有检测结果
            for i in range(n_obb):
                # 6.1 提取OBB角点
                corners = try_extract_obb_corners(r, i)
                if corners is None:
                    continue
                # 6.2 获取类别和置信度
                cls = int(r.obb.cls[i].item())# 类别ID
                conf = float(r.obb.conf[i].item()) # 置信度
                label = self.class_names.get(cls, f'cls{cls}') # 类别名称
                color = self.class_colors.get(cls, self.default_color) # 可视化颜色
                # 6.3 计算OBB中心（图像坐标）
                cx_pix = int(np.clip(np.mean(corners[:, 0]), 0, rgb.shape[1] - 1))
                cy_pix = int(np.clip(np.mean(corners[:, 1]), 0, rgb.shape[0] - 1))
                # 6.4 在图像上绘制OBB
                poly = corners.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(vis, [poly], True, color, 2)
                # 绘制角点
                for p in corners:
                    cv2.circle(vis, tuple(map(int, p)), 2, color, -1)
                # 绘制标签和置信度
                cv2.putText(vis, f'{label}:{conf:.2f}', (cx_pix, max(0, cy_pix - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # 6.5 计算3D中心点
                center3d = self._center3d_from_obb_depth(corners, depth,cls)
                if center3d is None:
                    continue
                # X, Y, Z = float(center3d[0]), float(center3d[1]), float(center3d[2])

                # ===== XYZ时序滤波 =====
                center3d_smooth = self.smooth_xyz(cls, center3d)
                X, Y, Z = float(center3d_smooth[0]), float(center3d_smooth[1]), float(center3d_smooth[2])

                # 6.6 计算偏航角
                yaw = self._estimate_yaw_0_pi(cls, corners)

                roll = 0.0
                pitch = 0.0
                # 6.8 在图像上显示3D位置和姿态信息
                cv2.putText(vis, f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} m',
                            (cx_pix, min(rgb.shape[0] - 5, cy_pix + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(vis, f'Yaw:[0,180]={np.degrees(yaw):.1f} deg',
                            (cx_pix, min(rgb.shape[0] - 5, cy_pix + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                # 6.9 更新最佳检测结果
                if cls == 0 and conf > best_pen_conf:
                    best_pen_conf = conf
                    best_pen = (X, Y, Z)
                    best_pen_rpy = (roll, pitch, yaw)

                if cls == 1 and conf > best_box_conf:
                    best_box_conf = conf
                    best_box = (X, Y, Z)
                    best_box_rpy = (roll, pitch, yaw)

                if cls == 2 and conf > best_cube_conf:
                    best_cube_conf = conf
                    best_cube = (X, Y, Z)
                    best_cube_rpy = (roll, pitch, yaw)
            # ========================
            # 7. 发布最佳检测结果
            # ========================
            # 7.1 发布pen的3D位置
            now_wall = time.time()
            if best_pen is not None:
                self.last_best_xyz[0] = np.array(best_pen, dtype=float)
                self.last_best_rpy[0] = best_pen_rpy
                self.last_update_wall[0] = now_wall
            # 7.2 发布box的3D位置
            if best_box is not None:

                self.last_best_xyz[1] = np.array(best_box, dtype=float)
                self.last_best_rpy[1] = best_box_rpy
                self.last_update_wall[1] = now_wall
            # 7.3 发布cube的3D位置
            if best_cube is not None:
               
                self.last_best_xyz[2] = np.array(best_cube, dtype=float)
                self.last_best_rpy[2] = best_cube_rpy
                self.last_update_wall[2] = now_wall
            # ========================
            # 8. 发布可视化图像
            # ========================
            try:
                self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            except Exception as e:
                self.get_logger().warn(f'publish vis failed: {e}')
        finally:
            self._last_dt = time.monotonic() - t0
            self._busy = False

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        # rclpy.spin(node)
        ex.spin()
    # except KeyboardInterrupt:
    #     pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



