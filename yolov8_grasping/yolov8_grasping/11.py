#!/usr/bin/env python3
import rclpy                                      # ROS2 Python 客户端库
from rclpy.node import Node                       # ROS2 节点基类
from sensor_msgs.msg import Image, CameraInfo     # 相机图像与相机内参消息
from geometry_msgs.msg import PointStamped        # 带时间戳与坐标系的点
from std_msgs.msg import Header                   # 标准消息头（时间戳、frame_id）
from cv_bridge import CvBridge, CvBridgeError     # ROS Image 与 OpenCV 图像互转
from ultralytics import YOLO                      # Ultralytics YOLO 推理接口（支持 OBB）
import cv2                                        # OpenCV 图像处理库
import numpy as np                                # 数值计算库
import threading                                  # 线程锁
import time                                       # 时间函数

from std_msgs.msg import Float32MultiArray        # 发布欧拉角 RPY 的消息类型（弧度）
from vision_msgs.msg import Detection2DArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

# -------------------------------------------------------------

# ========== 线性代数/姿态工具 ==========

def _rotmat_to_rpy_zyx(R: np.ndarray):
    """从 3x3 旋转矩阵分解欧拉角 roll、pitch、yaw（ZYX 顺序，单位：弧度）。"""
    assert R.shape == (3, 3), "R must be 3x3"
    yaw_raw = np.arctan2(R[1, 0], R[0, 0])
    # yaw = (yaw_raw + np.pi) % np.pi
    yaw = yaw_raw % np.pi
    sp  = -R[2, 0]
    cp  = np.sqrt(R[2, 1]**2 + R[2, 2]**2)
    pitch = np.arctan2(sp, cp)
    roll  = np.arctan2(R[2, 1], R[2, 2])
    # 如果大于π/2，映射到[0, π/2]
    if yaw > np.pi/2:
        yaw= np.pi - yaw
    
    # 确保非负
    yaw = max(0.0, yaw)
    return float(roll), float(pitch), float(yaw)

def _normalize(v: np.ndarray, eps: float = 1e-12):
    """向量归一化（范数过小则原样返回以避免除零）。"""
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n



def _try_extract_obb_corners(result, i_det):
    """从 YOLOv8 OBB 结果里拿四角点；"""
    obb = getattr(result, 'obb', None)
    if obb is None:
        return None
    # 多边形顶点
    if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > i_det:
        try:
            arr = obb.xyxyxyxy[i_det].detach().cpu().numpy().reshape(-1, 2)
            if arr.shape[0] >= 4:
                return arr[:4]
        except Exception:
            pass
    return None

# ========== 节点 ==========
class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector_obb')

        # ===== 参数设置 =====
        # self.declare_parameter('model_path','/home/robot/S622_robotarm/yolo_obb.pt')  # OBB 权重
        # self.declare_parameter('model_path','/home/robot/S622_robotarm/yolo-obb1.pt') 
        # self.declare_parameter('model_path','/home/robot/S622_robotarm/yolo-obb2.pt') 
        self.declare_parameter('model_path','/home/robot/S622_robotarm/yolo-obb3.pt') 
        self.declare_parameter('device', 'auto')     # 'cpu' 或 'cuda:0'
        self.declare_parameter('conf', 0.5)        # 置信度阈值（OBB 建议 0.2~0.5）
        self.declare_parameter('imgsz', 640)        # 推理输入尺寸
        # PCA 点云估计的参数
        self.declare_parameter('pca_stride', 3)          # ROI 抽样步长（像素）
        self.declare_parameter('pca_max_points', 5000)   # ROI 最多点数
        self.declare_parameter('depth_max_range', 10.0)   # 深度最大距离（米）
        self.declare_parameter('publish_rpy', True)      # 是否发布 RPY 话题
        self.class_colors = {0: (0, 255, 0), 1: (255, 0, 0)}     
        self.default_color = (0, 255, 255)  # 黄色

        # 读取参数
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf   = float(self.get_parameter('conf').value)
        self.imgsz  = int(self.get_parameter('imgsz').value)
        self.pca_stride      = int(self.get_parameter('pca_stride').value)
        self.pca_max_points  = int(self.get_parameter('pca_max_points').value)
        self.depth_max_range = float(self.get_parameter('depth_max_range').value)
        self.publish_rpy     = bool(self.get_parameter('publish_rpy').value)

        # 加载模型
        self.get_logger().info(f'Loading YOLOv8 model: {model_path}  device={self.device}')
        self.model = YOLO(model_path)
        if self.device != 'cpu':
            try:
                self.model.to(self.device)
            except Exception as e:
                self.get_logger().warn(f'Could not move model to {self.device}: {e}')

        self.bridge = CvBridge()

        # ===== 相机内参（自动获取）=====
        self.camera_intrinsics = None  # {'fx','fy','cx','cy'}

        # ===== 图像缓存 =====
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()



        # ===== 订阅 =====
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info
', self.camera_info_callback, 10)

        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)

        # self.depth_sub = self.create_subscription(
        #     Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)


        # ===== 发布 =====
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', 10)
        self.pub_pen_position = self.create_publisher(PointStamped, '/pen_position_3d', 10)
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', 10)
        self.pub_cube_position = self.create_publisher(PointStamped, '/cube_position_3d', 10)
        self.pub_pen_rpy = self.create_publisher(Float32MultiArray, '/pen_rpy', 10) if self.publish_rpy else None
        self.pub_box_rpy = self.create_publisher(Float32MultiArray, '/box_rpy', 10) if self.publish_rpy else None
        self.pub_cube_rpy = self.create_publisher(Float32MultiArray, '/cube_rpy', 10) if self.publish_rpy else None

        # ===== 类别映射（按你的 classes.txt）=====
        self.class_names = {0: 'pen', 1: 'box' , 2: 'cube'}

        # 定时器：10Hz
        self.detection_timer = self.create_timer(0.1, self.process_images)

        self.get_logger().info('YOLO检测节点启动，等待相机内参信息...')

    # -------------------- 回调们 --------------------
    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
            self.get_logger().info(
                f'相机内参已获取: fx={self.camera_intrinsics["fx"]:.1f}, '
                f'fy={self.camera_intrinsics["fy"]:.1f}, '
                f'cx={self.camera_intrinsics["cx"]:.1f}, '
                f'cy={self.camera_intrinsics["cy"]:.1f}'
            )
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    # RGB图像回调
    def rgb_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_rgb = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"RGB转换错误: {e}")

    #深度图像回调
    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            depth_image = depth_image.astype(np.float32)
            if depth_image.max() > 20.0:  # D435: 16UC1 毫米
                depth_image = depth_image / 1000.0
            with self.lock:
                self.latest_depth = depth_image
        except CvBridgeError as e:
            self.get_logger().error(f"深度图像转换错误: {e}")


    # -------------------- 核心：OBB+深度 -> 姿态 --------------------
    def _pose_from_obb_and_depth(self, poly_2d: np.ndarray, depth: np.ndarray):
        #获取深度图的高度和宽度,depth.shape = (480, 640),D435相机分辨率
        H, W = depth.shape[:2]

        #创建OBB区域的掩码,创建一个与深度图同尺寸的空白掩码（全零矩阵）
        mask = np.zeros((H, W), dtype=np.uint8)
        #将OBB多边形内部的像素设为255（白色），外部保持0（黑色）
        cv2.fillPoly(mask, [poly_2d.astype(np.int32)], 255)
        #         掩码示意图：
        # ┌──────────────────────┐
        # │ 0  0  0  0  0  0  0 │
        # │ 0  0 [255 255]  0  0│ ← OBB区域填充为255
        # │ 0  0 [255 255]  0  0│
        # │ 0  0  0  0  0  0  0 │
        # └──────────────────────┘

        #获取掩码内的像素坐标,找到掩码中所有非零（即白色）像素的坐标
        ys, xs = np.where(mask > 0)
        if xs.size < 10:           # # 如果OBB内的像素少于50个，返回失败
            return None, None
        
        #获取相机内参,fx = fy ≈ 615.0,焦距;cx ≈ 320.0,图像宽度/2;cy ≈ 240.0,图像高度/2
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']

        #在掩码区域内采样3D点
        pts = []
        stride = max(1, self.pca_stride)#采样步长（每隔几个像素取一个点）
        max_pts = max(200, self.pca_max_points)#最大点数（避免点云过大）
        count = 0
        for u, v in zip(xs[::stride], ys[::stride]):     # 按步长遍历像素
            # 原始点（stride=1）：
            # xs = [320, 321, 322, 323, 324, ...]
            # ys = [200, 200, 200, 200, 200, ...]

            # 采样后（stride=3）：
            # xs[::3] = [320, 323, 326, ...]  ← 每隔3个像素取一个
            # ys[::3] = [200, 200, 200, ...]

            Z = float(depth[v, u])                       # 获取深度值,u 是横坐标（宽）v 是纵坐标（高）
            if np.isfinite(Z) and 0.0 < Z <= self.depth_max_range:  #过滤无效深度值,0<=z<=5m
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                
                pts.append([X, Y, Z]) #将有效的3D点加入列表，达到最大点数后停止
                count += 1
                if count >= max_pts:        # 达到最大点数限制
                    break

        if len(pts) < 10:                   #有效3D点不足,说明深度数据质量差，放弃姿态估计
            return None, None

        #主成分分析(PCA)计算姿态
        P = np.asarray(pts, dtype=np.float32)    # 点云数据
        center = np.mean(P, axis=0)              # 计算点云中心
        Pc = P - center[None, :]                 # 中心化点云

        # 计算协方差矩阵
        C = (Pc.T @ Pc) / max(1, (Pc.shape[0] - 1))

        # 特征值分解
        eigvals, eigvecs = np.linalg.eigh(C)
        # 按特征值大小排序
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # 构建旋转矩阵
        x_axis = _normalize(eigvecs[:, 0])
        y_axis = _normalize(eigvecs[:, 1])
        z_axis = _normalize(np.cross(x_axis, y_axis))
        y_axis = _normalize(np.cross(z_axis, x_axis))

        # 构建3×3旋转矩阵
        R = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0
        return R, center.astype(np.float32)  # 返回旋转矩阵和中心位置

    def process_images(self):
        # 1. 检查相机内参是否就绪
        if self.camera_intrinsics is None:
            return
        
        # 2. 线程安全地获取最新图像
        with self.lock:
            rgb   = None if self.latest_rgb   is None else self.latest_rgb.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()

        if rgb is None or depth is None:
            return

        # 3. YOLOv8-OBB推理
        try:
            # 使用YOLO模型进行有向边界框检测
            results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
        except Exception as e:
            self.get_logger().error(f'YOLO推理错误: {e}')
            return

        # 4. 初始化可视化图像和消息
        vis = rgb.copy()     # 创建用于可视化的图像副本
        detection_array = Detection2DArray() # ROS2检测结果消息
        detection_array.header = Header()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_color_optical_frame"

        best_pen = None;  best_pen_conf = 0.0;  best_pen_rpy = None
        best_box = None;  best_box_conf = 0.0;  best_box_rpy = None
        best_cube = None; best_cube_conf = 0.0; best_cube_rpy = None

        # 5. 处理每个检测结果
        r = results[0]

        # 如果模型没有输出 OBB，直接返回
        if not hasattr(r, 'obb') or r.obb is None:
            try:
                self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
            except Exception as e:
                self.get_logger().warn(f'发布图像失败: {e}')
            return

        # 统计 OBB 数量
        if hasattr(r.obb, 'xyxyxyxy') and r.obb.xyxyxyxy is not None:
            n_obb = len(r.obb.xyxyxyxy)


        # 遍历所有 OBB
        for i in range(n_obb):
            obb_corners = _try_extract_obb_corners(r, i)
            if obb_corners is None:
                continue

            # 获取类别和置信度
            cls = int(r.obb.cls[i].item()) #类别ID (0: pen, 1: box)
            conf = float(r.obb.conf[i].item()) # 检测置信度
            label = self.class_names.get(cls, f'cls{cls}') # 类别名称

            # 6. 在图像上绘制检测结果
            color = self.class_colors.get(cls, self.default_color)

            # 计算OBB中心点（像素坐标）
            cx_pix = int(np.clip(np.mean(obb_corners[:, 0]), 0, rgb.shape[1]-1))
            cy_pix = int(np.clip(np.mean(obb_corners[:, 1]), 0, rgb.shape[0]-1))

            # 绘制OBB多边形边界
            poly = obb_corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(vis, [poly], True, color, 2)

            # 绘制角点
            for p in obb_corners:
                cv2.circle(vis, tuple(map(int, p)), 2, color, -1)
            # 添加标签文本
            cv2.putText(vis, f'{label}:{conf:.2f}', (cx_pix, max(0, cy_pix-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # 7. 姿态估计
            R, t_center = self._pose_from_obb_and_depth(obb_corners, depth)


            if (R is not None) and (t_center is not None):
                # 提取3D位置
                X, Y, Z = float(t_center[0]), float(t_center[1]), float(t_center[2])
                # 计算欧拉角和四元数
                roll, pitch, yaw = _rotmat_to_rpy_zyx(R)

                # 在图像上标注3D信息
                cv2.putText(vis, f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} m',
                            (cx_pix, min(rgb.shape[0]-5, cy_pix+15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
                cv2.putText(vis, f'R:{np.degrees(roll):.1f} P:{np.degrees(pitch):.1f} Y:{np.degrees(yaw):.1f}',
                            (cx_pix, min(rgb.shape[0]-5, cy_pix+30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

                # 绘制主方向指示线
                axis_len_px = 20
                dir_cam_x = R[:, 0]
                dx = int(axis_len_px * dir_cam_x[0])
                dy = int(axis_len_px * dir_cam_x[1])
                cv2.line(vis, (cx_pix - dx, cy_pix - dy), (cx_pix + dx, cy_pix + dy), (0, 0, 255), 2)

                # 记录最佳
                if cls == 0 and conf > best_pen_conf:
                    best_pen, best_pen_conf, best_pen_rpy = (X, Y, Z), conf, (roll, pitch, yaw)
                if cls == 1 and conf > best_box_conf:
                    best_box, best_box_conf, best_box_rpy = (X, Y, Z), conf, (roll, pitch, yaw)
                if cls == 2 and conf > best_cube_conf:
                    best_cube, best_cube_conf, best_cube_rpy = (X, Y, Z), conf, (roll, pitch, yaw)


            else:
                self.get_logger().warn(f'PCA失败: {label} at ({cx_pix},{cy_pix})')
                continue  # 直接跳过

        # 发布 pen / box / cube best
        if best_pen is not None:
            ps = PointStamped(); 
            ps.header = detection_array.header
            ps.point.x, ps.point.y, ps.point.z = map(float, best_pen)
            self.pub_pen_position.publish(ps)
            if self.publish_rpy and (best_pen_rpy is not None):
                m = Float32MultiArray(); 
                m.data = list(best_pen_rpy)
                self.pub_pen_rpy.publish(m)

        if best_box is not None:
            ps = PointStamped(); 
            ps.header = detection_array.header
            ps.point.x, ps.point.y, ps.point.z = map(float, best_box)
            self.pub_box_position.publish(ps)
            if self.publish_rpy and (best_box_rpy is not None):
                m = Float32MultiArray(); 
                m.data = list(best_box_rpy)
                self.pub_box_rpy.publish(m)

        if best_cube is not None:
            ps = PointStamped()
            ps.header = detection_array.header
            ps.point.x, ps.point.y, ps.point.z = map(float, best_cube)
            self.pub_cube_position.publish(ps)
            if self.publish_rpy and (best_cube_rpy is not None):
                m = Float32MultiArray()
                m.data = list(best_cube_rpy)
                self.pub_cube_rpy.publish(m)

        # 发布可视化
        try:
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')) # 发布可视化图像
        except Exception as e:
            self.get_logger().warn(f'发布图像失败: {e}')


# -------------------- main --------------------
def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"程序异常: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()