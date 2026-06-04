from pathlib import Path
#!/usr/bin/env python3
"""
独立的YOLO检测节点
发布检测到的pen和box的位置信息
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped     #带时间戳与坐标系的点
from std_msgs.msg import Header                #标准消息头（时间戳、frame_id）
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import threading
import time

# 自定义消息类型
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Int32, String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from sensor_msgs.msg import RegionOfInterest



def resolve_yolo_model_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(get_package_share_directory("yolo_model")) / path)


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # ===== 参数设置 =====
        self.declare_parameter('model_path', 'best.pt')  #YOLO 权重路径（默认指向 best.pt）
        
        self.declare_parameter('device', 'auto')
        self.declare_parameter('conf', 0.3)
        self.declare_parameter('imgsz', 640)

        model_path = resolve_yolo_model_path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.get_logger().info(f'Loading YOLOv8 model: {model_path}  device={self.device}')
        self.model = YOLO(model_path)
        if self.device != 'cpu':
            try:
                self.model.to(self.device)
            except Exception as e:
                self.get_logger().warn(f'Could not move model to {self.device}: {e}')

        self.bridge = CvBridge()

        # ===== 相机内参（自动获取）=====
        self.camera_intrinsics = None
        
        # ===== 图像数据存储 =====
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()

        # ===== 时间控制 =====
        self.last_print_time = time.time()
        self.print_interval = 0.5

        # ===== 订阅相机内参 =====
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 1
        )

        # ===== 订阅RGB和深度图像 =====
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10
        )
        # ===== 发布检测结果 =====
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', 10)  #发布画好框的可视化图（sensor_msgs/Image）
        self.pub_detections = self.create_publisher(Detection2DArray, '/yolo_detections', 10) #发布全部检测结果（vision_msgs/Detection2DArray）
        self.pub_pen_position = self.create_publisher(PointStamped, '/pen_position_3d', 10)  #发布“最好（最大置信度）”的笔的 3D 位置（PointStamped）
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', 10)  #发布“最好”的盒子的 3D 位置（PointStamped）

        # ===== 类别映射 =====
        self.class_names = {0: 'pen', 1: 'box'}

        # ===== 创建检测处理定时器 =====
        self.detection_timer = self.create_timer(0.1, self.process_images)
        
        self.get_logger().info('YOLO检测节点启动，等待相机内参信息...')

    def camera_info_callback(self, msg: CameraInfo):
        """相机内参回调函数"""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],    
                'fy': msg.k[4],            
                'cx': msg.k[2],
                'cy': msg.k[5]
            }  #K 是 3×3 展平数组 [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            
            self.get_logger().info(
                f'相机内参已获取: fx={self.camera_intrinsics["fx"]:.1f}, '
                f'fy={self.camera_intrinsics["fy"]:.1f}, '
                f'cx={self.camera_intrinsics["cx"]:.1f}, '
                f'cy={self.camera_intrinsics["cy"]:.1f}'
            )
            
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def rgb_callback(self, msg: Image):   #将 ROS Image 转为 OpenCV BGR8 格式
        """RGB图像回调函数"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_rgb = cv_image  #用线程锁保护共享数据，将最新图像存入 self.latest_rgb
        except CvBridgeError as e:
            self.get_logger().error(f"RGB转换错误: {e}")

    def depth_callback(self, msg: Image):
        """深度图像回调函数"""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")  #深度图用 "passthrough" 保持原始编码（比如 16UC1/32FC1），转为 NumPy
            depth_image = depth_image.astype(np.float32) / 1000.0 #将深度值转为 float32 并除以 1000,单位毫米mm
            with self.lock:
                self.latest_depth = depth_image #写入 self.latest_depth（线程安全）
        except CvBridgeError as e:
            self.get_logger().error(f"深度图像转换错误: {e}")

    def pixel_to_3d(self, x, y, depth): #像素 → 相机坐标系 3D
        """将像素坐标转换为相机坐标系下的3D坐标"""
        if self.camera_intrinsics is None:
            return None, None, None   #若内参未就绪，无法计算，返回空
            
        if y >= depth.shape[0] or x >= depth.shape[1]:
            return None, None, None #越界检查：像素坐标必须在深度图范围内
        
        Z = depth[y, x]
        # if Z <= 0 or Z > 2.0:   
        if Z <= 0:
            return None, None, None
            
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy'] 
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        return X, Y, Z

    def process_images(self):
        """处理RGB和深度图像"""
        if self.camera_intrinsics is None:
            return   #没拿到内参不处理
            
        rgb, depth = None, None
        with self.lock:
            if self.latest_rgb is not None and self.latest_depth is not None:
                rgb = self.latest_rgb.copy()
                depth = self.latest_depth.copy()   #带锁读取最近的 RGB 与深度
        
        if rgb is None or depth is None:
            return    #任一为空则不处理

        # YOLO推理
        results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)  #对 OpenCV BGR 图像直接推理，使用设定的置信度阈值与输入尺寸，关闭 verbose

        # 处理检测结果
        vis = rgb.copy()   #为可视化准备一份图像副本 vis
        curr_time = time.time()
        should_print = (curr_time - self.last_print_time) >= self.print_interval  
        
        # 创建检测结果消息
        detection_array = Detection2DArray()  #新建 Detection2DArray，填充 Header
        detection_array.header = Header()
        detection_array.header.stamp = self.get_clock().now().to_msg()  #时间戳取当前 ROS2 时钟
        detection_array.header.frame_id = "camera_color_optical_frame"  #frame_id 设置为 "camera_color_optical_frame"

        best_pen = None
        best_box = None
        best_pen_conf = 0.0
        best_box_conf = 0.0

        r = results[0]  #results 为一个列表（通常单张图像长度为 1），取第 0 个结果 r
        if hasattr(r, 'boxes') and r.boxes is not None:  #判断是否存在 boxes
            for b in r.boxes:   #遍历每个检测框 b
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()) #b.xyxy 是左上右下坐标（张量），取第 0 行，转 list，再转 int
                conf = float(b.conf[0].item())                #b.conf：置信度
                cls = int(b.cls[0].item()) if b.cls is not None else -1

                if conf < self.conf:  #二次基于阈值过滤
                    continue

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  #计算框中心像素坐标，用于取深度与 3D 反投影
                label = self.class_names.get(cls, "other")           #根据 cls 查找中文标签（pen/box），找不到标记为 "other"
                
                # 计算3D坐标
                X, Y, Z = self.pixel_to_3d(center_x, center_y, depth)
                
                if X is not None and Y is not None and Z is not None:
                    # 创建Detection2D消息
                    detection = Detection2D()
                    detection.header = detection_array.header
                    
                    # 设置边界框 
                    detection.bbox.center.position.x = float(center_x)
                    detection.bbox.center.position.y = float(center_y)
                    detection.bbox.center.theta = 0.0  # Pose2D 需要 theta 字段
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    
                    # 设置检测结果
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(cls) #hypothesis.class_id：类别 ID
                    hypothesis.hypothesis.score = float(conf) #hypothesis.score：置信度
                    hypothesis.pose.pose.position.x = float(X)
                    hypothesis.pose.pose.position.y = float(Y)
                    hypothesis.pose.pose.position.z = float(Z)
                    detection.results.append(hypothesis)
                    
                    detection_array.detections.append(detection)
                    
                    # 跟踪最佳检测结果
                    if cls == 0 and conf > best_pen_conf:  # pen
                        best_pen = (X, Y, Z)
                        best_pen_conf = conf
                    elif cls == 1 and conf > best_box_conf:  # box
                        best_box = (X, Y, Z)
                        best_box_conf = conf
                    
                    # 绘制检测框和信息
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f'{label}: {conf:.2f}', (x1, max(0, y1-50)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(vis, f'X:{X:.2f}m Y:{Y:.2f}m', (x1, max(0, y1-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(vis, f'Z:{Z:.2f}m', (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 发布检测结果
        self.pub_detections.publish(detection_array)
        
        # 发布最佳pen和box位置
        if best_pen is not None:
            pen_point = PointStamped()
            pen_point.header = detection_array.header
            pen_point.point.x = float(best_pen[0])
            pen_point.point.y = float(best_pen[1])
            pen_point.point.z = float(best_pen[2])
            self.pub_pen_position.publish(pen_point)
        
        if best_box is not None:
            box_point = PointStamped()
            box_point.header = detection_array.header
            box_point.point.x = float(best_box[0])
            box_point.point.y = float(best_box[1])
            box_point.point.z = float(best_box[2])
            self.pub_box_position.publish(box_point)

        # 发布可视化图像
        try:
            output_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            self.pub_vis.publish(output_msg)
        except Exception as e:
            self.get_logger().debug(f'发布图像失败: {e}')


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








