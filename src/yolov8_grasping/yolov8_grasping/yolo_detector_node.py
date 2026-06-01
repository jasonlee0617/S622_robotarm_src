import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2



class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # ===== 参数 =====
        self.declare_parameter('model_path', '/home/robot/S622_robotarm/best.pt')  # 改成你路径
        self.declare_parameter('device', 'auto')     # 'cpu' 或 'cuda:0'
        self.declare_parameter('conf', 0.5)        # 置信度阈值
        self.declare_parameter('imgsz', 640)        # 推理分辨率（越小越快）

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
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

        # ===== 订阅压缩图像 =====
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # 订阅压缩图像
        # self.sub = self.create_subscription(
        #     Image, '/oak/rgb/image_raw', self.image_cb, sensor_qos
        # )
        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_cb, sensor_qos
        )

        # 发布带检测框的图像
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', 10)

        self.get_logger().info('YOLOv8 detector ready. Subscribing: camera/camera/color/image_raw -> Publishing: /camera/detected_image')

        # 类别映射（box 和 pen）
        self.class_names = {0: 'pen', 1: 'box'}

    def image_cb(self, msg: Image):
        try:
            # 将压缩图像解码为 OpenCV 图像
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # 推理（YOLO）
        results = self.model.predict(frame)

        # 绘制检测框
        vis = frame
        r = results[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0].item())
                cls = int(b.cls[0].item()) if b.cls is not None else -1

                # 显示类别名称
                label = self.class_names.get(cls, "other")  # 如果不属于 box/pen 类别，标记为 "other"
                
                self.get_logger().info(f"Detected {label} with confidence {conf:.2f}")

                # 绘制框和标签
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f'{label}: {conf:.2f}', (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 将处理后的图像转换为 ROS 图像消息
        output_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')

        # 发布处理后的图像
        self.pub_vis.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

