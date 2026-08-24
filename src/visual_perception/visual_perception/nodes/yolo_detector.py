import cv2
import numpy as np
import threading
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from visual_perception_utils.model_utils import resolve_yolo_model_path
from visual_perception_utils.visualization import draw_detection_center


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        self.declare_parameter('model_path', 'best.pt')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('conf', 0.3)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.02)

        model_path = resolve_yolo_model_path(self.get_parameter('model_path').get_parameter_value().string_value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        self.get_logger().info(f'Loading YOLOv8 model: {model_path}  device={self.device}')
        self.model = YOLO(model_path)
        if self.device != 'cpu':
            try:
                self.model.to(self.device)
            except Exception as e:
                self.get_logger().warn(f'Could not move model to {self.device}: {e}')

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self._camera_info_signature = None
        self._camera_info_stable_count = 0
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_header = None
        self.lock = threading.Lock()

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data
        )
        self.rgb_sub = Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], self.sync_queue_size, self.sync_slop, allow_headerless=False
        )
        self.sync.registerCallback(self.synced_rgb_depth_callback)
        self.pub_vis = self.create_publisher(Image, '/camera/detected_result', 10)
        self.pub_detections = self.create_publisher(Detection2DArray, '/yolo_detections', 10)
        self.pub_elongated_object_position = self.create_publisher(
            PointStamped, '/elongated_object_position_3d', 10
        )
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', 10)

        self.class_names = {int(class_id): str(name) for class_id, name in self.model.names.items()}
        self.detection_timer = self.create_timer(0.1, self.process_images)
        self.get_logger().info(
            f'YOLO detection node started: rgb={self.rgb_topic}, depth={self.depth_topic}, '
            f'camera_info={self.camera_info_topic}'
        )

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is not None:
            return

        intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'width': msg.width,
            'height': msg.height,
            'frame_id': msg.header.frame_id,
        }
        signature = tuple(intrinsics.values())
        if signature == self._camera_info_signature:
            self._camera_info_stable_count += 1
        else:
            self._camera_info_signature = signature
            self._camera_info_stable_count = 1

        if self._camera_info_stable_count < 3:
            return

        self.camera_intrinsics = intrinsics
        if self.camera_info_sub is not None:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None
        self.get_logger().info(
            'CameraInfo locked after 3 stable frames: '
            f"topic={self.camera_info_topic} size={intrinsics['width']}x{intrinsics['height']} "
            f"frame={intrinsics['frame_id']!r} fx={intrinsics['fx']:.1f}, "
            f"fy={intrinsics['fy']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}"
        )

    def synced_rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
        except CvBridgeError as e:
            self.get_logger().error(f'RGB-D conversion error: {e}')
            return

        if depth_msg.encoding in ('16UC1', 'mono16'):
            depth /= 1000.0
        depth = np.nan_to_num(np.minimum(depth, 20.0), nan=0.0, posinf=20.0, neginf=0.0)
        with self.lock:
            self.latest_rgb = rgb
            self.latest_depth = depth
            self.latest_header = rgb_msg.header

    def pixel_to_3d(self, x, y, depth):
        if self.camera_intrinsics is None:
            return None, None, None
        if y >= depth.shape[0] or x >= depth.shape[1]:
            return None, None, None
        Z = depth[y, x]
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
        rgb, depth = None, None
        with self.lock:
            if self.latest_rgb is not None and self.latest_depth is not None:
                rgb = self.latest_rgb.copy()
                depth = self.latest_depth.copy()
                source_header = self.latest_header
        if rgb is None or depth is None:
            return

        results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
        vis = rgb.copy()

        detection_array = Detection2DArray()
        detection_array.header = Header()
        detection_array.header.stamp = (
            source_header.stamp if source_header is not None else self.get_clock().now().to_msg()
        )
        detection_array.header.frame_id = (
            self.camera_intrinsics['frame_id'] if self.camera_intrinsics is not None
            else (source_header.frame_id if source_header is not None else 'camera_color_optical_frame')
        )

        best_elongated_object = None
        best_box = None
        best_elongated_object_conf = 0.0
        best_box_conf = 0.0

        r = results[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0].item())
                cls = int(b.cls[0].item()) if b.cls is not None else -1
                if conf < self.conf:
                    continue

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                label = self.class_names.get(cls, "other")
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f'{label}: {conf:.2f}', (x1, max(0, y1-50)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                draw_detection_center(vis, (center_x, center_y))

                detection = Detection2D()
                detection.header = detection_array.header
                detection.bbox.center.position.x = float(center_x)
                detection.bbox.center.position.y = float(center_y)
                detection.bbox.center.theta = 0.0
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = label
                hypothesis.hypothesis.score = float(conf)
                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

                X, Y, Z = self.pixel_to_3d(center_x, center_y, depth)
                if X is not None and Y is not None and Z is not None:
                    hypothesis.pose.pose.position.x = float(X)
                    hypothesis.pose.pose.position.y = float(Y)
                    hypothesis.pose.pose.position.z = float(Z)

                    if cls == 0 and conf > best_elongated_object_conf:
                        best_elongated_object = (X, Y, Z)
                        best_elongated_object_conf = conf
                    elif cls == 1 and conf > best_box_conf:
                        best_box = (X, Y, Z)
                        best_box_conf = conf

                    cv2.putText(vis, f'X:{X:.2f}m Y:{Y:.2f}m', (x1, max(0, y1-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(vis, f'Z:{Z:.2f}m', (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        self.pub_detections.publish(detection_array)

        if best_elongated_object is not None:
            elongated_object_point = PointStamped()
            elongated_object_point.header = detection_array.header
            elongated_object_point.point.x = float(best_elongated_object[0])
            elongated_object_point.point.y = float(best_elongated_object[1])
            elongated_object_point.point.z = float(best_elongated_object[2])
            self.pub_elongated_object_position.publish(elongated_object_point)

        if best_box is not None:
            box_point = PointStamped()
            box_point.header = detection_array.header
            box_point.point.x = float(best_box[0])
            box_point.point.y = float(best_box[1])
            box_point.point.z = float(best_box[2])
            self.pub_box_position.publish(box_point)

        try:
            output_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            if source_header is not None:
                output_msg.header = source_header
            self.pub_vis.publish(output_msg)
        except Exception as e:
            self.get_logger().debug(f'Failed to publish image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Program exception: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
