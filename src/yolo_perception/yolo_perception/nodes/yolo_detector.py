#!/usr/bin/env python3
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Int32, String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from sensor_msgs.msg import RegionOfInterest

from yolo_perception_utils.model_utils import resolve_yolo_model_path


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        self.declare_parameter('model_path', 'best.pt')
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
        self.camera_intrinsics = None
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()
        self.last_print_time = time.time()
        self.print_interval = 0.5

        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 1
        )
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10
        )
        self.pub_vis = self.create_publisher(Image, '/camera/detected_image', 10)
        self.pub_detections = self.create_publisher(Detection2DArray, '/yolo_detections', 10)
        self.pub_pen_position = self.create_publisher(PointStamped, '/pen_position_3d', 10)
        self.pub_box_position = self.create_publisher(PointStamped, '/box_position_3d', 10)

        self.class_names = {0: 'pen', 1: 'box'}
        self.detection_timer = self.create_timer(0.1, self.process_images)
        self.get_logger().info('YOLO detection node started, waiting for camera info...')

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]
            }
            self.get_logger().info(
                f'Camera intrinsics: fx={self.camera_intrinsics["fx"]:.1f}, '
                f'fy={self.camera_intrinsics["fy"]:.1f}, '
                f'cx={self.camera_intrinsics["cx"]:.1f}, '
                f'cy={self.camera_intrinsics["cy"]:.1f}'
            )
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def rgb_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_rgb = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            depth_image = depth_image.astype(np.float32) / 1000.0
            with self.lock:
                self.latest_depth = depth_image
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion error: {e}")

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
        if self.camera_intrinsics is None:
            return
        rgb, depth = None, None
        with self.lock:
            if self.latest_rgb is not None and self.latest_depth is not None:
                rgb = self.latest_rgb.copy()
                depth = self.latest_depth.copy()
        if rgb is None or depth is None:
            return

        results = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
        vis = rgb.copy()
        curr_time = time.time()
        should_print = (curr_time - self.last_print_time) >= self.print_interval

        detection_array = Detection2DArray()
        detection_array.header = Header()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_color_optical_frame"

        best_pen = None
        best_box = None
        best_pen_conf = 0.0
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
                X, Y, Z = self.pixel_to_3d(center_x, center_y, depth)

                if X is not None and Y is not None and Z is not None:
                    detection = Detection2D()
                    detection.header = detection_array.header
                    detection.bbox.center.position.x = float(center_x)
                    detection.bbox.center.position.y = float(center_y)
                    detection.bbox.center.theta = 0.0
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(cls)
                    hypothesis.hypothesis.score = float(conf)
                    hypothesis.pose.pose.position.x = float(X)
                    hypothesis.pose.pose.position.y = float(Y)
                    hypothesis.pose.pose.position.z = float(Z)
                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)

                    if cls == 0 and conf > best_pen_conf:
                        best_pen = (X, Y, Z)
                        best_pen_conf = conf
                    elif cls == 1 and conf > best_box_conf:
                        best_box = (X, Y, Z)
                        best_box_conf = conf

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f'{label}: {conf:.2f}', (x1, max(0, y1-50)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(vis, f'X:{X:.2f}m Y:{Y:.2f}m', (x1, max(0, y1-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(vis, f'Z:{Z:.2f}m', (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        self.pub_detections.publish(detection_array)

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

        try:
            output_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
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
