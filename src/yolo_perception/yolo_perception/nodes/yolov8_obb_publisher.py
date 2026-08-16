#!/usr/bin/env python3

from ultralytics import YOLO
import copy
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from yolo_perception.msg import InferenceResult
from yolo_perception.msg import Yolov8Inference
from yolo_perception_utils.model_utils import (
    assign_obb_confidence,
    require_four_class_obb_model,
    resolve_yolo_model_path,
)

bridge = CvBridge()


class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.declare_parameter('model_path', 'yolo-obb-1024.pt')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('conf', 0.50)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('sync_slop', 0.02)
        self.declare_parameter('sync_watchdog_sec', 3.0)

        model_path = resolve_yolo_model_path(str(self.get_parameter('model_path').value))
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.sync_watchdog_sec = float(self.get_parameter('sync_watchdog_sec').value)
        self.model = YOLO(model_path)
        try:
            self.class_names = require_four_class_obb_model(self.model.names)
        except ValueError as exc:
            self.get_logger().fatal(str(exc))
            raise
        self.get_logger().info(f"Four-class YOLO-OBB contract accepted: {self.class_names}")

        self.yolov8_inference = Yolov8Inference()
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.depth_pub = self.create_publisher(Image, "/Yolov8_Inference/depth", 1)
        self.rgb_sub = None
        self.depth_sub = None
        self.sync = None
        self._frame_count = 0
        self._last_sync_activity = time.monotonic()
        self._start_sync()
        self.sync_watchdog = self.create_timer(1.0, self._check_sync)
        self.get_logger().info(
            f"YOLO OBB ready: rgb={self.rgb_topic}, depth={self.depth_topic}, "
            f"slop={self.sync_slop:.3f}s"
        )

    def _start_sync(self):
        for subscriber in (self.rgb_sub, self.depth_sub):
            if subscriber is not None:
                self.destroy_subscription(subscriber.sub)
        self.rgb_sub = Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.camera_callback)
        self._last_sync_activity = time.monotonic()

    def _check_sync(self):
        if time.monotonic() - self._last_sync_activity <= self.sync_watchdog_sec:
            return
        self.get_logger().warning(
            "No synchronized RGB-D frame received; recreating camera subscriptions."
        )
        self._start_sync()

    def camera_callback(self, rgb_msg, depth_msg):
        self._last_sync_activity = time.monotonic()
        img = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        results = self.model(img, conf=self.conf, imgsz=self.imgsz, verbose=False)

        self.yolov8_inference.header = rgb_msg.header

        for r in results:
            if(r.obb is not None):
                boxes = r.obb
                for box in boxes:
                    self.inference_result = InferenceResult()
                    b = box.xyxyxyxy[0].to('cpu').detach().numpy().copy()
                    c = box.cls
                    self.inference_result.class_name = self.class_names[int(c)]
                    assign_obb_confidence(self.inference_result, box)
                    a = b.reshape(1, 8)
                    self.inference_result.coordinates = copy.copy(a[0].tolist())
                    self.yolov8_inference.yolov8_inference.append(self.inference_result)

        detection_count = len(self.yolov8_inference.yolov8_inference)
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        img_msg.header = rgb_msg.header
        self.img_pub.publish(img_msg)
        self.depth_pub.publish(depth_msg)
        self._last_sync_activity = time.monotonic()
        self._frame_count += 1
        if self._frame_count == 1:
            self.get_logger().info(
                f"First synchronized inference published with "
                f"{detection_count} detections."
            )


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
