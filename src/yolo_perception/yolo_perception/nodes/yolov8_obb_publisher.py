#!/usr/bin/env python3

from ultralytics import YOLO
import copy
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from yolo_perception.msg import InferenceResult
from yolo_perception.msg import Yolov8Inference
from yolo_perception_utils.model_utils import resolve_yolo_model_path

bridge = CvBridge()


class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.declare_parameter('model_path', 'yolo-obb-gazebo-1024.pt')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('conf', 0.50)
        self.declare_parameter('rgb_topic', '/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('sync_slop', 0.02)

        model_path = resolve_yolo_model_path(str(self.get_parameter('model_path').value))
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.model = YOLO(model_path)

        self.yolov8_inference = Yolov8Inference()

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=float(self.get_parameter('sync_slop').value),
            allow_headerless=False,
        )
        self.sync.registerCallback(self.camera_callback)

        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.depth_pub = self.create_publisher(Image, "/Yolov8_Inference/depth", 1)

    def camera_callback(self, rgb_msg, depth_msg):
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
                    self.inference_result.class_name = self.model.names[int(c)]
                    a = b.reshape(1, 8)
                    self.inference_result.coordinates = copy.copy(a[0].tolist())
                    self.yolov8_inference.yolov8_inference.append(self.inference_result)

        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        img_msg.header = rgb_msg.header
        self.img_pub.publish(img_msg)
        self.depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
