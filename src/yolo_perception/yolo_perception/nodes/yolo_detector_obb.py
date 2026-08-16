#!/usr/bin/env python3
import math
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, Header
from ultralytics import YOLO

from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth
from yolo_perception_utils.model_utils import (
    FOUR_CLASS_OBB_NAMES,
    POSITION_3D_TOPICS,
    RPY_TOPICS,
    require_four_class_obb_model,
    resolve_yolo_model_path,
)
from yolo_perception_utils.obb_geometry import (
    angle_diff,
    choose_equivalent_angle,
    try_extract_obb_corners,
    wrap_to_pi,
    yaw_0_to_pi_right0_left180,
)
from yolo_perception_utils.visualization import draw_detection_center


class YoloDetectorObbNode(Node):
    """Strict four-class RGB-D OBB detector used by visual_grasping_gazebo."""

    def __init__(self):
        super().__init__("yolov8_detector_yaw_0_180")
        self.declare_parameter("model_path", "yolo-obb-1024.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("imgsz", 1024)
        self.declare_parameter("depth_max_range", 10.0)
        self.declare_parameter("publish_rpy", True)
        self.declare_parameter("stride_elongated_object", 5)
        self.declare_parameter("stride_box", 1)
        self.declare_parameter("stride_cube", 1)
        self.declare_parameter("max_points", 5000)
        self.declare_parameter("min_points_elongated_object", 20)
        self.declare_parameter("min_points_box", 200)
        self.declare_parameter("min_points_cube", 50)
        self.declare_parameter("depth_inlier_m", 0.08)
        self.declare_parameter("depth_mad_scale", 3.0)
        self.declare_parameter("min_depth_inlier_ratio", 0.6)
        self.declare_parameter("inference_period", 0.033)
        self.declare_parameter("pose_publish_rate", 30.0)
        self.declare_parameter("hold_last_seconds", 0.15)
        self.declare_parameter("yaw_smoothing_alpha", 0.3)
        self.declare_parameter("xyz_smoothing_alpha", 0.8)
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.02)

        model_path = resolve_yolo_model_path(str(self.get_parameter("model_path").value))
        self.device = str(self.get_parameter("device").value)
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.depth_max_range = float(self.get_parameter("depth_max_range").value)
        self.publish_rpy = bool(self.get_parameter("publish_rpy").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.depth_inlier_m = max(0.001, float(self.get_parameter("depth_inlier_m").value))
        self.depth_mad_scale = max(0.0, float(self.get_parameter("depth_mad_scale").value))
        self.min_depth_inlier_ratio = min(1.0, max(0.0, float(self.get_parameter("min_depth_inlier_ratio").value)))
        self.alpha = float(self.get_parameter("yaw_smoothing_alpha").value)
        self.xyz_alpha = float(self.get_parameter("xyz_smoothing_alpha").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.inference_period = float(self.get_parameter("inference_period").value)
        self.pose_publish_rate = float(self.get_parameter("pose_publish_rate").value)
        self.hold_last_seconds = float(self.get_parameter("hold_last_seconds").value)
        self.depth_params = {
            "elongated_object": (max(1, int(self.get_parameter("stride_elongated_object").value)), int(self.get_parameter("min_points_elongated_object").value)),
            "box": (max(1, int(self.get_parameter("stride_box").value)), int(self.get_parameter("min_points_box").value)),
            "cube": (max(1, int(self.get_parameter("stride_cube").value)), int(self.get_parameter("min_points_cube").value)),
            "stone": (max(1, int(self.get_parameter("stride_cube").value)), int(self.get_parameter("min_points_cube").value)),
        }

        if self.device.lower() == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.get_logger().info(f"CUDA {'is' if self.device != 'cpu' else 'not'} available, using {self.device}.")

        self.get_logger().info(f"Loading YOLOv8 OBB model: {model_path}, device={self.device}")
        self.model = YOLO(model_path)
        try:
            self.class_names = require_four_class_obb_model(self.model.names)
        except ValueError as exc:
            self.get_logger().fatal(str(exc))
            raise
        self.get_logger().info(f"Four-class YOLO-OBB contract accepted: {self.class_names}")
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self._camera_info_signature = None
        self._camera_info_stable_count = 0
        self._three_d_enabled = False
        self.latest_rgb = self.latest_depth = self.latest_header = None
        self.lock = threading.Lock()
        self.prev_yaw = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.prev_xyz = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.last_best_xyz = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.last_best_rpy = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.last_update_wall = {name: 0.0 for name in FOUR_CLASS_OBB_NAMES.values()}
        self._busy = False
        self._sync_logged = False
        self._last_diag_wall = 0.0

        self.cb_infer = MutuallyExclusiveCallbackGroup()
        self.cb_pub = MutuallyExclusiveCallbackGroup()
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], self.sync_queue_size, self.sync_slop, allow_headerless=False)
        self.sync.registerCallback(self.synced_rgb_depth_callback)

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub_vis = self.create_publisher(Image, "/camera/detected_image", qos)
        self.position_publishers = {
            name: self.create_publisher(PointStamped, POSITION_3D_TOPICS[name], qos)
            for name in FOUR_CLASS_OBB_NAMES.values()
        }
        self.rpy_publishers = {
            name: self.create_publisher(Float32MultiArray, RPY_TOPICS[name], qos) if self.publish_rpy else None
            for name in FOUR_CLASS_OBB_NAMES.values()
        }
        self.class_colors = {"box": (255, 0, 0), "elongated_object": (0, 255, 0), "cube": (0, 255, 255), "stone": (255, 0, 255)}

        self.detection_timer = self.create_timer(self.inference_period, self.process_images, callback_group=self.cb_infer)
        self.publish_timer = self.create_timer(1.0 / max(1.0, self.pose_publish_rate), self.publish_cached_outputs, callback_group=self.cb_pub)

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is not None:
            return

        intrinsics = {
            "fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5],
            "width": msg.width, "height": msg.height, "frame_id": msg.header.frame_id,
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
        self._three_d_enabled = True
        if self.camera_info_sub is not None:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None
        self.get_logger().info(
            "CameraInfo locked after 3 stable frames: "
            f"topic={self.camera_info_topic} size={intrinsics['width']}x{intrinsics['height']} "
            f"frame={intrinsics['frame_id']!r} fx={intrinsics['fx']:.1f}, "
            f"fy={intrinsics['fy']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}"
        )

    def synced_rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        except CvBridgeError as exc:
            self.get_logger().error(f"RGB-D conversion error: {exc}")
            return
        if depth_msg.encoding in ("16UC1", "mono16"):
            depth /= 1000.0
        depth = np.nan_to_num(np.minimum(depth, 20.0), nan=0.0, posinf=20.0, neginf=0.0)
        with self.lock:
            self.latest_rgb, self.latest_depth = rgb, depth
            self.latest_header = rgb_msg.header
        if not self._sync_logged:
            self._sync_logged = True
            self.get_logger().info(f"First synchronized RGB-D frame: rgb={rgb.shape}, depth={depth.shape}, slop={self.sync_slop:.3f}s")

    def _center3d_from_obb_depth(self, poly_2d: np.ndarray, semantic_name: str, depth: np.ndarray, camera_intrinsics: dict):
        stride, min_points = self.depth_params[semantic_name]
        return robust_center3d_from_obb_depth(
            poly_2d=poly_2d, depth=depth, camera_intrinsics=camera_intrinsics,
            stride=stride, min_points=min_points, max_points=self.max_points,
            depth_max_range=self.depth_max_range, depth_inlier_m=self.depth_inlier_m,
            depth_mad_scale=self.depth_mad_scale, min_depth_inlier_ratio=self.min_depth_inlier_ratio,
            xy_from_obb_center=(semantic_name == "box"),
        )

    def smooth_xyz(self, semantic_name: str, xyz_raw: np.ndarray) -> np.ndarray:
        previous = self.prev_xyz[semantic_name]
        xyz_out = xyz_raw if previous is None else previous + self.xyz_alpha * (xyz_raw - previous)
        self.prev_xyz[semantic_name] = xyz_out
        return xyz_out

    def _estimate_yaw_0_pi(self, semantic_name: str, corners: np.ndarray) -> float:
        yaw_meas = yaw_0_to_pi_right0_left180(corners)
        previous = self.prev_yaw[semantic_name]
        period = math.pi / 2.0 if semantic_name in ("box", "cube") else math.pi
        if previous is None:
            yaw_out = yaw_meas
        else:
            meas_rep = yaw_meas - math.pi if yaw_meas > math.pi / 2.0 else yaw_meas
            prev_rep = previous - math.pi if previous > math.pi / 2.0 else previous
            yaw_out = wrap_to_pi(prev_rep + self.alpha * angle_diff(choose_equivalent_angle(meas_rep, prev_rep, period=period), prev_rep))
            if yaw_out < 0.0:
                yaw_out += math.pi
        yaw_out = max(0.0, min(math.pi, float(yaw_out)))
        self.prev_yaw[semantic_name] = yaw_out
        return yaw_out

    def _maybe_log_diagnostics(self, n_obb, detected, depth_rejected, accepted):
        now = time.monotonic()
        if now - self._last_diag_wall < 2.0:
            return
        self._last_diag_wall = now
        self.get_logger().info(
            f"OBB diagnostics: raw={n_obb}, 2d={detected}, depth_rejected={depth_rejected}, 3d_accepted={accepted}"
        )

    def publish_cached_outputs(self):
        with self.lock:
            if not self._three_d_enabled:
                return
            header = Header()
            header.frame_id = self.camera_intrinsics["frame_id"]
            if self.latest_header is not None:
                header.stamp = self.latest_header.stamp
            else:
                header.stamp = self.get_clock().now().to_msg()
            now = time.time()
            for name, publisher in self.position_publishers.items():
                xyz = self.last_best_xyz[name]
                if xyz is None or now - self.last_update_wall[name] > self.hold_last_seconds:
                    continue
                point = PointStamped()
                point.header = header
                point.point.x, point.point.y, point.point.z = map(float, xyz)
                publisher.publish(point)
                rpy_publisher = self.rpy_publishers[name]
                if rpy_publisher is not None:
                    rpy = Float32MultiArray()
                    rpy.data = [float(value) for value in self.last_best_rpy[name]]
                    rpy_publisher.publish(rpy)

    def process_images(self):
        if self._busy:
            return
        self._busy = True
        try:
            with self.lock:
                rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                source_header = self.latest_header
            if rgb is None or depth is None:
                return
            try:
                result = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            except Exception as exc:
                self.get_logger().error(f"YOLO inference error: {exc}")
                return

            vis = rgb.copy()
            detected = {name: 0 for name in FOUR_CLASS_OBB_NAMES.values()}
            depth_rejected = {name: 0 for name in FOUR_CLASS_OBB_NAMES.values()}
            accepted = {name: 0 for name in FOUR_CLASS_OBB_NAMES.values()}
            best = {name: (-1.0, None, None) for name in FOUR_CLASS_OBB_NAMES.values()}
            n_obb = 0 if getattr(result, "obb", None) is None or result.obb.xyxyxyxy is None else len(result.obb.xyxyxyxy)
            camera_intrinsics = self.camera_intrinsics

            for index in range(n_obb):
                corners = try_extract_obb_corners(result, index)
                if corners is None:
                    continue
                class_id = int(result.obb.cls[index].item())
                name = self.class_names.get(class_id)
                if name is None:
                    self.get_logger().error(f"Unexpected class id {class_id} from validated model; ignoring detection.")
                    continue
                confidence = float(result.obb.conf[index].item())
                detected[name] += 1
                color = self.class_colors[name]
                center_pixel = tuple(np.clip(np.mean(corners, axis=0), [0, 0], [rgb.shape[1] - 1, rgb.shape[0] - 1]).astype(int))
                cv2.polylines(vis, [corners.reshape(-1, 1, 2).astype(np.int32)], True, color, 2)
                draw_detection_center(vis, center_pixel)
                cv2.putText(vis, f"{name}:{confidence:.2f}", (center_pixel[0], max(0, center_pixel[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if camera_intrinsics is None:
                    continue
                center3d, quality = self._center3d_from_obb_depth(corners, name, depth, camera_intrinsics)
                if center3d is None:
                    depth_rejected[name] += 1
                    continue
                xyz = self.smooth_xyz(name, center3d)
                yaw = self._estimate_yaw_0_pi(name, corners)
                cv2.putText(vis, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}m", (center_pixel[0], min(rgb.shape[0] - 5, center_pixel[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                score = confidence * quality
                if score > best[name][0]:
                    best[name] = (score, xyz, (0.0, 0.0, yaw))

            if self._three_d_enabled and self.camera_intrinsics is camera_intrinsics:
                now = time.time()
                with self.lock:
                    if self._three_d_enabled and self.camera_intrinsics is camera_intrinsics:
                        for name, (_score, xyz, rpy) in best.items():
                            if xyz is None:
                                continue
                            self.last_best_xyz[name], self.last_best_rpy[name], self.last_update_wall[name] = xyz, rpy, now
                            accepted[name] = 1
            self._maybe_log_diagnostics(n_obb, detected, depth_rejected, accepted)
            image_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            if source_header is not None:
                image_msg.header = source_header
            self.pub_vis.publish(image_msg)
        finally:
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorObbNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
