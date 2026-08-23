#!/usr/bin/env python3
import threading
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from std_srvs.srv import SetBool
from ultralytics import YOLO

from yolo_perception_utils.depth_estimation import robust_obb_depth_samples
from yolo_perception_utils.model_utils import (
    AXIS_3D_TOPICS,
    FOUR_CLASS_OBB_NAMES,
    POSITION_3D_TOPICS,
    require_four_class_obb_model,
    resolve_yolo_model_path,
)
from yolo_perception_utils.obb_geometry import (
    cube_edge_axis,
    pca_major_axis,
    try_extract_obb_corners,
    yaw_0_to_pi_right0_left180,
)
from yolo_perception_utils.visualization import (
    draw_detection_center,
    draw_detection_diagnostics,
    draw_obb_major_axis,
)


class YoloDetectorObbNode(Node):
    """Strict four-class RGB-D OBB detector used by visual_grasping_gazebo."""

    def __init__(self):
        super().__init__("yolo_detector_obb_node")
        self.declare_parameter("model_path", "yolo-obb-1280.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("imgsz", 1280)
        self.declare_parameter("depth_max_range", 10.0)
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
        self.declare_parameter("axis_smoothing_alpha", 0.3)
        self.declare_parameter("xyz_smoothing_alpha", 0.8)
        self.declare_parameter("min_pca_yaw_quality", 0.30)
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.02)
        self.declare_parameter("use_continuous_yolo", True)

        model_path = resolve_yolo_model_path(str(self.get_parameter("model_path").value))
        self.device = str(self.get_parameter("device").value)
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.depth_max_range = float(self.get_parameter("depth_max_range").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.depth_inlier_m = max(0.001, float(self.get_parameter("depth_inlier_m").value))
        self.depth_mad_scale = max(0.0, float(self.get_parameter("depth_mad_scale").value))
        self.min_depth_inlier_ratio = min(1.0, max(0.0, float(self.get_parameter("min_depth_inlier_ratio").value)))
        self.axis_alpha = float(self.get_parameter("axis_smoothing_alpha").value)
        self.xyz_alpha = float(self.get_parameter("xyz_smoothing_alpha").value)
        self.min_pca_yaw_quality = float(self.get_parameter("min_pca_yaw_quality").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)
        self.use_continuous_yolo = bool(self.get_parameter("use_continuous_yolo").value)
        self._inference_enabled = self.use_continuous_yolo
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
        self.prev_xyz = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.prev_axis = {name: None for name in AXIS_3D_TOPICS}
        self.last_best_xyz = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.last_best_axis = {name: None for name in AXIS_3D_TOPICS}
        self.last_best_header = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        self.last_update_wall = {name: 0.0 for name in FOUR_CLASS_OBB_NAMES.values()}
        self._busy = False
        self._sync_logged = False

        self.cb_infer = MutuallyExclusiveCallbackGroup()
        self.cb_pub = MutuallyExclusiveCallbackGroup()
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], self.sync_queue_size, self.sync_slop, allow_headerless=False)
        self.sync.registerCallback(self.synced_rgb_depth_callback)

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub_vis = self.create_publisher(Image, "/camera/detected_result", qos)
        self.position_publishers = {
            name: self.create_publisher(PointStamped, POSITION_3D_TOPICS[name], qos)
            for name in FOUR_CLASS_OBB_NAMES.values()
        }
        self.axis_publishers = {
            name: self.create_publisher(Vector3Stamped, AXIS_3D_TOPICS[name], qos)
            for name in AXIS_3D_TOPICS
        }
        self.class_colors = {"box": (255, 0, 0), "elongated_object": (0, 255, 0), "cube": (0, 255, 255), "stone": (255, 0, 255)}

        self.detection_timer = self.create_timer(self.inference_period, self.process_images, callback_group=self.cb_infer)
        self.publish_timer = self.create_timer(1.0 / max(1.0, self.pose_publish_rate), self.publish_cached_outputs, callback_group=self.cb_pub)
        self.create_service(SetBool, "/yolo_detector_obb/set_inference_enabled", self._set_inference_enabled)

    def _set_inference_enabled(self, request, response):
        with self.lock:
            self._inference_enabled = bool(request.data)
            self.latest_rgb = self.latest_depth = self.latest_header = None
            self.prev_xyz = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
            self.prev_axis = {name: None for name in AXIS_3D_TOPICS}
            self.last_best_xyz = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
            self.last_best_axis = {name: None for name in AXIS_3D_TOPICS}
            self.last_best_header = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
        response.success = True
        response.message = f"YOLO inference {'enabled' if request.data else 'disabled'}"
        return response

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
        if not self._inference_enabled:
            return
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
        return robust_obb_depth_samples(
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

    def smooth_axis(self, semantic_name: str, axis_raw: np.ndarray | None) -> np.ndarray | None:
        if axis_raw is None:
            return None
        axis = np.asarray(axis_raw, dtype=np.float64).reshape(3,)
        axis /= np.linalg.norm(axis)
        if semantic_name == "cube":
            return axis.astype(np.float32)
        previous = self.prev_axis[semantic_name]
        if previous is not None:
            previous = np.asarray(previous, dtype=np.float64).reshape(3,)
            if float(axis @ previous) < 0.0:
                axis = -axis
            axis = previous + self.axis_alpha * (axis - previous)
            axis /= np.linalg.norm(axis)
        self.prev_axis[semantic_name] = axis.astype(np.float32)
        return axis.astype(np.float32)

    @staticmethod
    def _capture_header(source_header, camera_intrinsics):
        if source_header is None:
            return None
        return Header(stamp=source_header.stamp, frame_id=camera_intrinsics["frame_id"])

    def publish_cached_outputs(self):
        with self.lock:
            if not self._three_d_enabled:
                return
            now = time.time()
            for name, publisher in self.position_publishers.items():
                xyz = self.last_best_xyz[name]
                header = self.last_best_header[name]
                if xyz is None or header is None or now - self.last_update_wall[name] > self.hold_last_seconds:
                    continue
                point = PointStamped()
                point.header = header
                point.point.x, point.point.y, point.point.z = map(float, xyz)
                publisher.publish(point)
                axis = self.last_best_axis.get(name)
                if axis is not None:
                    axis_msg = Vector3Stamped()
                    axis_msg.header = header
                    axis_msg.vector.x, axis_msg.vector.y, axis_msg.vector.z = map(float, axis)
                    self.axis_publishers[name].publish(axis_msg)

    def process_images(self):
        if not self._inference_enabled:
            return
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
            best = {name: None for name in FOUR_CLASS_OBB_NAMES.values()}
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
                color = self.class_colors[name]
                center_pixel = tuple(np.clip(np.mean(corners, axis=0), [0, 0], [rgb.shape[1] - 1, rgb.shape[0] - 1]).astype(int))
                cv2.polylines(vis, [corners.reshape(-1, 1, 2).astype(np.int32)], True, color, 2)
                draw_detection_center(vis, center_pixel)
                draw_obb_major_axis(vis, corners, color)
                if camera_intrinsics is None:
                    continue
                center3d, depth_quality, points_3d, pixels_uv = self._center3d_from_obb_depth(corners, name, depth, camera_intrinsics)
                if center3d is None:
                    continue
                yaw_image = yaw_0_to_pi_right0_left180(corners)
                axis = None
                yaw_quality = None
                if name == "elongated_object":
                    axis, pca_quality = pca_major_axis(points_3d)
                    yaw_quality = depth_quality * pca_quality
                elif name == "stone":
                    axis, pca_quality = pca_major_axis(points_3d, self.min_pca_yaw_quality)
                    yaw_quality = depth_quality * pca_quality
                elif name == "cube":
                    axis = cube_edge_axis(points_3d, pixels_uv, corners, self.depth_params[name][1])
                    yaw_quality = depth_quality if axis is not None else 0.0
                candidate = {
                    "score": confidence * depth_quality,
                    "confidence": confidence,
                    "center": center3d,
                    "depth_quality": depth_quality,
                    "axis": axis,
                    "yaw_image": yaw_image,
                    "yaw_quality": yaw_quality,
                    "center_pixel": center_pixel,
                }
                if best[name] is None or candidate["score"] > best[name]["score"]:
                    best[name] = candidate

            if self._three_d_enabled and self.camera_intrinsics is camera_intrinsics:
                now = time.time()
                header = self._capture_header(source_header, camera_intrinsics)
                with self.lock:
                    if self._three_d_enabled and self.camera_intrinsics is camera_intrinsics and header is not None:
                        for name, candidate in best.items():
                            if candidate is None:
                                continue
                            xyz = self.smooth_xyz(name, candidate["center"])
                            axis = self.smooth_axis(name, candidate["axis"])
                            self.last_best_xyz[name] = xyz
                            self.last_best_header[name] = header
                            if name in self.last_best_axis:
                                self.last_best_axis[name] = axis
                            self.last_update_wall[name] = now
                            x, y = candidate["center_pixel"]
                            lines = [
                                f"{name} conf={candidate['confidence']:.2f}",
                                f"XYZc: {xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}m",
                                f"Yaw_i: {np.degrees(candidate['yaw_image']):.1f}deg",
                                f"DepthQ: {candidate['depth_quality']:.2f}",
                                "YawQ: --" if candidate["yaw_quality"] is None else f"YawQ: {candidate['yaw_quality']:.2f}",
                            ]
                            draw_detection_diagnostics(
                                vis, (x, y), lines, self.class_colors[name]
                            )
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
