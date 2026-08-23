#!/usr/bin/env python3

from collections import deque
import copy
from dataclasses import dataclass
import gc
import math
import threading
import time
import cv2
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_srvs.srv import SetBool, Trigger
import numpy as np
import tf2_geometry_msgs  # noqa: F401  Registers PointStamped transforms.
import tf2_ros

from yolo_perception.msg import InferenceResult
from yolo_perception.msg import Yolov8Inference
from yolo_perception_utils.model_utils import (
    assign_obb_confidence,
    require_four_class_obb_model,
    resolve_yolo_model_path,
)
from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth
from yolo_perception_utils.visualization import (
    draw_detection_center,
    draw_detection_diagnostics,
    draw_obb_major_axis,
)


class LlmYoloPerceptionNode(Node):

    def __init__(self):
        super().__init__('llm_yolo_perception')
        from ultralytics import YOLO

        self.declare_parameter('model_path', 'yolo-obb-1280.pt')
        self.declare_parameter('imgsz', 1024)
        self.declare_parameter('conf', 0.50)
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('sync_slop', 0.02)
        self.declare_parameter('sync_watchdog_sec', 3.0)
        self.declare_parameter('use_continuous_yolo', True)

        self._yolo_class = YOLO
        self.model_path = resolve_yolo_model_path(str(self.get_parameter('model_path').value))
        self._model_lock = threading.RLock()
        self._control_callback_group = ReentrantCallbackGroup()
        self.model = None
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.sync_watchdog_sec = float(self.get_parameter('sync_watchdog_sec').value)
        self.use_continuous_yolo = bool(self.get_parameter('use_continuous_yolo').value)
        self._inference_enabled = self.use_continuous_yolo
        self.class_names = None
        if not self._load_model():
            raise RuntimeError("LLM YOLO model failed to load")

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._camera_intrinsics = None
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/yolo/detected_result", 1)
        visual_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.img_pub = self.create_publisher(Image, "/camera/detected_result", visual_qos)
        self.depth_pub = self.create_publisher(Image, "/yolo/detected_result/depth", 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.class_colors = {
            "box": (255, 0, 0),
            "elongated_object": (0, 255, 0),
            "cube": (0, 255, 255),
            "stone": (255, 0, 255),
        }
        self.rgb_sub = None
        self.depth_sub = None
        self.sync = None
        self._frame_count = 0
        self._last_sync_activity = time.monotonic()
        self._start_sync()
        self.sync_watchdog = self.create_timer(1.0, self._check_sync)
        self.create_service(
            SetBool,
            '/llm_yolo_perception/set_inference_enabled',
            self._set_inference_enabled,
            callback_group=self._control_callback_group,
        )
        self.create_service(
            Trigger,
            '/llm_yolo_perception/release_gpu',
            self._release_gpu,
            callback_group=self._control_callback_group,
        )
        self.get_logger().info(
            f"YOLO OBB ready: rgb={self.rgb_topic}, depth={self.depth_topic}, "
            f"slop={self.sync_slop:.3f}s"
        )

    def _set_inference_enabled(self, request, response):
        enabled = bool(request.data)
        # Disabling must not wait for a long-running CUDA inference callback.
        # The current frame may finish, but no later frame can start inference.
        if not enabled:
            self._inference_enabled = False
            response.success = True
            response.message = "LLM YOLO inference disabled"
            return response
        if enabled and not self._load_model():
            response.success = False
            response.message = "LLM YOLO model reload failed"
            return response
        with self._model_lock:
            self._inference_enabled = enabled
        response.success = True
        response.message = f"LLM YOLO inference {'enabled' if enabled else 'disabled'}"
        return response

    def _load_model(self):
        with self._model_lock:
            if self.model is not None:
                return True
            try:
                model = self._yolo_class(self.model_path)
                class_names = require_four_class_obb_model(model.names)
            except Exception as exc:
                self.get_logger().error(f"Failed to load LLM YOLO model: {exc}")
                return False
            self.model = model
            self.class_names = class_names
        self.get_logger().info(f"Four-class YOLO-OBB contract accepted: {self.class_names}")
        return True

    def _release_gpu(self, _request, response):
        self._unload_model()
        response.success = True
        response.message = "LLM YOLO GPU model released"
        self.get_logger().info(response.message)
        return response

    def _unload_model(self):
        with self._model_lock:
            self._inference_enabled = False
            model, self.model = self.model, None
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            self.get_logger().warning(f"LLM YOLO CUDA cache cleanup skipped: {exc}")

    def _release_after_cuda_oom(self):
        self._unload_model()

    @staticmethod
    def _is_cuda_oom(exc):
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message

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

    def _camera_info_callback(self, msg):
        if msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self._camera_intrinsics = {
                "fx": float(msg.k[0]), "fy": float(msg.k[4]),
                "cx": float(msg.k[2]), "cy": float(msg.k[5]),
            }

    def _transform_point(self, xyz, header):
        point = PointStamped()
        point.header = header
        point.point.x, point.point.y, point.point.z = (float(value) for value in xyz)
        return self.tf_buffer.transform(
            point, self.base_frame, timeout=Duration(seconds=0.2)
        )

    def _diagnostic_lines(self, corners, center_uv, header, depth):
        intrinsics = self._camera_intrinsics
        if intrinsics is None:
            return ["3D unavailable: camera_info"]
        center3d, quality = robust_center3d_from_obb_depth(
            poly_2d=corners,
            depth=depth,
            camera_intrinsics=intrinsics,
            stride=1,
            min_points=20,
            max_points=5000,
            depth_max_range=10.0,
            depth_inlier_m=0.08,
            depth_mad_scale=3.0,
            min_depth_inlier_ratio=0.6,
            xy_from_obb_center=False,
        )
        if center3d is None:
            return ["3D unavailable: depth"]
        edges = np.roll(corners, -1, axis=0) - corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1e-6:
            return ["3D unavailable: OBB axis"]
        axis_uv = edge / edge_norm * min(20.0, edge_norm / 2.0)
        z = float(center3d[2])
        axis3d = (
            (center_uv[0] + axis_uv[0] - intrinsics["cx"]) * z / intrinsics["fx"],
            (center_uv[1] + axis_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"],
            z,
        )
        try:
            center_base = self._transform_point(center3d, header)
            axis_base = self._transform_point(axis3d, header)
        except Exception:
            return ["3D unavailable: TF"]
        direction = (
            axis_base.point.x - center_base.point.x,
            axis_base.point.y - center_base.point.y,
        )
        if math.hypot(*direction) <= 1e-6:
            return ["3D unavailable: axis TF"]
        yaw = (math.atan2(direction[1], direction[0]) + math.pi / 2.0) % math.pi - math.pi / 2.0
        return [
            f"base: {center_base.point.x:.3f}, {center_base.point.y:.3f}, {center_base.point.z:.3f} m",
            f"yaw: {math.degrees(yaw):.1f} deg  depthQ: {quality:.2f}",
        ]

    def camera_callback(self, rgb_msg, depth_msg):
        self._last_sync_activity = time.monotonic()
        try:
            with self._model_lock:
                if not self._inference_enabled or self.model is None:
                    return
                img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                results = self.model(img, conf=self.conf, imgsz=self.imgsz, verbose=False)
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            self._release_after_cuda_oom()
            self.get_logger().error(
                f"LLM YOLO inference disabled after CUDA OOM: {exc}"
            )
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            ).astype(np.float32)
            if depth_msg.encoding in ("16UC1", "mono16"):
                depth /= 1000.0
        except Exception:
            depth = None
        inference = Yolov8Inference()
        inference.header = rgb_msg.header
        annotated_frame = img.copy()

        for r in results:
            if r.obb is not None:
                boxes = r.obb
                for box in boxes:
                    corners = box.xyxyxyxy[0].to('cpu').detach().numpy().copy().reshape(4, 2)
                    class_name = self.class_names[int(box.cls.item())]
                    inference_result = InferenceResult()
                    inference_result.class_name = class_name
                    assign_obb_confidence(inference_result, box)
                    inference_result.coordinates = copy.copy(corners.reshape(-1).tolist())
                    inference.yolov8_inference.append(inference_result)

                    center_pixel = tuple(np.clip(
                        np.mean(corners, axis=0),
                        [0, 0],
                        [img.shape[1] - 1, img.shape[0] - 1],
                    ).astype(int))
                    color = self.class_colors[class_name]
                    cv2.polylines(
                        annotated_frame,
                        [corners.reshape(-1, 1, 2).astype(np.int32)],
                        True,
                        color,
                        2,
                    )
                    draw_detection_center(annotated_frame, center_pixel)
                    draw_obb_major_axis(annotated_frame, corners, color)
                    lines = [
                        f"{class_name} conf={float(box.conf.item()):.2f}",
                        f"uv: {center_pixel[0]}, {center_pixel[1]}",
                    ]
                    if depth is None:
                        lines.append("3D unavailable: depth")
                    else:
                        lines.extend(self._diagnostic_lines(corners, center_pixel, rgb_msg.header, depth))
                    draw_detection_diagnostics(annotated_frame, center_pixel, lines, color)

        detection_count = len(inference.yolov8_inference)
        self.yolov8_pub.publish(inference)

        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
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
    node = LlmYoloPerceptionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class PerceptionUnavailable(ValueError):
    """The LLM task cannot safely plan from the current camera data."""


@dataclass(frozen=True)
class ResolvedCandidate:
    index: int
    class_name: str
    confidence: float
    center_uv: tuple[float, float]
    xyz: tuple[float, float, float]
    yaw: float
    frame_stamp_ns: int
    depth_inlier_ratio: float

    def public(self):
        return {
            "index": self.index, "class_name": self.class_name,
            "confidence": self.confidence, "center_uv": list(self.center_uv),
            "base_xyz": list(self.xyz), "yaw": self.yaw,
            "frame_stamp_ns": self.frame_stamp_ns,
            "depth_inlier_ratio": self.depth_inlier_ratio,
        }


def xy_shift(left, right):
    return math.hypot(left.xyz[0] - right.xyz[0], left.xyz[1] - right.xyz[1])


class RgbdPerception:
    """Resolve all LLM YOLO OBB candidates against synchronized RGB-D data."""

    def __init__(self, node, tf_buffer, *, base_frame, yolo_topic, depth_topic,
                 camera_info_topic, rgb_depth_tolerance_sec, detection_max_age_sec,
                 vision_wait_timeout_sec, callback_group):
        self.node, self.tf_buffer, self.base_frame = node, tf_buffer, base_frame
        self.yolo_topic, self.depth_topic = yolo_topic, depth_topic
        self.rgb_depth_tolerance_sec = rgb_depth_tolerance_sec
        self.detection_max_age_sec = detection_max_age_sec
        self.vision_wait_timeout_sec = vision_wait_timeout_sec
        self._lock = threading.RLock()
        self._bridge = CvBridge()
        self._depth_frames, self._yolo_frames = deque(maxlen=20), deque(maxlen=20)
        self._active_frame = self._camera_intrinsics = None
        self._camera_frame = ""
        self.yolo_subscription = node.create_subscription(
            Yolov8Inference, yolo_topic, self._yolo_callback, 10,
            callback_group=callback_group)
        self.depth_subscription = node.create_subscription(
            Image, depth_topic, self._depth_callback, 10,
            callback_group=callback_group)
        self.camera_info_subscription = node.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data, callback_group=callback_group)

    def clear_frames(self):
        with self._lock:
            self._depth_frames.clear()
            self._yolo_frames.clear()
            self._active_frame = None

    @staticmethod
    def _stamp_ns(header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _camera_info_callback(self, msg):
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            return
        with self._lock:
            self._camera_intrinsics = {
                "fx": float(msg.k[0]), "fy": float(msg.k[4]),
                "cx": float(msg.k[2]), "cy": float(msg.k[5]),
            }
            self._camera_frame = str(msg.header.frame_id)

    def _depth_callback(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough").astype(np.float32)
            if msg.encoding in ("16UC1", "mono16"):
                depth /= 1000.0
            depth = np.nan_to_num(depth, nan=0.0, posinf=20.0, neginf=0.0)
            depth[depth > 20.0] = 20.0
        except Exception as exc:
            self.node.get_logger().warning(f"Cannot decode YOLO depth: {exc}")
            return
        with self._lock:
            self._depth_frames.append((msg.header, depth))
            self._activate_frame_locked()

    def _yolo_callback(self, msg):
        with self._lock:
            self._yolo_frames.append(msg)
            self._activate_frame_locked()

    def _activate_frame_locked(self):
        if not self._yolo_frames or not self._depth_frames:
            return
        tolerance_ns = int(self.rgb_depth_tolerance_sec * 1e9)
        matches = (
            (self._stamp_ns(yolo.header),
             abs(self._stamp_ns(yolo.header) - self._stamp_ns(header)),
             yolo, header, depth)
            for yolo in self._yolo_frames
            for header, depth in self._depth_frames
        )
        try:
            stamp_ns, delta_ns, yolo, header, depth = max(
                (item for item in matches if item[1] <= tolerance_ns),
                key=lambda item: (item[0], -item[1]))
        except ValueError:
            return
        pair_key = (stamp_ns, self._stamp_ns(header))
        if self._active_frame is not None and self._active_frame["pair_key"] == pair_key:
            return
        self._active_frame = {
            "yolo": yolo, "depth_header": header, "depth": depth,
            "stamp_ns": stamp_ns, "sync_delta_sec": delta_ns / 1e9,
            "pair_key": pair_key, "received_monotonic": time.monotonic(),
        }

    def current_frame(self):
        with self._lock:
            frame = self._active_frame
        if (
            frame is None
            or time.monotonic() - frame["received_monotonic"] > self.detection_max_age_sec
        ):
            return None
        return frame

    @staticmethod
    def _detections(frame):
        for index, item in enumerate(frame["yolo"].yolov8_inference):
            try:
                points = np.asarray(item.coordinates, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            yield index, item, points, np.mean(points, axis=0)

    def metadata(self, frame=None):
        frame = self.current_frame() if frame is None else frame
        if frame is None:
            return []
        return [
            {"index": index, "class_name": str(item.class_name),
             "confidence": float(getattr(item, "confidence", 0.0)),
             "center_uv": [float(center[0]), float(center[1])]}
            for index, item, _points, center in self._detections(frame)
        ]

    def planning_metadata(self, frame):
        result = []
        shape = getattr(frame.get("depth"), "shape", ())
        image_size = [int(shape[1]), int(shape[0])] if len(shape) >= 2 else None
        for index, item, points, center in self._detections(frame):
            resolved = self._resolve_detection(index, item, points, center, frame)
            if resolved is not None:
                public = resolved.public()
                if image_size is not None:
                    public["image_size"] = image_size
                result.append(public)
        return result

    def wait_for_planning_metadata(self):
        deadline = time.monotonic() + max(0.0, self.vision_wait_timeout_sec)
        frame_seen = False
        while True:
            frame = self.current_frame()
            if frame is not None:
                frame_seen = True
                metadata = self.planning_metadata(frame)
                if metadata:
                    return metadata
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if frame_seen:
            raise PerceptionUnavailable(
                "No selectable detection has valid depth/TF; adjust the view and retry.")
        raise PerceptionUnavailable(self.vision_unavailable_message())

    def _publisher_count(self, topic):
        try:
            return int(self.node.count_publishers(topic))
        except Exception:
            return 0

    def vision_unavailable_message(self):
        missing = [topic for topic in (self.yolo_topic, self.depth_topic)
                   if self._publisher_count(topic) == 0]
        if missing:
            return (
                "Vision input unavailable: no publisher on "
                f"{', '.join(missing)}. Start "
                "`ros2 launch myrobot_simulation llm_robot_control_gazebo.launch.py` "
                "and wait for the first YOLO inference."
            )
        if self._camera_intrinsics is None:
            return "Vision input unavailable: camera_info has not arrived yet."
        return (
            "YOLO/depth publishers are connected but no fresh synchronized frame arrived; "
            "wait for the first inference or check the llm_yolo_perception warning log."
        )

    def _transform_point(self, xyz, header):
        point = PointStamped()
        point.header = header
        if not point.header.frame_id:
            point.header.frame_id = self._camera_frame
        point.point.x, point.point.y, point.point.z = (float(value) for value in xyz)
        return self.tf_buffer.transform(
            point, self.base_frame, timeout=Duration(seconds=0.2))

    def _resolve_detection(self, index, item, points, center_uv, frame):
        if self._camera_intrinsics is None:
            return None
        center3d, quality = robust_center3d_from_obb_depth(
            poly_2d=points, depth=frame["depth"],
            camera_intrinsics=self._camera_intrinsics,
            stride=1, min_points=20, max_points=5000,
            depth_max_range=10.0, depth_inlier_m=0.08, depth_mad_scale=3.0,
            min_depth_inlier_ratio=0.6, xy_from_obb_center=False)
        if center3d is None:
            return None
        edges = np.roll(points, -1, axis=0) - points
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1e-6:
            return None
        axis_uv = edge / edge_norm * min(20.0, edge_norm / 2.0)
        z = float(center3d[2])
        intrinsics = self._camera_intrinsics
        axis3d = (
            (center_uv[0] + axis_uv[0] - intrinsics["cx"]) * z / intrinsics["fx"],
            (center_uv[1] + axis_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"], z)
        try:
            center_base = self._transform_point(center3d, frame["yolo"].header)
            axis_base = self._transform_point(axis3d, frame["yolo"].header)
        except Exception as exc:
            self.node.get_logger().warning(f"camera-to-base TF unavailable: {exc}")
            return None
        direction = (axis_base.point.x - center_base.point.x,
                     axis_base.point.y - center_base.point.y)
        if math.hypot(*direction) <= 1e-6:
            return None
        yaw = math.atan2(direction[1], direction[0])
        yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        return ResolvedCandidate(
            index=int(index), class_name=str(item.class_name),
            confidence=float(getattr(item, "confidence", 0.0)),
            center_uv=(float(center_uv[0]), float(center_uv[1])),
            xyz=(float(center_base.point.x), float(center_base.point.y),
                 float(center_base.point.z)), yaw=float(yaw),
            frame_stamp_ns=int(frame["stamp_ns"]), depth_inlier_ratio=float(quality))

    def resolve_candidate(self, index, frame=None):
        frame = self.current_frame() if frame is None else frame
        if frame is None:
            return None
        for item_index, item, points, center in self._detections(frame):
            if item_index == int(index):
                return self._resolve_detection(item_index, item, points, center, frame)
        return None

    def fresh_match(self, old):
        frame = self.current_frame()
        if frame is None:
            return None
        matches = [
            resolved
            for index, item, points, center in self._detections(frame)
            if str(item.class_name) == old.class_name
            for resolved in [self._resolve_detection(index, item, points, center, frame)]
            if resolved is not None
        ]
        return min(matches, key=lambda candidate: xy_shift(old, candidate)) if matches else None

    def diagnostics(self, frame=None):
        frame = self.current_frame() if frame is None else frame
        with self._lock:
            return {
                "fresh_detection": frame is not None,
                "candidate_count": len(self.metadata(frame)),
                "yolo_buffer_count": len(self._yolo_frames),
                "depth_buffer_count": len(self._depth_frames),
                "yolo_publisher_count": self._publisher_count(self.yolo_topic),
                "depth_publisher_count": self._publisher_count(self.depth_topic),
                "rgb_depth_delta_sec": None if frame is None else frame["sync_delta_sec"],
                "camera_info_ready": self._camera_intrinsics is not None,
            }


if __name__ == '__main__':
    main()
