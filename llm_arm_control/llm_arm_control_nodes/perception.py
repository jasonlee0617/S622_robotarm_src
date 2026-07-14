"""Synchronized RGB-D perception for LLM arm tasks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import numpy as np
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs  # noqa: F401  Registers PointStamped transforms.
from yolo_perception.msg import Yolov8Inference
from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth

from .task_logic import ClarificationRequired


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
            "base_xyz": list(self.xyz),
            "yaw": self.yaw, "frame_stamp_ns": self.frame_stamp_ns,
            "depth_inlier_ratio": self.depth_inlier_ratio,
        }


def xy_shift(left, right):
    return math.hypot(left.xyz[0] - right.xyz[0], left.xyz[1] - right.xyz[1])


class RgbdPerception:
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
            Image, depth_topic, self._depth_callback, 10, callback_group=callback_group)
        self.camera_info_subscription = node.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback,
            qos_profile_sensor_data, callback_group=callback_group)

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
            "pair_key": pair_key,
            "received_monotonic": time.monotonic(),
        }

    def current_frame(self):
        with self._lock:
            frame = self._active_frame
        if frame is None or time.monotonic() - frame["received_monotonic"] > self.detection_max_age_sec:
            return None
        return frame

    @staticmethod
    def _detections(frame):
        for index, item in enumerate(frame["yolo"].yolov8_inference):
            try:
                points = np.asarray(item.coordinates, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            center = np.mean(points, axis=0)
            yield index, item, points, center

    def metadata(self, frame=None):
        frame = self.current_frame() if frame is None else frame
        if frame is None:
            return []
        return [
            {
                "index": index, "class_name": str(item.class_name),
                "confidence": float(getattr(item, "confidence", 0.0)),
                "center_uv": [float(center[0]), float(center[1])], }
            for index, item, _points, center in self._detections(frame)
        ]

    def planning_metadata(self, frame):
        result = []
        for index, item, points, center in self._detections(frame):
            resolved = self._resolve_detection(index, item, points, center, frame)
            if resolved is None:
                continue
            result.append(
                {
                    "index": index, "class_name": str(item.class_name),
                    "confidence": float(getattr(item, "confidence", 0.0)),
                    "center_uv": [float(center[0]), float(center[1])],
                    "base_xyz": list(resolved.xyz), "yaw": resolved.yaw,
                    "depth_inlier_ratio": resolved.depth_inlier_ratio, })
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
            raise ClarificationRequired(
                "No selectable detection has valid depth/TF; adjust the view and retry.")
        raise ClarificationRequired(self.vision_unavailable_message())

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
                "`ros2 launch gazebo_launch llm_yolo_control.launch.py` and wait for "
                "the first YOLO inference.")
        if self._camera_intrinsics is None:
            return "Vision input unavailable: camera_info has not arrived yet."
        return (
            "YOLO/depth publishers are connected but no fresh synchronized frame arrived; "
            "wait for the first inference or check the camera_subscriber warning log.")

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
            min_depth_inlier_ratio=0.6,
            xy_from_obb_center=False)
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
            (center_uv[1] + axis_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"],
            z)
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
            xyz=(
                float(center_base.point.x),
                float(center_base.point.y),
                float(center_base.point.z)),
            yaw=float(yaw),
            frame_stamp_ns=int(frame["stamp_ns"]),
            depth_inlier_ratio=float(quality))

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
                "yolo_buffer_count": len(self._yolo_frames), "depth_buffer_count": len(self._depth_frames),
                "yolo_publisher_count": self._publisher_count(self.yolo_topic),
                "depth_publisher_count": self._publisher_count(self.depth_topic),
                "rgb_depth_delta_sec": None if frame is None else frame["sync_delta_sec"],
                "camera_info_ready": self._camera_intrinsics is not None,
            }
