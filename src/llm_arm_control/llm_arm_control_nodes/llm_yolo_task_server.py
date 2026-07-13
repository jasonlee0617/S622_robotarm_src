"""Central DeepSeek, YOLO RGB-D, and Fairino task server."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
import uuid

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from llm_arm_control.action import ExecutePreview
from llm_arm_control.srv import PreviewCommand
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_geometry_msgs  # noqa: F401  Registers PointStamped transforms.
import tf2_ros
from yolo_perception.msg import Yolov8Inference
from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth

from .deepseek_client import DeepSeekClient
from .deepseek_credentials import get_deepseek_api_key
from .fairino_pose_control_server import FairinoPoseControlServer
from .task_logic import (
    ClarificationRequired,
    DetectionCandidate,
    SafetyState,
    TaskPlan,
    TaskPreview,
    apply_safety_command,
    build_semantic_history,
    complete_safety_reset,
    consume_preview,
    decide_box_relocation,
    execution_step_count,
    instruction_has_visual_intent,
    parse_llm_plan,
    preview_status,
    safety_execution_valid,
    validate_instruction,
    validate_plan_intent,
    validate_visual_state,
)


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
            "index": self.index,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "center_uv": list(self.center_uv),
            "base_xyz": list(self.xyz),
            "yaw": self.yaw,
            "frame_stamp_ns": self.frame_stamp_ns,
            "depth_inlier_ratio": self.depth_inlier_ratio,
        }


@dataclass
class PreviewRecord:
    preview: TaskPreview
    session_id: str
    instruction: str
    enriched_actions: list[dict]
    safety_epoch: int
    public: dict


SYSTEM_PROMPT = """You control a Fairino arm through a strictly validated local planner.
Return only JSON with exactly one top-level key: {"actions": [...]}.
Allowed action objects:
1. {"type":"pick","source_index":int}
2. {"type":"place","destination_index":int}
3. {"type":"pick_place","source_index":int,"destination_index":int}
4. {"type":"move_relative","dx":m,"dy":m,"dz":m,
   "droll_deg":deg,"dpitch_deg":deg,"dyaw_deg":deg,"frame_id":"base_link"}
5. {"type":"move_absolute","x":m,"y":m,"z":m,
   "qx":number,"qy":number,"qz":number,"qw":number,"frame_id":"base_link"}
6. {"type":"set_gripper","state":"open|close"}
7. {"type":"home"}
The visual class elongated_object includes language aliases pen and bolt. Never use blot.
Use pick for a requested grasp without a destination, place for an already-held object,
and pick_place when both source and destination are requested. Never replace a visual
pick/place request with set_gripper, move_relative, or move_absolute.
Use only listed detection indices. Never invent visual coordinates. If a request is ambiguous,
do not guess: return an action with an unavailable index so the local validator rejects it.
Candidate center_uv is in image pixels: leftmost has the smallest u and rightmost the largest u.
Candidate base_xyz is in base_link. For nearest/farthest requests, compare Euclidean distance
from base_xyz to current_pose. Use only candidates present in the current request.
For a visual task return exactly one pick, place, or pick_place action. A place action is
valid only when holding_class is not null. Directions without an explicit frame always use
base_link. Maximum eight actions.
"""

RETRY_PENDING_PLACE = "__retry_pending_place__"


class LlmYoloTaskServer(FairinoPoseControlServer):
    def __init__(self):
        super().__init__("llm_yolo_task_server")
        self._declare_task_parameters()
        self._read_task_parameters()
        self._lock = threading.RLock()
        self._bridge = CvBridge()
        self._depth_frames = deque(maxlen=20)
        self._yolo_frames = deque(maxlen=20)
        self._active_frame = None
        self._camera_intrinsics = None
        self._camera_frame = ""
        self._previews: dict[str, PreviewRecord] = {}
        self._consumed_previews = frozenset()
        self._sessions: dict[str, list[dict]] = {}
        self._client = None
        self._client_key = None
        self._state = "IDLE"
        self._safety = SafetyState()
        self._execution_active = False
        self._reset_failed = False
        self._held_source = None
        self._pending_place = None
        self._cached_box_fallback_available = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._setup_perception()
        self.abort.set_command_hook(self._advance_safety)
        self.clear_session_subscription = self.create_subscription(
            String,
            "/llm_arm/clear_session",
            self._clear_session,
            10,
            callback_group=self.callback_group,
        )
        self.preview_service = self.create_service(
            PreviewCommand,
            "/llm_arm/preview_command",
            self._preview_command,
            callback_group=self.callback_group,
        )
        self.status_service = self.create_service(
            Trigger, "/llm_arm/status", self._status, callback_group=self.callback_group
        )
        self.execute_action = ActionServer(
            self,
            ExecutePreview,
            "/llm_arm/execute_preview",
            execute_callback=self._execute_preview,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group,
        )
        self.abort.set_recovery_hooks(
            open_gripper_fn=self._open_gripper,
            go_home_fn=self._go_home,
            recovery_complete_fn=self._recovery_complete,
            wait_task_stopped_fn=self._wait_execution_stopped,
            stop_timeout_sec=self.reset_stop_timeout_sec,
        )
        self.get_logger().info(
            "LLM-YOLO task server ready: /llm_arm/preview_command, /llm_arm/execute_preview"
        )

    def _declare_task_parameters(self):
        defaults = {
            "yolo_topic": "/Yolov8_Inference",
            "depth_topic": "/Yolov8_Inference/depth",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "preview_max_age_sec": 15.0,
            "detection_max_age_sec": 1.0,
            "rgb_depth_tolerance_sec": 0.05,
            "vision_wait_timeout_sec": 15.0,
            "pick_classes": ["elongated_object", "cube"],
            "place_classes": ["box"],
            "workspace_min_xyz": [-0.10, -0.60, 0.01],
            "workspace_max_xyz": [0.60, 0.60, 0.55],
            "home_joints": [-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0],
            "use_visual_z": False,
            "fixed_grasp_z": 0.02,
            "fixed_approach_z": 0.12,
            "fixed_carry_z": 0.15,
            "fixed_release_z": 0.10,
            "visual_grasp_offset_z": 0.0,
            "visual_release_offset_z": 0.03,
            "approach_offset_z": 0.10,
            "carry_offset_z": 0.13,
            "grasp_yaw_offset_rad": 1.5719,
            "box_sample_count": 5,
            "source_revalidate_threshold_m": 0.02,
            "box_retarget_threshold_m": 0.01,
            "box_max_shift_m": 0.05,
            "box_stability_threshold_m": 0.01,
            "box_retarget_timeout_sec": 5.0,
            "reset_stop_timeout_sec": 5.0,
            "deepseek_base_url": "https://api.deepseek.com",
            "deepseek_model": "deepseek-chat",
            "deepseek_timeout_sec": 30.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_task_parameters(self):
        value = lambda name: self.get_parameter(name).value
        self.yolo_topic = str(value("yolo_topic"))
        self.depth_topic = str(value("depth_topic"))
        self.camera_info_topic = str(value("camera_info_topic"))
        self.preview_max_age_sec = float(value("preview_max_age_sec"))
        self.detection_max_age_sec = float(value("detection_max_age_sec"))
        self.rgb_depth_tolerance_sec = float(value("rgb_depth_tolerance_sec"))
        self.vision_wait_timeout_sec = float(value("vision_wait_timeout_sec"))
        self.pick_classes = frozenset(str(item) for item in value("pick_classes"))
        self.place_classes = frozenset(str(item) for item in value("place_classes"))
        self.workspace_min_xyz = tuple(float(item) for item in value("workspace_min_xyz"))
        self.workspace_max_xyz = tuple(float(item) for item in value("workspace_max_xyz"))
        self.home_joints = tuple(float(item) for item in value("home_joints"))
        self.use_visual_z = bool(value("use_visual_z"))
        for name in (
            "fixed_grasp_z", "fixed_approach_z", "fixed_carry_z", "fixed_release_z",
            "visual_grasp_offset_z", "visual_release_offset_z", "approach_offset_z",
            "carry_offset_z", "grasp_yaw_offset_rad", "source_revalidate_threshold_m",
            "box_retarget_threshold_m", "box_max_shift_m", "box_stability_threshold_m",
            "box_retarget_timeout_sec", "reset_stop_timeout_sec", "deepseek_timeout_sec",
        ):
            setattr(self, name, float(value(name)))
        self.box_sample_count = int(value("box_sample_count"))
        if self.box_sample_count < 2:
            raise ValueError("box_sample_count must be at least two")
        if not 0.0 <= self.box_retarget_threshold_m <= self.box_max_shift_m:
            raise ValueError(
                "box thresholds must satisfy 0 <= retarget threshold <= maximum shift"
            )
        if self.source_revalidate_threshold_m < 0.0:
            raise ValueError("source_revalidate_threshold_m must be non-negative")
        if self.box_stability_threshold_m < 0.0:
            raise ValueError("box_stability_threshold_m must be non-negative")
        self.deepseek_base_url = str(value("deepseek_base_url"))
        self.deepseek_model = str(value("deepseek_model"))

    def _setup_perception(self):
        self.yolo_subscription = self.create_subscription(
            Yolov8Inference,
            self.yolo_topic,
            self._yolo_callback,
            10,
            callback_group=self.callback_group,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self._depth_callback,
            10,
            callback_group=self.callback_group,
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )

    def _advance_safety(self, command):
        command = str(command).strip().lower()
        with self._lock:
            if command in ("reset", "resume"):
                self._reset_failed = False
            updated = apply_safety_command(self._safety, command)
            if updated is self._safety:
                return
            self._safety = updated
            if command == "stop":
                self._state = "RESET_FAILED" if self._reset_failed else "STOPPED"
            elif command == "reset":
                self._state = "RESETTING"
            elif command == "resume":
                if self._state in ("STOPPED", "RESETTING", "RESET_FAILED"):
                    if not self._execution_active:
                        self._state = self._resting_state_locked()

    def _resting_state_locked(self):
        if self._pending_place is not None:
            return "HOLDING_RECOVERY"
        return "HOLDING" if self._held_source is not None else "IDLE"

    def _motion_block_reason_locked(self):
        reasons = []
        if self._safety.blocked:
            reasons.append("safety state is blocked")
        if self.abort.is_set():
            reasons.append("abort manager is set")
        return "; ".join(reasons)

    def _clear_session(self, msg):
        session_id = str(msg.data).strip()
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)
            stale_preview_ids = [
                preview_id
                for preview_id, record in self._previews.items()
                if record.session_id == session_id
            ]
            for preview_id in stale_preview_ids:
                self._previews.pop(preview_id, None)
            if self._state == "PREVIEW_READY" and not self._previews:
                self._state = self._resting_state_locked()
        self.get_logger().info(f"Cleared language session {session_id!r}.")

    def _execution_interrupted(self, execution_epoch, goal_handle=None):
        with self._lock:
            valid = safety_execution_valid(self._safety, execution_epoch)
        cancel_requested = goal_handle is not None and goal_handle.is_cancel_requested
        return not valid or self.abort.is_set() or cancel_requested

    def _mark_stop_state(self):
        with self._lock:
            if self.abort.is_stop_requested() or self._safety.command == "stop":
                self._state = "STOPPED"

    def _mark_holding_recovery(self, source, destination):
        with self._lock:
            if self._held_source is not None:
                self._held_source = source
                self._pending_place = (source, destination)
                self._cached_box_fallback_available = False
                self._state = "HOLDING_RECOVERY"

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
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
            if msg.encoding in ("16UC1", "mono16"):
                depth /= 1000.0
            depth = np.nan_to_num(depth, nan=0.0, posinf=20.0, neginf=0.0)
            depth[depth > 20.0] = 20.0
        except Exception as exc:
            self.get_logger().warning(f"Cannot decode YOLO depth: {exc}")
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
            (
                self._stamp_ns(yolo.header),
                abs(self._stamp_ns(yolo.header) - self._stamp_ns(header)),
                yolo,
                header,
                depth,
            )
            for yolo in self._yolo_frames
            for header, depth in self._depth_frames
        )
        valid_matches = (item for item in matches if item[1] <= tolerance_ns)
        try:
            stamp_ns, delta_ns, yolo, header, depth = max(
                valid_matches, key=lambda item: (item[0], -item[1])
            )
        except ValueError:
            return
        active_key = (stamp_ns, self._stamp_ns(header))
        if self._active_frame is not None and self._active_frame.get("pair_key") == active_key:
            return
        self._active_frame = {
            "yolo": yolo,
            "depth_header": header,
            "depth": depth,
            "stamp_ns": stamp_ns,
            "sync_delta_sec": delta_ns / 1e9,
            "pair_key": active_key,
            "received_monotonic": time.monotonic(),
        }

    def _current_frame(self):
        with self._lock:
            frame = self._active_frame
        if frame is None or time.monotonic() - frame["received_monotonic"] > self.detection_max_age_sec:
            return None
        return frame

    def _metadata(self, frame=None):
        frame = self._current_frame() if frame is None else frame
        if frame is None:
            return []
        result = []
        for index, item in enumerate(frame["yolo"].yolov8_inference):
            try:
                points = np.asarray(item.coordinates, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            center = np.mean(points, axis=0)
            result.append({
                "index": index,
                "class_name": str(item.class_name),
                "confidence": float(getattr(item, "confidence", 0.0)),
                "center_uv": [float(center[0]), float(center[1])],
            })
        return result

    def _planning_metadata(self, frame):
        result = []
        for item in self._metadata(frame):
            resolved = self._resolve_candidate(item["index"], frame)
            if resolved is None:
                continue
            result.append({
                **item,
                "base_xyz": list(resolved.xyz),
                "yaw": resolved.yaw,
                "depth_inlier_ratio": resolved.depth_inlier_ratio,
            })
        return result

    def _publisher_count(self, topic):
        try:
            return int(self.count_publishers(topic))
        except Exception:
            return 0

    def _vision_unavailable_message(self):
        missing = [
            topic
            for topic in (self.yolo_topic, self.depth_topic)
            if self._publisher_count(topic) == 0
        ]
        if missing:
            return (
                "Vision input unavailable: no publisher on "
                f"{', '.join(missing)}. Start "
                "`ros2 launch gazebo_launch llm_yolo_control.launch.py` and wait for "
                "the first YOLO inference."
            )
        if self._camera_intrinsics is None:
            return "Vision input unavailable: camera_info has not arrived yet."
        return (
            "YOLO/depth publishers are connected but no fresh synchronized frame arrived; "
            "wait for the first inference or check the camera_subscriber warning log."
        )

    def _wait_for_planning_metadata(self):
        deadline = time.monotonic() + max(0.0, self.vision_wait_timeout_sec)
        frame_seen = False
        while True:
            frame = self._current_frame()
            if frame is not None:
                frame_seen = True
                metadata = self._planning_metadata(frame)
                if metadata:
                    return metadata
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if frame_seen:
            raise ClarificationRequired(
                "No selectable detection has valid depth/TF; adjust the view and retry."
            )
        raise ClarificationRequired(self._vision_unavailable_message())

    def _transform_point(self, xyz, header):
        point = PointStamped()
        point.header = header
        if not point.header.frame_id:
            point.header.frame_id = self._camera_frame
        point.point.x, point.point.y, point.point.z = (float(value) for value in xyz)
        return self.tf_buffer.transform(point, self.base_frame, timeout=Duration(seconds=0.2))

    def _resolve_candidate(self, index: int, frame=None):
        frame = self._current_frame() if frame is None else frame
        if frame is None or self._camera_intrinsics is None:
            return None
        try:
            item = frame["yolo"].yolov8_inference[int(index)]
            points = np.asarray(item.coordinates, dtype=np.float32).reshape(4, 2)
        except (IndexError, TypeError, ValueError):
            return None
        center3d, quality = robust_center3d_from_obb_depth(
            poly_2d=points,
            depth=frame["depth"],
            camera_intrinsics=self._camera_intrinsics,
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
            return None
        edges = np.roll(points, -1, axis=0) - points
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= 1e-6:
            return None
        center_uv = np.mean(points, axis=0)
        axis_uv = edge / edge_norm * min(20.0, edge_norm / 2.0)
        z = float(center3d[2])
        intrinsics = self._camera_intrinsics
        axis3d = (
            (center_uv[0] + axis_uv[0] - intrinsics["cx"]) * z / intrinsics["fx"],
            (center_uv[1] + axis_uv[1] - intrinsics["cy"]) * z / intrinsics["fy"],
            z,
        )
        try:
            center_base = self._transform_point(center3d, frame["yolo"].header)
            axis_base = self._transform_point(axis3d, frame["yolo"].header)
        except Exception as exc:
            self.get_logger().warning(f"camera-to-base TF unavailable: {exc}")
            return None
        direction = (
            axis_base.point.x - center_base.point.x,
            axis_base.point.y - center_base.point.y,
        )
        if math.hypot(*direction) <= 1e-6:
            return None
        yaw = math.atan2(direction[1], direction[0])
        yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
        return ResolvedCandidate(
            index=int(index),
            class_name=str(item.class_name),
            confidence=float(getattr(item, "confidence", 0.0)),
            center_uv=(float(center_uv[0]), float(center_uv[1])),
            xyz=(float(center_base.point.x), float(center_base.point.y), float(center_base.point.z)),
            yaw=float(yaw),
            frame_stamp_ns=int(frame["stamp_ns"]),
            depth_inlier_ratio=float(quality),
        )

    def _deepseek(self):
        key = get_deepseek_api_key()
        if self._client is None or self._client_key != key:
            self._client = DeepSeekClient(key, self.deepseek_base_url, self.deepseek_timeout_sec)
            self._client_key = key
        return self._client

    def _current_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, self.ee_frame, Time(), timeout=Duration(seconds=0.2)
        )
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    @staticmethod
    def _has_disambiguator(instruction: str) -> bool:
        words = (
            "左", "右", "前", "后", "上", "下", "最近", "最远", "靠近", "第", "编号", "索引",
            "left", "right", "front", "back", "nearest", "farthest", "index", "number",
        )
        lowered = instruction.lower()
        return any(word in lowered for word in words)

    def _llm_plan(self, session_id, instruction, metadata):
        validate_instruction(instruction)
        current_pose = self._current_pose()
        pose = current_pose.pose
        with self._lock:
            holding_class = (
                self._held_source.class_name if self._held_source is not None else None
            )
            history = list(self._sessions.get(session_id, ()))
        context = {
            "instruction": instruction,
            "candidates": metadata,
            "holding_class": holding_class,
            "current_pose": {
                "frame_id": self.base_frame,
                "x": pose.position.x, "y": pose.position.y, "z": pose.position.z,
                "qx": pose.orientation.x, "qy": pose.orientation.y,
                "qz": pose.orientation.z, "qw": pose.orientation.w,
            },
        }
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        messages.append({"role": "user", "content": json.dumps(context, ensure_ascii=False)})
        response_text = self._deepseek().chat(messages, self.deepseek_model)
        candidates = [
            DetectionCandidate(item["index"], item["class_name"])
            for item in metadata
        ]
        try:
            plan = parse_llm_plan(
                response_text,
                candidates,
                pick_classes=self.pick_classes,
                place_classes=self.place_classes,
                reject_ambiguous=not self._has_disambiguator(instruction),
            )
            validate_plan_intent(instruction, plan)
        except ClarificationRequired:
            semantic_history = build_semantic_history(instruction)
            with self._lock:
                history = self._sessions.setdefault(session_id, [])
                history.extend(semantic_history)
                del history[:-20]
            raise
        semantic_history = build_semantic_history(instruction, plan, candidates)
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.extend(semantic_history)
            del history[:-20]
        return plan

    def _workspace_ok(self, xyz):
        return all(lower <= float(value) <= upper for value, lower, upper in zip(
            xyz, self.workspace_min_xyz, self.workspace_max_xyz
        ))

    def _pose_from_xyz_quat(self, xyz, quat):
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in xyz)
        pose.pose.orientation.x, pose.pose.orientation.y = float(quat[0]), float(quat[1])
        pose.pose.orientation.z, pose.pose.orientation.w = float(quat[2]), float(quat[3])
        return pose

    @staticmethod
    def _pose_public(pose):
        return {
            "frame_id": pose.header.frame_id,
            "position": [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            "orientation_xyzw": [pose.pose.orientation.x, pose.pose.orientation.y,
                                 pose.pose.orientation.z, pose.pose.orientation.w],
        }

    def _check_pose(self, pose):
        xyz = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        if not self._workspace_ok(xyz):
            raise ValueError(f"target outside workspace whitelist: {xyz}")
        quat = (pose.pose.orientation.x, pose.pose.orientation.y,
                pose.pose.orientation.z, pose.pose.orientation.w)
        if self.moveit2_arm.compute_ik(xyz, quat, wait_for_server_timeout_sec=1.0) is None:
            raise ValueError("target pose has no collision-aware IK solution")

    def _enrich_plan(self, plan: TaskPlan):
        frame = self._current_frame()
        current_pose = self._current_pose()
        relative_base_known = True
        enriched = []
        public_steps = []
        detections = {}
        for action in plan.actions:
            action_type = action["type"]
            if action_type in ("pick", "place", "pick_place"):
                with self._lock:
                    held_source = self._held_source
                    pending_place = self._pending_place
                validate_visual_state(
                    action_type,
                    holding=held_source is not None,
                    recovery=pending_place is not None,
                )
                if action_type in ("pick", "pick_place"):
                    source = self._resolve_candidate(action["source_index"], frame)
                    if source is None:
                        raise ValueError("selected pick target has invalid depth/TF")
                else:
                    source = held_source
                destination = None
                if action_type in ("place", "pick_place"):
                    destination = self._resolve_candidate(
                        action["destination_index"], frame
                    )
                    if destination is None:
                        raise ValueError("selected box has invalid depth/TF")
                enriched_action = {**action, "source": source}
                detections[source.index] = source.public()
                if destination is not None:
                    enriched_action["destination"] = destination
                    detections[destination.index] = destination.public()
                enriched.append(enriched_action)
                if action_type == "pick":
                    poses = self._pick_preview_poses(source)
                    public_steps.extend(self._pick_public_steps(source))
                elif action_type == "place":
                    poses = self._place_preview_poses(source, destination)
                    public_steps.extend(self._place_public_steps(source, destination))
                else:
                    poses = self._pick_place_preview_poses(source, destination)
                    public_steps.extend(self._pick_place_public_steps(source, destination))
                for pose in poses.values():
                    self._check_pose(pose)
                relative_base_known = False
            elif action_type in ("move_relative", "move_absolute"):
                if action_type == "move_relative":
                    if not relative_base_known:
                        raise ValueError(
                            "move_relative after a visual action or Home is unsafe because its "
                            "execution-time reference pose is not known; use move_absolute instead"
                        )
                    q = current_pose.pose.orientation
                    current_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
                    delta = Rotation.from_euler(
                        "xyz", [action["droll_deg"], action["dpitch_deg"], action["dyaw_deg"]], degrees=True
                    )
                    quat = (delta * current_rotation).as_quat()
                    xyz = (
                        current_pose.pose.position.x + action["dx"],
                        current_pose.pose.position.y + action["dy"],
                        current_pose.pose.position.z + action["dz"],
                    )
                else:
                    xyz = (action["x"], action["y"], action["z"])
                    quat = (action["qx"], action["qy"], action["qz"], action["qw"])
                pose = self._pose_from_xyz_quat(xyz, quat)
                self._check_pose(pose)
                enriched.append({**action, "target_pose": pose})
                public_steps.append({"type": action_type, "target_pose": self._pose_public(pose), "source": "llm_validated"})
                current_pose = pose
                relative_base_known = True
            else:
                enriched.append(dict(action))
                if action_type == "set_gripper":
                    width = abs(self.open_finger_position) * 2.0 if action["state"] == "open" else 0.0
                    public_steps.append({
                        "type": "set_gripper", "state": action["state"], "width_m": width,
                    })
                else:
                    public_steps.append({"type": "home", "target_joints": list(self.home_joints)})
                    relative_base_known = False
        return enriched, list(detections.values()), public_steps

    def _grasp_quat(self, yaw):
        return Rotation.from_euler("xyz", [0.0, -180.0, math.degrees(yaw + self.grasp_yaw_offset_rad)], degrees=True).as_quat()

    def _pick_heights(self, source):
        if self.use_visual_z:
            grasp = source.xyz[2] + self.visual_grasp_offset_z
            approach = grasp + self.approach_offset_z
            carry = grasp + self.carry_offset_z
        else:
            grasp, approach, carry = (
                self.fixed_grasp_z,
                self.fixed_approach_z,
                self.fixed_carry_z,
            )
        return grasp, approach, carry

    def _release_height(self, destination):
        if self.use_visual_z:
            return destination.xyz[2] + self.visual_release_offset_z
        return self.fixed_release_z

    def _pick_preview_poses(self, source):
        grasp, approach, carry = self._pick_heights(source)
        quat = self._grasp_quat(source.yaw)
        return {
            "approach_pick": self._pose_from_xyz_quat(
                (source.xyz[0], source.xyz[1], approach), quat
            ),
            "grasp": self._pose_from_xyz_quat(
                (source.xyz[0], source.xyz[1], grasp), quat
            ),
            "carry": self._pose_from_xyz_quat(
                (source.xyz[0], source.xyz[1], carry), quat
            ),
        }

    def _place_preview_poses(self, source, destination):
        _grasp, _approach, carry = self._pick_heights(source)
        release = self._release_height(destination)
        quat = self._grasp_quat(source.yaw)
        return {
            "approach_box": self._pose_from_xyz_quat(
                (destination.xyz[0], destination.xyz[1], carry), quat
            ),
            "release": self._pose_from_xyz_quat(
                (destination.xyz[0], destination.xyz[1], release), quat
            ),
        }

    def _pick_place_preview_poses(self, source, destination):
        return {
            **self._pick_preview_poses(source),
            **self._place_preview_poses(source, destination),
        }

    def _pick_public_steps(self, source):
        poses = self._pick_preview_poses(source)
        open_width = abs(self.open_finger_position) * 2.0
        return [
            {"type": "open_gripper", "state": "open", "width_m": open_width},
            {"type": "home", "target_joints": list(self.home_joints)},
            {"type": "approach_pick", "target_pose": self._pose_public(poses["approach_pick"]),
             "source": "vision"},
            {"type": "grasp", "target_pose": self._pose_public(poses["grasp"]),
             "source": "vision"},
            {"type": "close_gripper", "state": "close", "width_m": 0.0},
            {"type": "carry", "target_pose": self._pose_public(poses["carry"]),
             "source": "vision"},
        ]

    def _place_public_steps(self, source, destination):
        poses = self._place_preview_poses(source, destination)
        open_width = abs(self.open_finger_position) * 2.0
        return [
            {
                "type": "re_detect_box",
                "fresh_frame_count": self.box_sample_count,
                "stability_threshold_m": self.box_stability_threshold_m,
                "retarget_threshold_m": self.box_retarget_threshold_m,
                "maximum_shift_m": self.box_max_shift_m,
                "timeout_sec": self.box_retarget_timeout_sec,
            },
            {"type": "approach_box", "target_pose": self._pose_public(poses["approach_box"]),
             "source": "vision_preview"},
            {"type": "release", "target_pose": self._pose_public(poses["release"]),
             "source": "vision_preview"},
            {"type": "release_gripper", "state": "open", "width_m": open_width},
            {"type": "box_retreat", "target_pose": self._pose_public(poses["approach_box"]),
             "source": "vision_retargeted_at_execution"},
            {"type": "return_home", "target_joints": list(self.home_joints)},
            {"type": "final_gripper_close", "state": "close", "width_m": 0.0},
        ]

    def _pick_place_public_steps(self, source, destination):
        return [
            *self._pick_public_steps(source),
            *self._place_public_steps(source, destination),
        ]

    def _handle_control_pose(self, request, response):
        if bool(request.execute):
            response.success = False
            response.message = (
                "Direct execute=true motion is disabled on the central task server; "
                "use /llm_arm/preview_command then /llm_arm/execute_preview."
            )
            return response
        return super()._handle_control_pose(request, response)

    @staticmethod
    def _is_retry_record(record):
        return bool(
            record
            and len(record.enriched_actions) == 1
            and record.enriched_actions[0].get("type") == "retry_place"
        )

    def _record_holding_valid_locked(self, record):
        if self._is_retry_record(record):
            return (
                self._pending_place is not None
                and self._held_source is not None
                and self._held_source == self._pending_place[0]
            )
        if record is None:
            return False
        visual = next(
            (
                action
                for action in record.enriched_actions
                if action.get("type") in ("pick", "place", "pick_place")
            ),
            None,
        )
        if visual is None:
            return True
        if visual["type"] == "place":
            return (
                self._pending_place is None
                and self._held_source is not None
                and self._held_source == visual["source"]
            )
        return self._held_source is None and self._pending_place is None

    def _retry_preview(self, session_id, response):
        with self._lock:
            pending = self._pending_place
            state = self._state
            retry_epoch = self._safety.epoch
            cached_fallback = self._cached_box_fallback_available
        if state != "HOLDING_RECOVERY" or pending is None:
            response.message = "No suspended placement is waiting for a confirmed retry."
            return response
        source, destination = pending
        current = self._fresh_match(destination)
        use_cached = current is None and cached_fallback
        if current is None:
            if not use_cached:
                response.message = "Box is not currently detectable with valid depth and TF."
                return response
            current = destination
        if self._xy_shift(destination, current) > self.box_max_shift_m:
            response.message = "Box shifted beyond the configured safe retry range."
            return response
        poses = self._place_preview_poses(source, current)
        self._check_pose(poses["approach_box"])
        self._check_pose(poses["release"])
        preview_id = uuid.uuid4().hex
        plan = TaskPlan(({"type": "retry_place"},))
        preview = TaskPreview(preview_id, plan, time.monotonic(), self.preview_max_age_sec)
        steps = self._place_public_steps(source, current)
        if use_cached:
            age_sec = max(
                0.0,
                (self.get_clock().now().nanoseconds - current.frame_stamp_ns) / 1e9,
            )
            steps[0] = {
                "type": "manual_cached_box_pose",
                "base_xyz": list(current.xyz),
                "yaw": current.yaw,
                "frame_stamp_ns": current.frame_stamp_ns,
                "detection_age_sec": age_sec,
                "warning": "box is not currently visible; operator confirmation required",
            }
        public = {
            "version": 1,
            "preview_id": preview_id,
            "frame_id": self.base_frame,
            "instruction": (
                "retry suspended placement with manually confirmed cached box pose"
                if use_cached else "retry suspended placement"
            ),
            "actions": [{"type": "retry_place", "use_cached_box_pose": use_cached}],
            "detections": [current.public()],
            "steps": steps,
            "valid_for_sec": self.preview_max_age_sec,
            "checks": (
                ["manual_cached_box_confirmation", "workspace", "collision_aware_ik"]
                if use_cached
                else ["fresh_rgbd", "depth", "tf", "workspace", "collision_aware_ik"]
            ),
        }
        record = PreviewRecord(
            preview,
            session_id,
            RETRY_PENDING_PLACE,
            [{
                "type": "retry_place",
                "source": source,
                "destination": current,
                "use_cached_box_pose": use_cached,
            }],
            retry_epoch,
            public,
        )
        with self._lock:
            motion_block_reason = self._motion_block_reason_locked()
            if (
                self._state != "HOLDING_RECOVERY"
                or self._pending_place != pending
                or not safety_execution_valid(self._safety, retry_epoch)
                or motion_block_reason
            ):
                detail = f" ({motion_block_reason})" if motion_block_reason else ""
                response.message = (
                    f"Suspended placement changed while retry preview was generated{detail}."
                )
                return response
            self._previews[preview_id] = record
        response.accepted = True
        response.status = "ready"
        response.preview_id = preview_id
        response.preview_json = json.dumps(public, ensure_ascii=False)
        if use_cached:
            response.message = (
                "Vision still cannot confirm the box. Review the cached base_link pose above; "
                "press y to accept it or n to keep holding the object."
            )
        else:
            response.message = "Retry preview ready. Press y to run re-detection and placement."
        return response

    def _preview_command(self, request, response):
        response.accepted = False
        response.status = "rejected"
        instruction = str(request.instruction).strip()
        session_id = str(request.session_id).strip() or uuid.uuid4().hex
        if not instruction:
            response.message = "Instruction is empty."
            return response
        with self._lock:
            state = self._state
            motion_block_reason = self._motion_block_reason_locked()
            preview_epoch = self._safety.epoch
        if motion_block_reason:
            response.message = (
                f"Motion is blocked ({motion_block_reason}); press r after the stop "
                "condition is safe."
            )
            return response
        if instruction == RETRY_PENDING_PLACE:
            if state in ("STOPPED", "RESETTING", "RESET_FAILED"):
                response.message = (
                    "Motion is stopped or resetting; resume before creating a retry preview."
                )
                return response
            try:
                return self._retry_preview(session_id, response)
            except ValueError as exc:
                response.message = str(exc)
                return response
        if state == "HOLDING_RECOVERY":
            response.message = "Arm is holding an object; request a retry preview or press h for recovery."
            return response
        if state in ("STOPPED", "RESETTING", "RESET_FAILED"):
            response.message = "Motion is stopped or resetting; press r after the stop condition is safe."
            return response
        if state == "EXECUTING":
            response.message = "A task is already executing."
            return response
        try:
            frame = self._current_frame()
            if instruction_has_visual_intent(instruction):
                metadata = self._wait_for_planning_metadata()
            else:
                metadata = self._metadata(frame)
            plan = self._llm_plan(session_id, instruction, metadata)
            enriched, detections, steps = self._enrich_plan(plan)
            preview_id = uuid.uuid4().hex
            preview = TaskPreview(preview_id, plan, time.monotonic(), self.preview_max_age_sec)
            public = {
                "version": 1,
                "preview_id": preview_id,
                "frame_id": self.base_frame,
                "instruction": instruction,
                "actions": [dict(action) for action in plan.actions],
                "detections": detections,
                "steps": steps,
                "valid_for_sec": self.preview_max_age_sec,
                "checks": ["fresh_rgbd", "depth", "tf", "workspace", "collision_aware_ik"],
            }
            record = PreviewRecord(
                preview, session_id, instruction, enriched, preview_epoch, public
            )
            with self._lock:
                motion_block_reason = self._motion_block_reason_locked()
                if (
                    not safety_execution_valid(self._safety, preview_epoch)
                    or motion_block_reason
                    or self._state in ("STOPPED", "RESETTING", "EXECUTING", "HOLDING_RECOVERY")
                ):
                    detail = f": {motion_block_reason}" if motion_block_reason else ""
                    raise ValueError(
                        f"motion safety state changed while preview was generated{detail}"
                    )
                self._previews[preview_id] = record
                self._state = "PREVIEW_READY"
            response.accepted = True
            response.status = "ready"
            response.preview_id = preview_id
            response.preview_json = json.dumps(public, ensure_ascii=False)
            response.message = (
                f"Preview ready. Press y within {self.preview_max_age_sec:g} seconds "
                "to execute the complete plan."
            )
        except ClarificationRequired as exc:
            response.status = "clarification_required"
            response.message = str(exc)
        except ValueError as exc:
            response.message = str(exc)
        except Exception as exc:
            response.message = f"Preview rejected: {exc}"
        return response

    def _goal_callback(self, goal_request):
        rejection_reason = ""
        with self._lock:
            record = self._previews.get(goal_request.preview_id)
            is_retry = self._is_retry_record(record)
            motion_block_reason = self._motion_block_reason_locked()
            if motion_block_reason:
                rejection_reason = motion_block_reason
            elif self._state in ("STOPPED", "RESETTING", "RESET_FAILED", "EXECUTING"):
                rejection_reason = f"state={self._state}"
            elif self._state == "HOLDING_RECOVERY" and not is_retry:
                rejection_reason = "holding recovery accepts only retry_place"
            elif is_retry and self._state != "HOLDING_RECOVERY":
                rejection_reason = "retry_place requires HOLDING_RECOVERY"
            elif record is None:
                rejection_reason = "preview id is unknown"
            elif record.session_id != goal_request.session_id:
                rejection_reason = "preview belongs to a different CLI session"
            elif record.safety_epoch != self._safety.epoch:
                rejection_reason = "safety epoch changed"
            elif not self._record_holding_valid_locked(record):
                rejection_reason = "holding state changed"
            else:
                status = preview_status(
                    record.preview, consumed_preview_ids=self._consumed_previews
                )
                if status != "ready":
                    rejection_reason = f"preview status is {status}"
        if rejection_reason:
            self.get_logger().warning(
                f"Execute preview rejected ({goal_request.preview_id}): {rejection_reason}."
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle):
        if self.abort.request_abort("execute action cancelled", command="stop"):
            self.abort.cancel_all_motion_now()
        return CancelResponse.ACCEPT

    @staticmethod
    def _xy_shift(a, b):
        return math.hypot(a.xyz[0] - b.xyz[0], a.xyz[1] - b.xyz[1])

    def _fresh_match(self, old):
        frame = self._current_frame()
        matches = []
        if frame is not None:
            for item in self._metadata(frame):
                if item["class_name"] == old.class_name:
                    resolved = self._resolve_candidate(item["index"], frame)
                    if resolved is not None:
                        matches.append(resolved)
        return min(matches, key=lambda candidate: self._xy_shift(old, candidate)) if matches else None

    def _revalidate_candidate(self, previous, label, max_shift_m):
        current = self._fresh_match(previous)
        if current is None:
            raise ValueError(f"{label} is no longer detectable")
        threshold_mm = max_shift_m * 1000.0
        if self._xy_shift(previous, current) > max_shift_m:
            raise ValueError(
                f"{label} moved more than {threshold_mm:g} mm; regenerate preview"
            )
        return current

    def _revalidate_action(self, action):
        action_type = action["type"]
        if action_type in ("pick", "pick_place"):
            action["source"] = self._revalidate_candidate(
                action["source"], "pick target", self.source_revalidate_threshold_m
            )
        if action_type in ("place", "pick_place"):
            action["destination"] = self._revalidate_candidate(
                action["destination"], "box", self.box_max_shift_m
            )

    def _revalidate(self, record):
        with self._lock:
            if not self._record_holding_valid_locked(record):
                raise ValueError("held-object state changed; regenerate preview")
        for action in record.enriched_actions:
            if action["type"] in ("pick", "place", "pick_place"):
                self._revalidate_action(action)

    def _feedback(self, goal_handle, index, count, phase, message, pose=None):
        feedback = ExecutePreview.Feedback()
        feedback.step_index = int(index)
        feedback.step_count = int(count)
        feedback.phase = str(phase)
        feedback.message = str(message)
        if pose is not None:
            feedback.active_target = pose
        goal_handle.publish_feedback(feedback)

    def _move_pose(self, pose, name, cartesian=False, velocity=None):
        return self.motion.move_to_pose(
            pose,
            planning_client="fairino",
            cartesian=cartesian,
            action_name=name,
            max_velocity=self.arm_max_velocity if velocity is None else velocity,
            max_acceleration=self.arm_max_acceleration if velocity is None else velocity,
            max_step_size=self.max_step_size,
            allowed_planning_time=self.allowed_planning_time,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            allowed_start_tolerance=self.allowed_start_tolerance,
            timeout_sec=self.execute_timeout_sec,
        )

    def _go_home(self):
        return self.motion.move_to_joints(
            self.home_joints, action_name="Return Home", planning_client="fairino", timeout_sec=180.0
        )

    def _wait_execution_stopped(self, timeout_sec):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while time.monotonic() <= deadline:
            with self._lock:
                if not self._execution_active:
                    return True
            if self.abort.is_stop_requested():
                return False
            time.sleep(0.02)
        return False

    def _recovery_complete(self, ok_home):
        released = self.abort.recovery_released()
        stopped = self.abort.is_stop_requested()
        with self._lock:
            self._previews.clear()
            if released:
                self._held_source = None
                self._pending_place = None
                self._cached_box_fallback_available = False
            if ok_home:
                self._reset_failed = False
                self._safety = complete_safety_reset(self._safety)
                self._state = "STOPPED" if self._execution_active else "IDLE"
            elif stopped:
                self._reset_failed = False
                self._state = "STOPPED"
            else:
                self._reset_failed = True
                self._state = "RESET_FAILED"

    def _collect_box_samples(self, old, execution_epoch=None, goal_handle=None):
        deadline = time.monotonic() + self.box_retarget_timeout_sec
        samples = deque(maxlen=self.box_sample_count)
        seen_stamps = set()
        unstable_windows = 0
        latest = None
        while time.monotonic() < deadline:
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return None, "task stopped while re-detecting box"
            candidate = self._fresh_match(old)
            if candidate is not None and candidate.frame_stamp_ns not in seen_stamps:
                seen_stamps.add(candidate.frame_stamp_ns)
                samples.append(candidate.xyz)
                latest = candidate
                if len(samples) == self.box_sample_count:
                    try:
                        relocation = decide_box_relocation(
                            old.xyz,
                            samples,
                            sample_count=self.box_sample_count,
                            retarget_threshold_m=self.box_retarget_threshold_m,
                            max_shift_m=self.box_max_shift_m,
                            stability_threshold_m=self.box_stability_threshold_m,
                        )
                    except ValueError as exc:
                        if "not stable" in str(exc):
                            unstable_windows += 1
                            continue
                        return None, str(exc)
                    if relocation.decision == "reject":
                        return None, (
                            f"box shifted more than {self.box_max_shift_m * 1000.0:g} mm"
                        )
                    return ResolvedCandidate(
                        latest.index,
                        latest.class_name,
                        latest.confidence,
                        latest.center_uv,
                        relocation.target_xyz,
                        latest.yaw,
                        latest.frame_stamp_ns,
                        latest.depth_inlier_ratio,
                    ), relocation.decision
            time.sleep(0.02)
        return None, (
            f"box was not stable within {self.box_retarget_timeout_sec:g} seconds: "
            f"{len(seen_stamps)} fresh frames, {unstable_windows} unstable windows; "
            f"requires {self.box_sample_count} stable frames"
        )

    def _execute_place_tail(
        self,
        source,
        destination,
        goal_handle=None,
        start_index=0,
        step_count=1,
        execution_epoch=None,
        use_cached_destination=False,
    ):
        if execution_epoch is not None and self._execution_interrupted(execution_epoch, goal_handle):
            self._mark_stop_state()
            return False, "task stopped before box re-detection"
        if use_cached_destination:
            with self._lock:
                cached_allowed = (
                    self._cached_box_fallback_available
                    and self._pending_place == (source, destination)
                    and self._held_source == source
                )
                if cached_allowed:
                    self._cached_box_fallback_available = False
            if not cached_allowed:
                return False, "cached box pose confirmation is no longer valid"
            current = self._fresh_match(destination)
            if current is not None:
                shift = self._xy_shift(destination, current)
                if shift > self.box_retarget_threshold_m:
                    return False, (
                        "box became visible and moved beyond the cached-pose tolerance; "
                        "create a fresh retry preview"
                    )
                updated, decision = current, "cached_pose_verified"
            else:
                updated, decision = destination, "manual_cached_pose"
            if goal_handle is not None:
                self._feedback(
                    goal_handle,
                    start_index + 1,
                    step_count,
                    "cached_box_pose",
                    decision,
                )
        else:
            with self._lock:
                self._cached_box_fallback_available = False
            if goal_handle is not None:
                self._feedback(
                    goal_handle,
                    start_index + 1,
                    step_count,
                    "re_detect_box",
                    f"collecting {self.box_sample_count} fresh stable frames",
                )
            updated, decision = self._collect_box_samples(
                destination, execution_epoch=execution_epoch, goal_handle=goal_handle
            )
        if updated is None:
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return False, decision
            with self._lock:
                self._state = "HOLDING_RECOVERY"
                self._pending_place = (source, destination)
                self._cached_box_fallback_available = (
                    "0 fresh frames, 0 unstable windows" in decision
                )
            return False, decision
        poses = self._place_preview_poses(source, updated)
        sequence = (
            ("approach_box", poses["approach_box"], False, 0.5),
            ("release", poses["release"], True, 0.2),
        )
        for offset, (name, pose, cartesian, velocity) in enumerate(sequence, 2):
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return False, "task stopped"
            if goal_handle is not None:
                self._feedback(goal_handle, start_index + offset, step_count, name, decision, pose)
            if not self._move_pose(pose, name, cartesian, velocity):
                self._mark_holding_recovery(source, destination)
                return False, f"{name} failed"
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return False, "task stopped"
        operations = (
            ("release_gripper", lambda: self._apply_gripper(abs(self.open_finger_position) * 2.0)),
            ("box_retreat", lambda: self._move_pose(poses["approach_box"], "box_retreat", True, 0.5)),
            ("return_home", self._go_home),
            ("final_gripper_close", lambda: self._apply_gripper(0.0)),
        )
        for offset, (name, run) in enumerate(operations, 4):
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return False, "task stopped"
            if goal_handle is not None:
                self._feedback(goal_handle, start_index + offset, step_count, name, "executing")
            if not run():
                if name == "release_gripper":
                    self._mark_holding_recovery(source, destination)
                return False, f"{name} failed"
            if name == "release_gripper":
                with self._lock:
                    self._held_source = None
                    self._pending_place = None
                    self._cached_box_fallback_available = False
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return False, "task stopped"
        with self._lock:
            self._held_source = None
            self._pending_place = None
            self._cached_box_fallback_available = False
        return True, f"placed into box ({decision})"

    def _execute_pick(self, source, goal_handle, step_index, step_count, execution_epoch):
        poses = self._pick_preview_poses(source)
        sequence = (
            ("open_gripper", lambda: self._apply_gripper(abs(self.open_finger_position) * 2.0)),
            ("home", self._go_home),
            ("approach_pick", lambda: self._move_pose(poses["approach_pick"], "approach_pick", False, 0.50)),
            ("grasp", lambda: self._move_pose(poses["grasp"], "grasp", True, 0.2)),
            ("close_gripper", lambda: self._apply_gripper(0.0)),
            ("carry", lambda: self._move_pose(poses["carry"], "carry", True, 0.2)),
        )
        for offset, (name, run) in enumerate(sequence, 1):
            if self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_stop_state()
                return False, "task stopped"
            self._feedback(goal_handle, step_index + offset, step_count, name, "executing")
            if not run():
                with self._lock:
                    holding = self._held_source is not None
                if holding:
                    with self._lock:
                        self._state = "HOLDING"
                return False, f"{name} failed"
            if name == "close_gripper":
                with self._lock:
                    self._held_source = source
            if self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_stop_state()
                return False, "task stopped"
        return True, "pick complete; holding object"

    def _execute_pick_place(self, action, goal_handle, step_index, step_count, execution_epoch):
        source, destination = action["source"], action["destination"]
        ok, message = self._execute_pick(
            source, goal_handle, step_index, step_count, execution_epoch
        )
        if not ok:
            with self._lock:
                holding = self._held_source is not None
            if holding and not self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_holding_recovery(source, destination)
            return ok, message
        return self._execute_place_tail(
            source,
            destination,
            goal_handle,
            step_index + 6,
            step_count,
            execution_epoch,
        )

    def _execute_preview(self, goal_handle):
        request = goal_handle.request
        result = ExecutePreview.Result()
        with self._lock:
            record = self._previews.get(request.preview_id)
            execution_epoch = self._safety.epoch
            if self._execution_active:
                goal_handle.abort()
                result.terminal_state = "REJECTED"
                result.message = "another task began execution before this goal"
                return result
            if (
                record is None
                or record.safety_epoch != execution_epoch
                or not safety_execution_valid(self._safety, execution_epoch)
                or self.abort.is_set()
                or not self._record_holding_valid_locked(record)
            ):
                goal_handle.abort()
                result.terminal_state = "STOPPED"
                result.message = "motion is stopped or preview is unavailable"
                return result
            try:
                self._consumed_previews = consume_preview(
                    record.preview, consumed_preview_ids=self._consumed_previews
                )
            except (AttributeError, ValueError) as exc:
                goal_handle.abort()
                result.terminal_state, result.message = "REJECTED", str(exc)
                return result
            self._execution_active = True
            self._state = "EXECUTING"
        try:
            self._revalidate(record)
            if self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_stop_state()
                goal_handle.abort()
                result.terminal_state, result.message = "STOPPED", "task invalidated before execution"
                return result
            step_count = execution_step_count(record.enriched_actions)
            step_index = 0
            for action in record.enriched_actions:
                if self._execution_interrupted(execution_epoch, goal_handle):
                    self._mark_stop_state()
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                    else:
                        goal_handle.abort()
                    result.terminal_state, result.message = "STOPPED", "task cancelled"
                    return result
                action_type = action["type"]
                if action_type == "pick":
                    ok, message = self._execute_pick(
                        action["source"], goal_handle, step_index, step_count, execution_epoch
                    )
                    step_index += execution_step_count([action])
                elif action_type == "place":
                    ok, message = self._execute_place_tail(
                        action["source"],
                        action["destination"],
                        goal_handle,
                        step_index,
                        step_count,
                        execution_epoch,
                    )
                    step_index += execution_step_count([action])
                elif action_type == "pick_place":
                    ok, message = self._execute_pick_place(
                        action, goal_handle, step_index, step_count, execution_epoch
                    )
                    step_index += execution_step_count([action])
                elif action_type == "retry_place":
                    ok, message = self._execute_place_tail(
                        action["source"],
                        action["destination"],
                        goal_handle,
                        step_index,
                        step_count,
                        execution_epoch,
                        use_cached_destination=bool(
                            action.get("use_cached_box_pose", False)
                        ),
                    )
                    step_index += execution_step_count([action])
                elif action_type in ("move_relative", "move_absolute"):
                    self._feedback(goal_handle, step_index + 1, step_count, action_type, "executing", action["target_pose"])
                    ok = self._move_pose(action["target_pose"], action_type)
                    message = f"{action_type} {'done' if ok else 'failed'}"
                    step_index += 1
                elif action_type == "set_gripper":
                    width = abs(self.open_finger_position) * 2.0 if action["state"] == "open" else 0.0
                    self._feedback(
                        goal_handle, step_index + 1, step_count, "set_gripper",
                        f"state={action['state']}, width={width:.4f} m",
                    )
                    ok = self._apply_gripper(width)
                    message = f"gripper {action['state']} {'done' if ok else 'failed'}"
                    if ok and action["state"] == "open":
                        with self._lock:
                            self._held_source = None
                            self._pending_place = None
                            self._cached_box_fallback_available = False
                    step_index += 1
                else:
                    self._feedback(
                        goal_handle, step_index + 1, step_count, "home",
                        f"target_joints={list(self.home_joints)}",
                    )
                    ok = self._go_home()
                    message = f"Home {'done' if ok else 'failed'}"
                    step_index += 1
                if self._execution_interrupted(execution_epoch, goal_handle):
                    self._mark_stop_state()
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                    else:
                        goal_handle.abort()
                    result.terminal_state, result.message = "STOPPED", "task invalidated by stop/reset"
                    return result
                if not ok:
                    goal_handle.abort()
                    result.terminal_state = (
                        self._state
                        if self._state in ("HOLDING", "HOLDING_RECOVERY")
                        else "FAILED"
                    )
                    result.message = message
                    return result
            goal_handle.succeed()
            result.success = True
            with self._lock:
                terminal_state = self._resting_state_locked()
                self._state = terminal_state
            result.terminal_state = terminal_state if terminal_state == "HOLDING" else "COMPLETED"
            picked = any(action["type"] == "pick" for action in record.enriched_actions)
            if terminal_state == "HOLDING":
                result.message = (
                    "pick complete; holding object"
                    if picked
                    else "complete plan executed; still holding object"
                )
            else:
                result.message = "complete plan executed"
            return result
        except Exception as exc:
            goal_handle.abort()
            if self._execution_interrupted(execution_epoch, goal_handle):
                self._mark_stop_state()
                result.terminal_state, result.message = "STOPPED", str(exc)
            else:
                result.terminal_state, result.message = "FAILED", str(exc)
            return result
        finally:
            self._mark_stop_state()
            with self._lock:
                self._execution_active = False
                if self._state == "EXECUTING":
                    self._state = self._resting_state_locked()
                elif self._state == "STOPPED" and not self._safety.blocked:
                    self._state = self._resting_state_locked()

    def _status(self, _request, response):
        frame = self._current_frame()
        with self._lock:
            state = self._state
            pending = self._pending_place is not None
            holding = self._held_source is not None
            cached_box_fallback = self._cached_box_fallback_available
            yolo_buffer_count = len(getattr(self, "_yolo_frames", ()))
            depth_buffer_count = len(getattr(self, "_depth_frames", ()))
            yolo_publisher_count = self._publisher_count(
                getattr(self, "yolo_topic", "/Yolov8_Inference")
            )
            depth_publisher_count = self._publisher_count(
                getattr(self, "depth_topic", "/Yolov8_Inference/depth")
            )
        response.success = True
        response.message = json.dumps({
            "state": state,
            "fresh_detection": frame is not None,
            "candidate_count": len(self._metadata(frame)),
            "yolo_buffer_count": yolo_buffer_count,
            "depth_buffer_count": depth_buffer_count,
            "yolo_publisher_count": yolo_publisher_count,
            "depth_publisher_count": depth_publisher_count,
            "rgb_depth_delta_sec": None if frame is None else frame["sync_delta_sec"],
            "camera_info_ready": getattr(self, "_camera_intrinsics", None) is not None,
            "holding": holding,
            "holding_recovery": pending,
            "cached_box_fallback": cached_box_fallback,
            "recovery_active": self.abort.recovery_active(),
            "reset_message": self.abort.recovery_message(),
            "credential_source": "configured" if self._client_key else "not_loaded",
        }, ensure_ascii=False)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LlmYoloTaskServer()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.execute_action.destroy()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
