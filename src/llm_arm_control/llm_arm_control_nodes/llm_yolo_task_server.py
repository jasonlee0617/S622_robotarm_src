"""Central DeepSeek, YOLO RGB-D, and Fairino task server."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
import uuid

from geometry_msgs.msg import PoseStamped
from llm_arm_control.action import ExecutePreview
from llm_arm_control.srv import PreviewCommand
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros

from .deepseek_client import DeepSeekClient
from .deepseek_credentials import get_deepseek_api_key
from .fairino_pose_control_server import FairinoPoseControlServer
from .perception import ResolvedCandidate, RgbdPerception, xy_shift
from .task_logic import (
    ClarificationRequired,
    DetectionCandidate,
    RETRY_PENDING_PLACE,
    SafetyState,
    SYSTEM_PROMPT,
    TaskPlan,
    TaskPreview,
    apply_safety_command,
    build_semantic_history,
    complete_safety_reset,
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


@dataclass
class PreviewRecord:
    preview: TaskPreview
    session_id: str
    instruction: str
    enriched_actions: list[dict]
    safety_epoch: int
    public: dict


class LlmYoloTaskServer(FairinoPoseControlServer):
    def __init__(self):
        super().__init__("llm_yolo_task_server")
        self._declare_task_parameters()
        self._read_task_parameters()
        self._lock = threading.RLock()
        self._previews: dict[str, PreviewRecord] = {}
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
        self.perception = RgbdPerception(
            self,
            self.tf_buffer,
            base_frame=self.base_frame,
            yolo_topic=self.yolo_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            rgb_depth_tolerance_sec=self.rgb_depth_tolerance_sec,
            detection_max_age_sec=self.detection_max_age_sec,
            vision_wait_timeout_sec=self.vision_wait_timeout_sec,
            callback_group=self.callback_group,
        )
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

    def _clear_holding_locked(self):
        self._held_source = None
        self._pending_place = None
        self._cached_box_fallback_available = False

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

    def _prune_previews_locked(self, now=None):
        expired = [
            preview_id
            for preview_id, record in self._previews.items()
            if preview_status(record.preview, now) != "ready"
        ]
        for preview_id in expired:
            self._previews.pop(preview_id, None)
        if self._state == "PREVIEW_READY" and not self._previews:
            self._state = self._resting_state_locked()

    def _take_preview_locked(self, preview_id):
        self._prune_previews_locked()
        return self._previews.pop(preview_id, None)

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
        frame = self.perception.current_frame()
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
                    source = self.perception.resolve_candidate(action["source_index"], frame)
                    if source is None:
                        raise ValueError("selected pick target has invalid depth/TF")
                else:
                    source = held_source
                destination = None
                if action_type in ("place", "pick_place"):
                    destination = self.perception.resolve_candidate(
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
        current = self.perception.fresh_match(destination)
        use_cached = current is None and cached_fallback
        if current is None:
            if not use_cached:
                response.message = "Box is not currently detectable with valid depth and TF."
                return response
            current = destination
        if xy_shift(destination, current) > self.box_max_shift_m:
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
            self._prune_previews_locked()
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
            self._prune_previews_locked()
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
            frame = self.perception.current_frame()
            if instruction_has_visual_intent(instruction):
                metadata = self.perception.wait_for_planning_metadata()
            else:
                metadata = self.perception.metadata(frame)
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
            self._prune_previews_locked()
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
                status = preview_status(record.preview)
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

    def _revalidate_candidate(self, previous, label, max_shift_m):
        current = self.perception.fresh_match(previous)
        if current is None:
            raise ValueError(f"{label} is no longer detectable")
        threshold_mm = max_shift_m * 1000.0
        if xy_shift(previous, current) > max_shift_m:
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
        self.moveit2_arm.max_velocity = self.arm_max_velocity
        self.moveit2_arm.max_acceleration = self.arm_max_acceleration
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
                self._clear_holding_locked()
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
            candidate = self.perception.fresh_match(old)
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
            current = self.perception.fresh_match(destination)
            if current is not None:
                shift = xy_shift(destination, current)
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
                    self._clear_holding_locked()
            if execution_epoch is not None and self._execution_interrupted(
                execution_epoch, goal_handle
            ):
                self._mark_stop_state()
                return False, "task stopped"
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
            self._prune_previews_locked()
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
            if self._take_preview_locked(request.preview_id) is not record:
                goal_handle.abort()
                result.terminal_state = "REJECTED"
                result.message = "preview expired before execution began"
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
                            self._clear_holding_locked()
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
        diagnostics = self.perception.diagnostics()
        with self._lock:
            state = self._state
            pending = self._pending_place is not None
            holding = self._held_source is not None
            cached_box_fallback = self._cached_box_fallback_available
        response.success = True
        response.message = json.dumps({
            "state": state,
            **diagnostics,
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
