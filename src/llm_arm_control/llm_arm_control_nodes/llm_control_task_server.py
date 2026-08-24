#!/usr/bin/env python3
"""Central DeepSeek, YOLO RGB-D, and robot task server."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from llm_arm_control.action import ExecutePreview
from llm_arm_control.srv import PreviewCommand
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
import tf2_ros

from llm_arm_control_nodes.deepseek_client import DeepSeekClient
from llm_arm_control_nodes.deepseek_credentials import get_deepseek_api_key
from llm_arm_control_nodes.robot_pose_control_server import RobotPoseControlServer
from visual_perception_nodes.llm_visual_perception import (
    PerceptionUnavailable,
    RgbdPerception,
)
from llm_arm_control_nodes.task_logic import (
    ClarificationRequired,
    DetectionCandidate,
    SafetyState,
    SYSTEM_PROMPT,
    TaskPlan,
    TaskPreview,
    apply_safety_command,
    build_semantic_history,
    complete_safety_reset,
    deterministic_visual_plan,
    execution_step_count,
    instruction_has_visual_intent,
    parse_llm_plan,
    preview_status,
    safety_execution_valid,
    validate_plan_intent,
    validate_visual_state,
)
from llm_arm_control_nodes.task.llm_control_state_machine import (
    LlmControlTaskState,
    LlmControlTaskStateMachine,
    LlmGraspnetState,
    LlmGraspnetStateMachine,
)
from graspnet_bringup.task.graspnet_candidate_utils import (
    build_candidates,
    prepare_candidate,
)
from graspnet_bringup.task.candidate_ros import pose_to_base
from llm_arm_control_nodes.task.preview_store import PreviewRecord, clear_session, prune, take


class LlmControlTaskServer(RobotPoseControlServer):
    def __init__(self):
        super().__init__("llm_control_task_server")
        self._declare_task_parameters()
        self._read_task_parameters()
        self._lock = threading.RLock()
        self._previews: dict[str, PreviewRecord] = {}
        self._sessions: dict[str, list[dict]] = {}
        self._client = None
        self._client_key = None
        self._state = LlmControlTaskState.PREGRASP_POSE.value
        self._safety = SafetyState()
        self.active_mode = "yolo"
        self._graspnet_state = LlmGraspnetState.WAIT_G.value
        self._graspnet_g_requested = False
        self._graspnet_compute_cancelled = False
        self._graspnet_last_error = ""
        self._mode_switch_error = ""
        self._mode_switch_active = False
        self._graspnet_result_lock = threading.RLock()
        self._graspnet_poses = None
        self._graspnet_scores = []
        self._graspnet_metadata = []
        self._graspnet_seq = 0
        self._graspnet_start_seq = 0
        self._graspnet_candidate = None
        self._execution_active = False
        self._reset_failed = False
        self._held_source = None

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
        self.abort.set_command_enabled(lambda: self.active_mode in ("yolo", "graspnet"))
        mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/llm_control/active_mode", self._on_active_mode, mode_qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String, "/motion_control/command", self._on_llm_motion_command, 10,
            callback_group=self.callback_group,
        )
        self.graspnet_compute_client = self.create_client(
            Trigger, "/grasp/compute", callback_group=self.callback_group
        )
        self.llm_yolo_inference_client = self.create_client(
            SetBool, "/llm_visual_perception/set_inference_enabled", callback_group=self.callback_group
        )
        self.llm_yolo_release_gpu_client = self.create_client(
            Trigger, "/llm_visual_perception/release_gpu", callback_group=self.callback_group
        )
        self.graspnet_release_gpu_client = self.create_client(
            Trigger, "/grasp/release_gpu", callback_group=self.callback_group
        )
        self.create_subscription(PoseArray, self.graspnet_poses_topic, self._on_graspnet_poses, 10,
                                 callback_group=self.callback_group)
        self.create_subscription(Float32MultiArray, self.graspnet_scores_topic, self._on_graspnet_scores, 10,
                                 callback_group=self.callback_group)
        self.create_subscription(Float32MultiArray, self.graspnet_metadata_topic, self._on_graspnet_metadata, 10,
                                 callback_group=self.callback_group)
        self.clear_session_subscription = self.create_subscription(
            String,
            "/llm_control/clear_session",
            self._clear_session,
            10,
            callback_group=self.callback_group,
        )
        self.preview_service = self.create_service(
            PreviewCommand,
            "/llm_control/preview_command",
            self._preview_command,
            callback_group=self.callback_group,
        )
        self.status_service = self.create_service(
            Trigger, "/llm_control/status", self._status, callback_group=self.callback_group
        )
        self.execute_action = ActionServer(
            self,
            ExecutePreview,
            "/llm_control/execute_preview",
            execute_callback=self._execute_preview,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.callback_group,
        )
        self.abort.set_recovery_hooks(
            open_gripper_fn=self._open_gripper,
            close_gripper_fn=self._close_gripper,
            go_home_fn=self._move_to_pregrasp_pose,
            recovery_complete_fn=self._recovery_complete,
            wait_task_stopped_fn=self._wait_execution_stopped,
            stop_timeout_sec=self.reset_stop_timeout_sec,
        )
        self._state_machine = LlmControlTaskStateMachine(self)
        self._graspnet_state_machine = LlmGraspnetStateMachine(self)
        self._pregrasp_timer = self.create_timer(
            0.2, self._tick_state_machines, callback_group=self.callback_group
        )
        self.get_logger().info(
            "LLM control task server ready: /llm_control/preview_command, /llm_control/execute_preview"
        )

    def _declare_task_parameters(self):
        defaults = {
            "yolo_topic": "/yolo/detected_result",
            "depth_topic": "/yolo/detected_result/depth",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            "preview_max_age_sec": 15.0,
            "detection_max_age_sec": 1.0,
            "rgb_depth_tolerance_sec": 0.05,
            "vision_wait_timeout_sec": 15.0,
            "pick_classes": ["elongated_object", "cube", "stone"],
            "place_classes": ["box"],
            "workspace_min_xy": [-0.9, -0.9],
            "workspace_max_xy": [0.9, 0.9],
            "pregrasp_pose.x": 0.1,
            "pregrasp_pose.y": 0.35,
            "pregrasp_pose.z": 0.30,
            "pregrasp_pose.roll": 0.0,
            "pregrasp_pose.pitch": -180.0,
            "pregrasp_pose.yaw": 100.0,
            "grasp_above": 0.04,
            "grasp_offset": 0.010,
            "place_offset": 0.08,
            "descend_to_box": 0.04,
            "grasp.elongated_object.roll": 0.0,
            "grasp.elongated_object.pitch": -180.0,
            "grasp.elongated_object.yaw_offset": 90.0,
            "grasp.cube.roll": 0.0,
            "grasp.cube.pitch": -180.0,
            "grasp.cube.yaw_offset": 0.0,
            "grasp.stone.roll": 0.0,
            "grasp.stone.pitch": -180.0,
            "grasp.stone.yaw_offset": -45.0,
            "reset_stop_timeout_sec": 5.0,
            "deepseek_base_url": "https://api.deepseek.com",
            "deepseek_model": "deepseek-chat",
            "deepseek_timeout_sec": 30.0,
            "use_continuous_yolo": True,
            "graspnet_poses_topic": "/grasp/poses",
            "graspnet_scores_topic": "/grasp/scores",
            "graspnet_metadata_topic": "/grasp/metadata",
            "graspnet_max_candidates": 50,
            "graspnet_approach_distance_m": 0.08,
            "graspnet_grasp_offset_m": -0.01,
            "graspnet_lift_distance": 0.08,
            "graspnet_to_ee_rpy_deg": [90.0, 0.0, 90.0],
            "graspnet_use_width": False,
            "graspnet_result_timeout_sec": 8.0,
            "graspnet_compute_timeout_sec": 600.0,
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
        self.workspace_min_xy = tuple(float(item) for item in value("workspace_min_xy"))
        self.workspace_max_xy = tuple(float(item) for item in value("workspace_max_xy"))
        if len(self.workspace_min_xy) != 2 or len(self.workspace_max_xy) != 2:
            raise ValueError("workspace XY bounds must each contain exactly two values")
        if any(lower > upper for lower, upper in zip(self.workspace_min_xy, self.workspace_max_xy)):
            raise ValueError("workspace XY lower bounds must not exceed upper bounds")
        self.pregrasp_pose_cfg = {
            axis: float(value(f"pregrasp_pose.{axis}"))
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.pregrasp_pose = self._build_pregrasp_pose()
        for name in (
            "grasp_above", "grasp_offset", "place_offset", "descend_to_box",
            "reset_stop_timeout_sec", "deepseek_timeout_sec",
        ):
            setattr(self, name, float(value(name)))
        self.grasp_profiles = {
            name: tuple(float(value(f"grasp.{name}.{axis}"))
                        for axis in ("roll", "pitch", "yaw_offset"))
            for name in ("elongated_object", "cube", "stone")
        }
        self.deepseek_base_url = str(value("deepseek_base_url"))
        self.deepseek_model = str(value("deepseek_model"))
        self.use_continuous_yolo = bool(value("use_continuous_yolo"))
        self.graspnet_poses_topic = str(value("graspnet_poses_topic"))
        self.graspnet_scores_topic = str(value("graspnet_scores_topic"))
        self.graspnet_metadata_topic = str(value("graspnet_metadata_topic"))
        self.graspnet_max_candidates = int(value("graspnet_max_candidates"))
        self.graspnet_approach_distance_m = float(value("graspnet_approach_distance_m"))
        self.graspnet_grasp_offset_m = float(value("graspnet_grasp_offset_m"))
        self.graspnet_lift_distance = float(value("graspnet_lift_distance"))
        self.graspnet_to_ee_rpy_deg = tuple(float(v) for v in value("graspnet_to_ee_rpy_deg"))
        self.graspnet_use_width = bool(value("graspnet_use_width"))
        self.graspnet_result_timeout_sec = float(value("graspnet_result_timeout_sec"))
        self.graspnet_compute_timeout_sec = float(value("graspnet_compute_timeout_sec"))

    def _tick_state_machines(self):
        self._state_machine.tick()
        self._graspnet_state_machine.tick()

    def _on_graspnet_poses(self, msg):
        with self._graspnet_result_lock:
            self._graspnet_poses = msg
            self._graspnet_seq += 1

    def _on_graspnet_scores(self, msg):
        with self._graspnet_result_lock:
            self._graspnet_scores = [float(value) for value in msg.data]

    def _on_graspnet_metadata(self, msg):
        with self._graspnet_result_lock:
            self._graspnet_metadata = [float(value) for value in msg.data]

    def _advance_safety(self, command):
        command = str(command).strip().lower()
        if command == "g":
            with self._lock:
                accepted = (
                    self.active_mode == "graspnet"
                    and self._graspnet_state == LlmGraspnetState.WAIT_G.value
                    and getattr(self, "_held_source", None) is None
                )
                if accepted:
                    self._graspnet_g_requested = False
                    self._graspnet_state = LlmGraspnetState.COMPUTE.value
                    return
                holding = getattr(self, "_held_source", None) is not None
            if holding:
                self.get_logger().warning("Ignoring GraspNet request while an object is held.")
            return
        if getattr(self, "active_mode", "yolo") not in ("yolo", "graspnet"):
            return
        if command in ("stop", "reset"):
            self._set_llm_yolo_inference(False)
            self._graspnet_g_requested = False
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

    def _on_llm_motion_command(self, msg):
        if str(msg.data).strip().lower() == "g":
            self._advance_safety("g")

    def _on_active_mode(self, msg):
        mode = str(msg.data).strip().lower()
        if mode not in ("yolo", "graspnet"):
            return
        with self._lock:
            if self._execution_active:
                self.get_logger().warning("Ignoring mode switch while LLM task is executing.")
                return
            if mode == "graspnet" and self._state not in (
                LlmControlTaskState.IDLE.value,
                LlmControlTaskState.HOLDING.value,
            ):
                self.get_logger().warning("Ignoring GraspNet mode entry outside LLM IDLE/HOLDING.")
                return
            if mode == "yolo" and self._graspnet_state != LlmGraspnetState.WAIT_G.value:
                self.get_logger().warning("Ignoring YOLO mode entry outside LLM GraspNet WAIT_G.")
                return
            if mode == self.active_mode:
                return
            if getattr(self, "_mode_switch_active", False):
                self.get_logger().warning("Ignoring duplicate LLM mode switch request.")
                return
            self._mode_switch_active = True
            self._mode_switch_error = ""
        try:
            if mode == "graspnet":
                if not self._set_llm_yolo_inference(False, force=True):
                    self._record_mode_switch_error("Failed to stop LLM YOLO inference before GraspNet mode.")
                    return
                if not self._release_llm_yolo_gpu():
                    self._record_mode_switch_error("Failed to release LLM YOLO GPU memory before GraspNet mode.")
                    return
            else:
                if not self._release_graspnet_gpu():
                    self._record_mode_switch_error("Failed to release GraspNet GPU memory before YOLO mode.")
                    return
                if not self._set_llm_yolo_inference(True, force=True):
                    self._record_mode_switch_error("Failed to reload LLM YOLO before YOLO mode.")
                    return
            with self._lock:
                if self._execution_active:
                    self._record_mode_switch_error("Mode switch was superseded by an active task.")
                    return
                self.active_mode = mode
                self._mode_switch_error = ""
                self._previews.clear()
                self._graspnet_g_requested = False
                if mode == "graspnet":
                    self._graspnet_reset()
                    self._graspnet_state = LlmGraspnetState.WAIT_G.value
                if mode == "yolo" and self._state != LlmControlTaskState.PREGRASP_POSE.value:
                    self._state = self._resting_state_locked()
            self.get_logger().info(f"LLM mode switched to {mode}.")
        finally:
            with self._lock:
                self._mode_switch_active = False

    def _record_mode_switch_error(self, message: str):
        with self._lock:
            self._mode_switch_error = message
        self.get_logger().error(message)

    def _set_llm_yolo_inference(self, enabled: bool, *, force: bool = False) -> bool:
        if self.use_continuous_yolo and not force:
            return True
        client = self.llm_yolo_inference_client
        if not client.wait_for_service(timeout_sec=5.0):
            return False
        future = client.call_async(SetBool.Request(data=bool(enabled)))
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        response = future.result() if future.done() else None
        if response is not None and response.success:
            self.perception.clear_frames()
            return True
        return False

    def _call_trigger(self, client, timeout_sec=60.0) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            return False
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        response = future.result() if future.done() else None
        return bool(response is not None and response.success)

    def _release_llm_yolo_gpu(self) -> bool:
        return self._call_trigger(self.llm_yolo_release_gpu_client)

    def _release_graspnet_gpu(self) -> bool:
        return self._call_trigger(self.graspnet_release_gpu_client)

    def _graspnet_reset(self):
        self._graspnet_g_requested = False
        self._graspnet_compute_cancelled = False
        self._graspnet_candidate = None
        self._graspnet_candidates = None

    def _graspnet_motion_failed(self, reason: str):
        self._graspnet_last_error = f"GraspNet motion failed: {reason}"
        self.get_logger().error(
            f"{self._graspnet_last_error}; stopped. Press h for one pregrasp reset."
        )
        self._stop_for_motion_failure(self._graspnet_last_error)

    def _graspnet_compute(self) -> bool:
        self._graspnet_compute_cancelled = False
        if not self.graspnet_compute_client.wait_for_service(timeout_sec=0.2):
            self._graspnet_last_error = "GraspNet compute service is unavailable"
            self.get_logger().warning(self._graspnet_last_error)
            return False
        with self._graspnet_result_lock:
            self._graspnet_start_seq = self._graspnet_seq
        future = self.graspnet_compute_client.call_async(Trigger.Request())
        deadline = time.monotonic() + self.graspnet_compute_timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if self.abort.is_set():
                return False
            time.sleep(0.02)
        response = future.result() if future.done() else None
        self._graspnet_compute_cancelled = bool(
            response is not None
            and not response.success
            and str(response.message).startswith("CANCELED:")
        )
        if response is None:
            self._graspnet_last_error = "GraspNet compute timed out"
            self.get_logger().error(self._graspnet_last_error)
            return False
        if not response.success:
            self._graspnet_last_error = str(response.message)
            self.get_logger().error(f"GraspNet compute failed: {self._graspnet_last_error}")
            return False
        self._graspnet_last_error = ""
        self.get_logger().info("GraspNet compute accepted.")
        return True

    def _graspnet_select(self) -> bool:
        deadline = time.monotonic() + self.graspnet_result_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            with self._graspnet_result_lock:
                if self._graspnet_poses is not None and self._graspnet_seq > self._graspnet_start_seq:
                    poses, scores, metadata = self._graspnet_poses, list(self._graspnet_scores), list(self._graspnet_metadata)
                    break
            time.sleep(0.02)
        else:
            self._graspnet_last_error = "GraspNet produced no result before timeout"
            return False
        candidates = build_candidates(
            poses.poses, scores, metadata, self.graspnet_max_candidates
        )
        self._graspnet_candidates = (poses.header, candidates)
        if not candidates:
            self._graspnet_last_error = "GraspNet produced no valid candidates"
            return False
        self._graspnet_last_error = ""
        return True

    def _graspnet_plan(self) -> bool:
        header, candidates = getattr(self, "_graspnet_candidates", (None, ()))
        if header is None:
            self._graspnet_last_error = "GraspNet candidate set is unavailable"
            return False
        for candidate in candidates:
            pose = self._graspnet_pose_to_base(header, candidate.camera_pose)
            if pose is None:
                continue
            candidate.base_pose = pose
            prepare_candidate(
                candidate,
                grasp_offset_m=self.graspnet_grasp_offset_m,
                orientation_rpy_deg=self.graspnet_to_ee_rpy_deg,
                approach_distance_m=self.graspnet_approach_distance_m,
                lift_distance_m=self.graspnet_lift_distance,
            )
            try:
                self._check_pose(self._pose_stamped(candidate.approach))
                self._check_pose(self._pose_stamped(candidate.grasp))
                self._check_pose(self._pose_stamped(candidate.lift))
            except ValueError:
                continue
            self._graspnet_candidate = candidate
            return True
        self._graspnet_last_error = "No executable GraspNet candidate"
        return False

    def _graspnet_preopen(self) -> bool:
        candidate = self._graspnet_candidate
        if candidate is None:
            return False
        if self.graspnet_use_width and candidate.preopen_positions is not None:
            return self.motion.control_gripper(open_gripper=False, positions=candidate.preopen_positions, timeout_sec=90.0)
        return self._apply_gripper(abs(self.open_finger_position) * 2.0)

    def _graspnet_move(self, name: str, cartesian: bool, velocity: float) -> bool:
        candidate = self._graspnet_candidate
        pose = getattr(candidate, name, None) if candidate is not None else None
        return pose is not None and self._move_pose(self._pose_stamped(pose), f"graspnet_{name}", cartesian, velocity)

    def _graspnet_pose_to_base(self, header, pose: Pose):
        return pose_to_base(
            self.tf_buffer,
            self.base_frame,
            header,
            pose,
            default_frame="camera_color_optical_frame",
        )

    def _pose_stamped(self, pose: Pose):
        stamped = PoseStamped()
        stamped.header.frame_id = self.base_frame
        stamped.pose = pose
        return stamped

    def _resting_state_locked(self):
        return "HOLDING" if self._held_source is not None else "IDLE"

    def _clear_holding_locked(self):
        self._held_source = None

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
            clear_session(self._sessions, self._previews, session_id)
            if self._state == "PREVIEW_READY" and not self._previews:
                self._state = self._resting_state_locked()
        self.get_logger().info(f"Cleared language session {session_id!r}.")

    def _prune_previews_locked(self, now=None):
        prune(self._previews, now)
        if self._state == "PREVIEW_READY" and not self._previews:
            self._state = self._resting_state_locked()

    def _take_preview_locked(self, preview_id):
        self._prune_previews_locked()
        return take(self._previews, preview_id)

    def _execution_interrupted(self, execution_epoch, goal_handle=None):
        with self._lock:
            valid = safety_execution_valid(self._safety, execution_epoch)
        cancel_requested = goal_handle is not None and goal_handle.is_cancel_requested
        return not valid or self.abort.is_set() or cancel_requested

    def _mark_stop_state(self):
        with self._lock:
            if self.abort.is_stop_requested() or self._safety.command == "stop":
                self._state = "STOPPED"

    def _stop_for_motion_failure(self, reason: str):
        if self.abort.request_abort(str(reason), command="stop"):
            self.abort.cancel_all_motion_now()
        with self._lock:
            self._state = "STOPPED"

    def _mark_holding_recovery(self, source, destination):
        with self._lock:
            if self._held_source is not None:
                self._held_source = source
                self._state = "HOLDING"

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
        candidates = [
            DetectionCandidate(item["index"], item["class_name"])
            for item in metadata
        ]
        try:
            plan = deterministic_visual_plan(
                instruction,
                metadata,
                current_xyz=(pose.position.x, pose.position.y, pose.position.z),
                pick_classes=self.pick_classes,
                place_classes=self.place_classes,
            )
            if plan is None:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
                messages.append({"role": "user", "content": json.dumps(context, ensure_ascii=False)})
                response_text = self._deepseek().chat(messages, self.deepseek_model)
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
        x, y = (float(value) for value in xyz[:2])
        return all(lower <= value <= upper for value, lower, upper in zip(
            (x, y), self.workspace_min_xy, self.workspace_max_xy
        ))

    def _pose_from_xyz_quat(self, xyz, quat):
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in xyz)
        pose.pose.orientation.x, pose.pose.orientation.y = float(quat[0]), float(quat[1])
        pose.pose.orientation.z, pose.pose.orientation.w = float(quat[2]), float(quat[3])
        return pose

    def _build_pregrasp_pose(self):
        cfg = self.pregrasp_pose_cfg
        quat = Rotation.from_euler(
            "xyz", [cfg["roll"], cfg["pitch"], cfg["yaw"]], degrees=True
        ).as_quat()
        return self._pose_from_xyz_quat((cfg["x"], cfg["y"], cfg["z"]), quat)

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
                validate_visual_state(
                    action_type,
                    holding=held_source is not None,
                    recovery=False,
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
                if source is not None:
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
                    public_steps.append({
                        "type": "pregrasp_pose",
                        "target_pose": self._pose_public(self.pregrasp_pose),
                    })
                    relative_base_known = False
        return enriched, list(detections.values()), public_steps

    def _grasp_quat(self, source):
        roll, pitch, yaw_offset = self.grasp_profiles[source.class_name]
        return Rotation.from_euler(
            "xyz", [roll, pitch, math.degrees(source.yaw) + yaw_offset], degrees=True
        ).as_quat()

    def _pick_heights(self, source):
        return (
            source.xyz[2] + self.grasp_offset,
            source.xyz[2] + self.grasp_above,
            source.xyz[2] + self.place_offset,
        )

    def _pick_preview_poses(self, source):
        grasp, approach, carry = self._pick_heights(source)
        quat = self._grasp_quat(source)
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
        if source is None:
            orientation = self.pregrasp_pose.pose.orientation
            quat = (orientation.x, orientation.y, orientation.z, orientation.w)
        else:
            quat = self._grasp_quat(source)
        return {
            "approach_box": self._pose_from_xyz_quat(
                (destination.xyz[0], destination.xyz[1], destination.xyz[2] + self.place_offset), quat
            ),
            "release": self._pose_from_xyz_quat(
                (destination.xyz[0], destination.xyz[1], destination.xyz[2] + self.descend_to_box), quat
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
            {"type": "approach_pick", "target_pose": self._pose_public(poses["approach_pick"]),
             "source": "vision"},
            {"type": "grasp", "target_pose": self._pose_public(poses["grasp"]),
             "source": "vision"},
            {"type": "close_gripper", "state": "close", "width_m": 0.0},
            {"type": "carry", "target_pose": self._pose_public(poses["carry"]),
             "source": "vision"},
            {
                "type": "return_pregrasp_pose",
                "target_pose": self._pose_public(self.pregrasp_pose),
            },
        ]

    def _place_public_steps(self, source, destination):
        poses = self._place_preview_poses(source, destination)
        open_width = abs(self.open_finger_position) * 2.0
        return [
            {"type": "approach_box", "target_pose": self._pose_public(poses["approach_box"]),
             "source": "vision_preview"},
            {"type": "release", "target_pose": self._pose_public(poses["release"]),
             "source": "vision_preview"},
            {"type": "release_gripper", "state": "open", "width_m": open_width},
            {
                "type": "return_pregrasp_pose",
                "target_pose": self._pose_public(self.pregrasp_pose),
            },
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
                "use /llm_control/preview_command then /llm_control/execute_preview."
            )
            return response
        return super()._handle_control_pose(request, response)

    def _record_holding_valid_locked(self, record):
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
            return self._held_source == visual["source"]
        return self._held_source is None

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
            active_mode = getattr(self, "active_mode", "yolo")
            motion_block_reason = self._motion_block_reason_locked()
            preview_epoch = self._safety.epoch
        if active_mode == "graspnet" and self._graspnet_state != LlmGraspnetState.WAIT_G.value:
            response.message = "LLM GraspNet accepts manual motion only in WAIT_G."
            return response
        if motion_block_reason:
            response.message = (
                f"Motion is blocked ({motion_block_reason}); press r after the stop "
                "condition is safe."
            )
            return response
        if state == LlmControlTaskState.PREGRASP_POSE.value:
            response.message = "Moving to pregrasp pose; wait until /llm_control/status reports IDLE."
            return response
        if state in ("STOPPED", "RESETTING", "RESET_FAILED"):
            response.message = "Motion is stopped or resetting; press r after the stop condition is safe."
            return response
        if state == "EXECUTING":
            response.message = "A task is already executing."
            return response
        with self._lock:
            self._state = LlmControlTaskState.SEARCHING.value
        visual_intent = instruction_has_visual_intent(instruction)
        if visual_intent and active_mode != "yolo":
            response.message = "Visual pick/place requires mode yolo."
            with self._lock:
                self._state = self._resting_state_locked()
            return response
        if visual_intent and not self._set_llm_yolo_inference(True):
            response.message = "YOLO inference is unavailable."
            with self._lock:
                self._state = self._resting_state_locked()
            return response
        try:
            if visual_intent:
                self.perception.current_frame()
                metadata = self.perception.wait_for_planning_metadata()
            else:
                metadata = []
            plan = self._llm_plan(session_id, instruction, metadata)
            if active_mode == "graspnet" and any(
                action["type"] not in ("move_relative", "move_absolute", "home")
                for action in plan.actions
            ):
                raise ValueError("GraspNet mode accepts absolute or relative base_link motion only")
            if any(action["type"] == "place" for action in plan.actions):
                with self._lock:
                    self._state = LlmControlTaskState.SEARCHING_BOX.value
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
                    or self._state in ("STOPPED", "RESETTING", "EXECUTING")
                ):
                    detail = f": {motion_block_reason}" if motion_block_reason else ""
                    raise ValueError(
                        f"motion safety state changed while preview was generated{detail}"
                    )
                self._previews[preview_id] = record
                self._state = LlmControlTaskState.PREVIEW_READY.value
            response.accepted = True
            response.status = "ready"
            response.preview_id = preview_id
            response.preview_json = json.dumps(public, ensure_ascii=False)
            response.message = (
                f"Preview ready. Press y within {self.preview_max_age_sec:g} seconds "
                "to execute the complete plan."
            )
        except (ClarificationRequired, PerceptionUnavailable) as exc:
            response.status = "clarification_required"
            response.message = str(exc)
        except ValueError as exc:
            response.message = str(exc)
        except Exception as exc:
            response.message = f"Preview rejected: {exc}"
        finally:
            if visual_intent:
                self._set_llm_yolo_inference(False)
        if not response.accepted:
            with self._lock:
                if self._state in (
                    LlmControlTaskState.SEARCHING.value,
                    LlmControlTaskState.SEARCHING_BOX.value,
                ):
                    self._state = self._resting_state_locked()
        return response

    def _goal_callback(self, goal_request):
        rejection_reason = ""
        with self._lock:
            self._prune_previews_locked()
            record = self._previews.get(goal_request.preview_id)
            motion_block_reason = self._motion_block_reason_locked()
            motion_only = record is not None and all(
                action.get("type") in ("move_relative", "move_absolute", "home")
                for action in record.enriched_actions
            )
            if getattr(self, "active_mode", "yolo") == "graspnet" and (
                not motion_only or self._graspnet_state != LlmGraspnetState.WAIT_G.value
            ):
                rejection_reason = "GraspNet mode accepts manual base_link motion only in WAIT_G"
            elif motion_block_reason:
                rejection_reason = motion_block_reason
            elif self._state in (
                LlmControlTaskState.PREGRASP_POSE.value,
                "STOPPED",
                "RESETTING",
                "RESET_FAILED",
                "EXECUTING",
            ):
                rejection_reason = f"state={self._state}"
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

    def _revalidate(self, record):
        with self._lock:
            if not self._record_holding_valid_locked(record):
                raise ValueError("held-object state changed; regenerate preview")
        pick_actions = [
            action for action in record.enriched_actions
            if action["type"] in ("pick", "pick_place")
        ]
        if not pick_actions:
            return
        if not self._set_llm_yolo_inference(True):
            raise ValueError("YOLO inference is unavailable for pick revalidation")
        try:
            self.perception.wait_for_planning_metadata()
            for action in pick_actions:
                source = self.perception.fresh_match(action["source"])
                if source is None:
                    raise ValueError("pick target is no longer detectable")
                action["source"] = source
        finally:
            self._set_llm_yolo_inference(False)

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

    def _move_to_pregrasp_pose(self):
        return self._move_pose(self.pregrasp_pose, "Move to pregrasp pose")

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
                self._graspnet_reset()
                self._graspnet_state = LlmGraspnetState.WAIT_G.value
                self._state = "STOPPED" if self._execution_active else "IDLE"
            elif stopped:
                self._reset_failed = False
                self._state = "STOPPED"
            else:
                self._reset_failed = True
                self._state = "RESET_FAILED"

    def _execute_preview(self, goal_handle):
        request = goal_handle.request
        result = ExecutePreview.Result()
        with self._lock:
            self._prune_previews_locked()
            record = self._previews.get(request.preview_id)
            execution_epoch = self._safety.epoch
            motion_only = record is not None and all(
                action.get("type") in ("move_relative", "move_absolute", "home")
                for action in record.enriched_actions
            )
            if getattr(self, "active_mode", "yolo") == "graspnet" and (
                not motion_only or self._graspnet_state != LlmGraspnetState.WAIT_G.value
            ):
                goal_handle.abort()
                result.terminal_state = "REJECTED"
                result.message = "GraspNet mode accepts manual base_link motion only in WAIT_G"
                return result
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
                ok, message, action_steps = self._state_machine.execute_action(
                    action, goal_handle, step_index, step_count, execution_epoch
                )
                step_index += action_steps
                if self._execution_interrupted(execution_epoch, goal_handle):
                    self._mark_stop_state()
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                    else:
                        goal_handle.abort()
                    result.terminal_state, result.message = "STOPPED", "task invalidated by stop/reset"
                    return result
                if not ok:
                    self._stop_for_motion_failure(message)
                    goal_handle.abort()
                    result.terminal_state = "STOPPED"
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
            holding = self._held_source is not None
            graspnet_request_pending = self._graspnet_g_requested
        response.success = True
        response.message = json.dumps({
            "state": state,
            "active_mode": getattr(self, "active_mode", "yolo"),
            "graspnet_state": self._graspnet_state,
            "graspnet_request_pending": graspnet_request_pending,
            "graspnet_last_error": self._graspnet_last_error,
            "mode_switch_error": self._mode_switch_error,
            **diagnostics,
            "holding": holding,
            "recovery_active": self.abort.recovery_active(),
            "reset_message": self.abort.recovery_message(),
            "credential_source": "configured" if self._client_key else "not_loaded",
        }, ensure_ascii=False)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LlmControlTaskServer()
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
