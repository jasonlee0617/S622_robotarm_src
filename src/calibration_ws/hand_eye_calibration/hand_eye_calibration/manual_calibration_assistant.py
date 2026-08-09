#!/usr/bin/env python3
"""Strict companion for the easy_handeye2 manual calibration GUI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import select
import sys
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from easy_handeye2 import GET_SAMPLE_LIST_TOPIC
from easy_handeye2_msgs.srv import TakeSample
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from hand_eye_calibration.config import CalibrationType, load_manual_config
from hand_eye_calibration.solver import (
    CalibrationSample,
    TransformMatrix,
    finalize_calibration,
    rotation_delta_deg,
)
from hand_eye_calibration.sampling_runtime import SamplingRuntime


TARGET_SAMPLES = 20
MINIMUM_SAMPLES = 15
ASSISTANT_NAMESPACE = "/manual_calibration_assistant/"


@dataclass(frozen=True)
class Guidance:
    index: int
    category: str
    instruction: str
    axis: tuple[float, float, float]
    angle_deg: float
    orientation_label: str
    translation_hint: str


_GUIDANCE_SPECS = (
    (1, "ROOT", "保持当前安全位姿和清晰视野，点击 Take Sample 记录 root。", (0.0, 0.0, 1.0), 0.0, "ROOT", "标定码居中、清晰、距离适中。"),
    (2, "+Z 滚转", "相对 root 的{frame}局部 Z 轴正向滚转约 20°，再用视觉反馈居中标定码。", (0.0, 0.0, 1.0), 20.0, "+Z roll", "无固定平移目标；按画面居中补偿。"),
    (3, "-Z 滚转", "相对 root 的{frame}局部 Z 轴负向滚转约 20°，再用视觉反馈居中标定码。", (0.0, 0.0, -1.0), 20.0, "-Z roll", "无固定平移目标；按画面居中补偿。"),
    (4, "+X 倾斜", "相对 root 的{frame}局部 X 轴正向倾斜约 15°，再用视觉反馈居中标定码。", (1.0, 0.0, 0.0), 15.0, "+X tilt", "无固定平移目标；按画面居中补偿。"),
    (5, "-X 倾斜", "相对 root 的{frame}局部 X 轴负向倾斜约 15°，再用视觉反馈居中标定码。", (-1.0, 0.0, 0.0), 15.0, "-X tilt", "无固定平移目标；按画面居中补偿。"),
    (6, "+Y 倾斜", "相对 root 的{frame}局部 Y 轴正向倾斜约 15°，再用视觉反馈居中标定码。", (0.0, 1.0, 0.0), 15.0, "+Y tilt", "无固定平移目标；按画面居中补偿。"),
    (7, "-Y 倾斜", "相对 root 的{frame}局部 Y 轴负向倾斜约 15°，再用视觉反馈居中标定码。", (0.0, -1.0, 0.0), 15.0, "-Y tilt", "无固定平移目标；按画面居中补偿。"),
    (8, "+X 倾斜", "相对 root 的{frame}局部 X 轴正向倾斜约 28°，再用视觉反馈居中标定码。", (1.0, 0.0, 0.0), 28.0, "+X tilt", "无固定平移目标；按画面居中补偿。"),
    (9, "-X 倾斜", "相对 root 的{frame}局部 X 轴负向倾斜约 28°，再用视觉反馈居中标定码。", (-1.0, 0.0, 0.0), 28.0, "-X tilt", "无固定平移目标；按画面居中补偿。"),
    (10, "+Y 倾斜", "相对 root 的{frame}局部 Y 轴正向倾斜约 28°，再用视觉反馈居中标定码。", (0.0, 1.0, 0.0), 28.0, "+Y tilt", "无固定平移目标；按画面居中补偿。"),
    (11, "-Y 倾斜", "相对 root 的{frame}局部 Y 轴负向倾斜约 28°，再用视觉反馈居中标定码。", (0.0, -1.0, 0.0), 28.0, "-Y tilt", "无固定平移目标；按画面居中补偿。"),
    (12, "XY 复合 +/+", "相对 root 的{frame}组合 +X/+Y 倾斜，总角约 25°，再用视觉反馈居中标定码。", (1.0, 1.0, 0.0), 25.0, "XY +/+", "无固定平移目标；按画面居中补偿。"),
    (13, "XY 复合 +/-", "相对 root 的{frame}组合 +X/-Y 倾斜，总角约 25°，再用视觉反馈居中标定码。", (1.0, -1.0, 0.0), 25.0, "XY +/-", "无固定平移目标；按画面居中补偿。"),
    (14, "XY 复合 -/+", "相对 root 的{frame}组合 -X/+Y 倾斜，总角约 25°，再用视觉反馈居中标定码。", (-1.0, 1.0, 0.0), 25.0, "XY -/+", "无固定平移目标；按画面居中补偿。"),
    (15, "XY 复合 -/-", "相对 root 的{frame}组合 -X/-Y 倾斜，总角约 25°，再用视觉反馈居中标定码。", (-1.0, -1.0, 0.0), 25.0, "XY -/-", "无固定平移目标；按画面居中补偿。"),
    (16, "横向 +X + 姿态变化", "相对 root 的{frame}局部 +X 横向平移 50-80 mm，并绕 +Y 倾斜约 18°；再按视觉反馈居中。", (0.0, 1.0, 0.0), 18.0, "+Y tilt", "建议 +X 横向平移 50-80 mm；不执行精确平移目标。"),
    (17, "横向 -X + 姿态变化", "相对 root 的{frame}局部 -X 横向平移 50-80 mm，并绕 -Y 倾斜约 18°；再按视觉反馈居中。", (0.0, -1.0, 0.0), 18.0, "-Y tilt", "建议 -X 横向平移 50-80 mm；不执行精确平移目标。"),
    (18, "横向 +Y + 姿态变化", "相对 root 的{frame}局部 +Y 横向平移 50-80 mm，并绕 +X 倾斜约 18°；再按视觉反馈居中。", (1.0, 0.0, 0.0), 18.0, "+X tilt", "建议 +Y 横向平移 50-80 mm；不执行精确平移目标。"),
    (19, "横向 -Y + 姿态变化", "相对 root 的{frame}局部 -Y 横向平移 50-80 mm，并绕 -X 倾斜约 18°；再按视觉反馈居中。", (-1.0, 0.0, 0.0), 18.0, "-X tilt", "建议 -Y 横向平移 50-80 mm；不执行精确平移目标。"),
    (20, "视觉深度变化 + 复合姿态", "依据相机 depth 使标定码相对 root 拉远或拉近 50-80 mm，并作 XY +/+ 总角约 20°；不要将{frame}局部 Z 当作相机深度。", (1.0, 1.0, 0.0), 20.0, "XY +/+", "按实时 depth 选择拉远或拉近；不执行固定 ±Z 平移。"),
)


def guidance_for(calibration_type: CalibrationType) -> tuple[Guidance, ...]:
    frame = "末端" if calibration_type is CalibrationType.EYE_IN_HAND else "标定板/腕部"
    return tuple(
        Guidance(index, category, instruction.format(frame=frame), axis, angle, label, hint)
        for index, category, instruction, axis, angle, label, hint in _GUIDANCE_SPECS
    )


GUIDANCE = guidance_for(CalibrationType.EYE_IN_HAND)


def target_relative_rotation(guide: Guidance) -> R:
    axis = np.asarray(guide.axis, dtype=float)
    axis /= float(np.linalg.norm(axis))
    return R.from_rotvec(axis * math.radians(guide.angle_deg))


def guidance_pose_metrics(guide: Guidance, root: R, current: R) -> tuple[str, str, float]:
    desired = root * target_relative_rotation(guide)
    actual_relative = root.inv() * current
    error = rotation_delta_deg(desired, current)
    if guide.angle_deg == 0.0:
        return "ROOT", f"总角 {math.degrees(actual_relative.magnitude()):.1f}°", error
    axis = np.asarray(guide.axis, dtype=float)
    axis /= float(np.linalg.norm(axis))
    if np.count_nonzero(axis) == 1:
        actual = math.degrees(float(np.dot(actual_relative.as_rotvec(), axis)))
        actual_text = f"{guide.orientation_label} {actual:+.1f}°"
    else:
        actual_text = f"总角 {math.degrees(actual_relative.magnitude()):.1f}°"
    return f"{guide.orientation_label} {guide.angle_deg:.0f}°", actual_text, error


def guidance_readiness(vision_ok: bool, pose_error_deg: float, vision_note: str) -> str:
    if vision_ok and pose_error_deg <= 5.0:
        return "READY ✓"
    return "VISION OK / POSE ADJUST" if vision_ok else f"VISION ADJUST: {vision_note}"


def coordinate_frame_markers(
    *, frame_id: str, stamp, namespace: str, start_id: int,
    pose: TransformMatrix, alpha: float,
) -> list[Marker]:
    markers = []
    origin = np.asarray(pose.translation, dtype=float)
    rotation = pose.rotation.as_matrix()
    for offset, (axis, color) in enumerate(((0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0)), (2, (0.0, 0.0, 1.0)))):
        endpoint = origin + rotation[:, axis] * 0.10
        marker = Marker()
        marker.header.frame_id = frame_id
        if stamp is not None:
            marker.header.stamp = stamp
        marker.ns, marker.id = namespace, start_id + offset
        marker.type, marker.action = Marker.ARROW, Marker.ADD
        marker.points = [
            Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
            Point(x=float(endpoint[0]), y=float(endpoint[1]), z=float(endpoint[2])),
        ]
        marker.scale.x, marker.scale.y, marker.scale.z = 0.010, 0.020, 0.028
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = *color, alpha
        markers.append(marker)
    return markers


class ManualSessionState:
    """Pure state machine shared by ROS callbacks and unit tests."""

    def __init__(self):
        self.records: list[CalibrationSample] = []
        self.pending_reason: Optional[str] = None
        self.validating = False
        self.saving = False
        self.saved = False
        self.stopped = False
        self.last_message = "等待第 1/20 组 root 样本"

    def begin_validation(self, easy_count: int) -> tuple[bool, str]:
        if self.stopped or self.saved:
            return False, "标定会话已结束"
        if self.pending_reason:
            return False, "最新样本未通过，请先点击 Remove Sample"
        if self.validating or self.saving:
            return False, "已有操作正在执行"
        if len(self.records) >= TARGET_SAMPLES:
            return False, "20 组样本已完成，请点击 Save"
        expected = len(self.records) + 1
        if easy_count != expected:
            message = f"Easy 样本数不同步：{easy_count}，期望 {expected}"
            if easy_count > len(self.records):
                self.pending_reason = message
                self.last_message = f"{message}；请删除 Easy 中的最新样本"
            return False, self.last_message if self.pending_reason else message
        self.validating = True
        self.last_message = f"正在严格验证第 {expected}/20 组的 10 帧观测"
        return True, self.last_message

    def accept(self, record: CalibrationSample) -> str:
        self.validating = False
        self.records.append(record)
        self.last_message = f"第 {len(self.records)}/20 组通过"
        return self.last_message

    def reject(self, reason: str) -> str:
        self.validating = False
        self.pending_reason = str(reason)
        self.last_message = f"最新样本不合格：{reason}；请点击 Remove Sample"
        return self.last_message

    def remove_after_easy(self, easy_count: int) -> tuple[bool, str]:
        if self.validating or self.saving:
            return False, "操作进行中，暂不能删除"
        if self.pending_reason is not None:
            if easy_count < len(self.records):
                return False, f"Easy 样本数不同步：{easy_count}，期望 {len(self.records)}"
            if easy_count > len(self.records):
                self.last_message = f"仍有 {easy_count - len(self.records)} 个未配对 Easy 样本，请继续删除最新样本"
                return True, self.last_message
            self.pending_reason = None
            self.last_message = "已清除不合格样本，可以重新采集"
            return True, self.last_message
        if not self.records:
            self.last_message = (
                f"仍有 {easy_count} 个启动前 Easy 样本，请继续删除最新样本"
                if easy_count
                else "Easy 样本列表已清空，可以记录 root"
            )
            return True, self.last_message
        if easy_count != len(self.records) - 1:
            return False, f"Easy 样本数不同步：{easy_count}，期望 {len(self.records) - 1}"
        removed = self.records.pop()
        self.saved = False
        self.last_message = (
            "root 已删除，会话已重置" if removed.waypoint_index == 1
            else f"已删除第 {removed.waypoint_index}/20 组"
        )
        return True, self.last_message

    def begin_save(self, easy_count: int) -> tuple[bool, str]:
        if self.stopped:
            return False, "会话已由 q/Ctrl-C 停止，不会保存"
        if self.pending_reason or self.validating or self.saving:
            return False, "存在未处理样本或进行中的操作"
        if self.saved:
            return False, "本会话已经保存"
        accepted = len(self.records)
        if easy_count != accepted:
            return False, f"Easy 样本数不同步：{easy_count}，期望 {accepted}"
        if accepted < MINIMUM_SAMPLES:
            return False, f"至少需要 {MINIMUM_SAMPLES} 组有效样本，当前 {accepted} 组"
        self.saving = True
        self.last_message = "正在执行 Park/Horaud 严格求解与质量验收"
        return True, self.last_message

    def finish_save(self, success: bool, message: str) -> None:
        self.saving = False
        self.saved = bool(success)
        self.last_message = str(message)

    def stop(self, reason: str) -> None:
        self.stopped = True
        self.validating = False
        self.saving = False
        self.last_message = f"会话停止且未保存：{reason}"

    def status(self) -> dict:
        accepted = len(self.records)
        busy = self.validating or self.saving
        return {
            "accepted": accepted,
            "target": TARGET_SAMPLES,
            "blocked": bool(self.pending_reason),
            "busy": busy,
            "saved": self.saved,
            "stopped": self.stopped,
            "can_take": (
                not (self.stopped or self.saved or self.pending_reason or busy)
                and accepted < TARGET_SAMPLES
            ),
            "can_save": (
                not (self.stopped or self.saved or self.pending_reason or busy)
                and accepted >= MINIMUM_SAMPLES
            ),
            "message": self.last_message,
            "next_index": min(accepted + 1, TARGET_SAMPLES),
        }


class ManualCalibrationAssistant(Node, SamplingRuntime):
    """Validate Easy GUI samples and own the strict result saved by Save."""

    def __init__(self):
        super().__init__("manual_calibration_assistant")
        (
            self.frames_config,
            self.motion_config,
            self.sampling_config,
        ) = load_manual_config(self)
        self._use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self.state = ManualSessionState()
        self._state_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._collection_active = threading.Event()
        self._collection_active.set()
        self._collector_output_stem = None

        self._callback_group = MutuallyExclusiveCallbackGroup()
        self._io_group = ReentrantCallbackGroup()
        self._service_group = MutuallyExclusiveCallbackGroup()
        self._motion_thread = None
        self._keyboard_stream = self._open_keyboard_stream()
        self._initialize_sampling_runtime()

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            CameraInfo, self.frames_config.camera_info_topic, self._on_camera_info,
            sensor_qos, callback_group=self._io_group,
        )
        self.create_subscription(
            Image, self.frames_config.image_topic, self._on_image, sensor_qos,
            callback_group=self._io_group,
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10,
            callback_group=self._io_group,
        )
        self._easy_samples = self.create_client(
            TakeSample, GET_SAMPLE_LIST_TOPIC, callback_group=self._io_group,
        )

        self.create_service(
            Trigger, ASSISTANT_NAMESPACE + "status", self._status_service,
            callback_group=self._io_group,
        )
        self.create_service(
            Trigger, ASSISTANT_NAMESPACE + "validate_latest", self._validate_service,
            callback_group=self._service_group,
        )
        self.create_service(
            Trigger, ASSISTANT_NAMESPACE + "remove_latest", self._remove_service,
            callback_group=self._service_group,
        )
        self.create_service(
            Trigger, ASSISTANT_NAMESPACE + "save", self._save_service,
            callback_group=self._service_group,
        )

        marker_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._guidance_publisher = self.create_publisher(
            MarkerArray, ASSISTANT_NAMESPACE + "guidance", marker_qos,
        )
        self.create_timer(0.5, self._publish_guidance, callback_group=self._io_group)
        if self._keyboard_stream is not None:
            self.create_timer(
                self.motion_config.keyboard_poll_period, self._poll_keyboard,
                callback_group=self._io_group,
            )
        self.get_logger().info(
            "Manual assistant configured: "
            f"calibration_type={self.frames_config.calibration_type.value} "
            f"ee_frame={self.frames_config.ee_frame} "
            f"move_group_ns={self.motion_config.move_group_ns_fairino} "
            f"use_sim_time={self._use_sim_time} "
            f"output={self.sampling_config.calibration_output_directory}"
        )
        self._print_workflow()

    def _should_stop(self) -> bool:
        return (
            self._stop_requested.is_set()
            or not rclpy.ok()
            or bool(self._abort is not None and self._abort.is_set())
        )

    def _easy_count(self, timeout: float = 2.0) -> int:
        if not self._easy_samples.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("Easy get_sample_list service is unavailable")
        future = self._easy_samples.call_async(TakeSample.Request())
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline and not self._should_stop():
            time.sleep(0.01)
        if not future.done() or future.result() is None:
            raise RuntimeError("Easy get_sample_list service timed out")
        return len(future.result().samples.samples)

    def _joint_snapshot_deg(self) -> tuple[float, ...]:
        with self._joint_lock:
            if not self._joint_history:
                raise RuntimeError("JointState does not contain all configured joints")
            positions = self._joint_history[-1][1]
        return tuple(math.degrees(value) for value in positions)

    def _is_diverse(self, pose: TransformMatrix) -> tuple[bool, str]:
        for sample in self.state.records:
            translation = float(np.linalg.norm(
                np.asarray(pose.translation) - np.asarray(sample.robot_pose.translation)
            ))
            rotation = rotation_delta_deg(pose.rotation, sample.robot_pose.rotation)
            if (
                translation < self.sampling_config.minimum_translation_delta_m
                and rotation < self.sampling_config.minimum_rotation_delta_deg
            ):
                return False, (
                    f"与第 {sample.waypoint_index} 组重复："
                    f"{translation * 1000.0:.1f}mm/{rotation:.2f}°"
                )
        return True, "样本位姿具有足够差异"

    def _status_service(self, _request, response):
        try:
            easy_count = self._easy_count(timeout=0.5)
        except Exception:
            easy_count = None
        with self._state_lock:
            payload = self.state.status()
            guide = guidance_for(self.frames_config.calibration_type)[payload["next_index"] - 1]
            payload["next_category"] = guide.category
            payload["next_instruction"] = (
                "20 组已完成，请点击 Save"
                if payload["accepted"] == TARGET_SAMPLES else guide.instruction
            )
            if not self.state.records and easy_count:
                payload.update(
                    blocked=True,
                    can_take=False,
                    can_save=False,
                    message=f"Easy 中已有 {easy_count} 个旧样本，请从最新项开始全部删除",
                )
        response.success = not payload["stopped"]
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    def _validate_service(self, _request, response):
        try:
            easy_count = self._easy_count()
            with self._state_lock:
                ok, message = self.state.begin_validation(easy_count)
            if not ok:
                response.success, response.message = False, message
                return response

            stationary, reason = self._wait_for_joint_stationary()
            if not stationary:
                raise RuntimeError(f"关节未稳定：{reason}")
            robot, tracking, reason = self._stable_sample()
            if robot is None:
                raise RuntimeError(f"视觉质量门未通过：{reason}")
            diverse, diversity_note = self._is_diverse(robot)
            if not diverse:
                raise RuntimeError(diversity_note)
            joints_deg = self._joint_snapshot_deg()
            with self._state_lock:
                index = len(self.state.records) + 1
            if self._easy_count() != index:
                raise RuntimeError("验证期间 Easy 样本列表发生变化")
            with self._state_lock:
                message = self.state.accept(CalibrationSample(index, joints_deg, robot, tracking))
            response.success = True
            response.message = (
                f"{message}；{reason}；{diversity_note}；"
                "按 h+Enter 返回 root 后继续下一组；q+Enter 停止且不保存。"
            )
            self.get_logger().info(response.message)
        except Exception as exc:
            with self._state_lock:
                message = (
                    self.state.reject(str(exc))
                    if not self.state.pending_reason else self.state.last_message
                )
            response.success, response.message = False, message
            self.get_logger().warn(message)
        self._publish_guidance()
        return response

    def _remove_service(self, _request, response):
        try:
            easy_count = self._easy_count()
            with self._state_lock:
                before = len(self.state.records)
                response.success, response.message = self.state.remove_after_easy(easy_count)
                reset = response.success and before == 1 and not self.state.records
                if reset:
                    self._collector_output_stem = None
            if reset:
                self.vision_gate.reset_window()
            if response.success:
                self.get_logger().info(response.message)
            else:
                self.get_logger().warn(response.message)
        except Exception as exc:
            response.success, response.message = False, str(exc)
        self._publish_guidance()
        return response

    def _save_service(self, _request, response):
        try:
            easy_count = self._easy_count()
            with self._state_lock:
                ok, message = self.state.begin_save(easy_count)
                records = tuple(self.state.records)
            if not ok:
                response.success, response.message = False, message
                return response
            success = finalize_calibration(self, records)
            path = str(self._collector_output_stem.with_suffix(".calib")) if success else ""
            if success:
                result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["transform"]
                translation, rotation = result["translation"], result["rotation"]
                message = (
                    f"严格标定已保存：{path}\n"
                    f"translation=({translation['x']:.6f}, {translation['y']:.6f}, "
                    f"{translation['z']:.6f}) m\n"
                    f"quaternion=({rotation['x']:.6f}, {rotation['y']:.6f}, "
                    f"{rotation['z']:.6f}, {rotation['w']:.6f})"
                )
            else:
                message = "严格质量门未通过，未保存 .calib"
            with self._state_lock:
                self.state.finish_save(success, message)
            response.success, response.message = success, message
        except Exception as exc:
            with self._state_lock:
                self.state.finish_save(False, f"严格保存失败：{exc}")
                response.message = self.state.last_message
            response.success = False
        return response

    def _print_workflow(self) -> None:
        rows = [
            "辅助标定：共 20 组；在 RViz/MoveIt 手动移动，静止并保持标记清晰后点击 Take Sample。",
            "不合格样本会保留并锁定 Take/Save；点击 Remove Sample 后重试。h+Enter 返回 root，q+Enter 停止且不保存。",
        ]
        rows.extend(
            f"{guide.index:02d}. {guide.category}: {guide.instruction}"
            for guide in guidance_for(self.frames_config.calibration_type)
        )
        self.get_logger().info("\n".join(rows))

    @staticmethod
    def _open_keyboard_stream():
        if sys.stdin.isatty():
            return sys.stdin
        try:
            return open("/dev/tty", "r", encoding="utf-8")
        except OSError:
            return None

    def _poll_keyboard(self) -> None:
        try:
            ready, _, _ = select.select([self._keyboard_stream], [], [], 0.0)
        except (OSError, TypeError, ValueError):
            return
        if not ready:
            return
        command = self._keyboard_stream.readline().strip().lower()
        if command in ("h", "home", "return"):
            self._return_root()
        elif command in ("q", "quit", "exit"):
            self.stop_without_save("keyboard q")
        elif command:
            self.get_logger().warn("使用 h+Enter 返回 root，q+Enter 停止且不保存")

    def _return_root(self) -> None:
        with self._state_lock:
            if not self.state.records:
                self.get_logger().warn("尚未记录 root 样本")
                return
            if self.state.validating or self.state.saving:
                self.get_logger().warn("当前操作完成后才能返回 root")
                return
            root_joints = tuple(
                math.radians(value) for value in self.state.records[0].target_joints_deg
            )
        if self._motion_thread is not None and self._motion_thread.is_alive():
            self.get_logger().warn("返回 root 已在执行")
            return

        def move():
            try:
                if self._motion is None:
                    self._setup_motion()
                self._motion.move_to_joints(
                    root_joints,
                    action_name="Manual calibration: return root",
                    max_velocity=self.motion_config.max_velocity,
                    max_acceleration=self.motion_config.max_acceleration,
                    allowed_planning_time=self.motion_config.allowed_planning_time,
                    allowed_start_tolerance=self.motion_config.allowed_start_tolerance,
                )
            except Exception as exc:
                self.get_logger().error(f"返回 root 失败：{exc}")

        self._motion_thread = threading.Thread(target=move, daemon=True)
        self._motion_thread.start()

    def stop_without_save(self, reason: str) -> None:
        self._stop_requested.set()
        with self._state_lock:
            self.state.stop(reason)
        if self._abort is not None:
            self._abort.request_abort(reason)
            self._abort.cancel_all_motion_now()
        self.get_logger().info(self.state.last_message)

    def _root_base_T_ee(self) -> Optional[TransformMatrix]:
        with self._state_lock:
            if not self.state.records:
                return None
            robot = self.state.records[0].robot_pose
        if self.frames_config.calibration_type is CalibrationType.EYE_IN_HAND:
            return robot
        matrix = np.linalg.inv(robot.matrix())
        return TransformMatrix(
            R.from_matrix(matrix[:3, :3]),
            tuple(float(value) for value in matrix[:3, 3]),
        )

    def _publish_guidance(self) -> None:
        with self._state_lock:
            status = self.state.status()
        guide = guidance_for(self.frames_config.calibration_type)[status["next_index"] - 1]
        root = self._root_base_T_ee()
        try:
            current = self._latest_base_to_ee()
        except Exception:
            current = None
        display = root or current or TransformMatrix(R.identity(), (0.0, 0.0, 0.6))
        target = TransformMatrix(
            display.rotation * target_relative_rotation(guide), display.translation,
        )
        observation, vision_ok, vision_note = self.vision_gate.latest_observation()
        camera = self.vision_gate.camera_info_snapshot()
        if observation is None or not camera.ready:
            vision_text = f"MARKER: {vision_note}"
        else:
            centre_error = math.hypot(
                observation.center_px[0] - camera.width / 2.0,
                observation.center_px[1] - camera.height / 2.0,
            )
            vision_text = (
                f"MARKER: center {centre_error:.0f} px | depth {observation.tvec[2]:.2f} m"
                f" | margin {observation.margin_px:.0f} px"
            )

        if root is not None and current is not None:
            target_text, actual_text, error_deg = guidance_pose_metrics(
                guide, root.rotation, current.rotation,
            )
            readiness = guidance_readiness(vision_ok, error_deg, vision_note)
            pose_text = (
                f"TARGET: {target_text}\nACTUAL: {actual_text}\nERROR: {error_deg:.1f}°"
            )
        else:
            readiness = "RECORD ROOT" if root is None else "WAITING FOR EE TF"
            pose_text = "TARGET: root-relative orientation\nACTUAL: waiting for root/current EE"

        stamp = self.get_clock().now().to_msg()

        clear = Marker()
        clear.action = Marker.DELETEALL
        text = Marker()
        text.header.frame_id, text.header.stamp = self.frames_config.base_frame, stamp
        text.ns, text.id = "manual_calibration_text", 20
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position = Point(
            x=float(display.translation[0]), y=float(display.translation[1]),
            z=float(display.translation[2] + 0.16),
        )
        text.pose.orientation.w = 1.0
        text.scale.z = 0.035
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 0.9, 0.1, 1.0
        text.text = (
            "20/20 COMPLETE\n请点击 Save 执行严格验收"
            if status["accepted"] == TARGET_SAMPLES
            else (
                f"{guide.index:02d}/20 {guide.category}\n{guide.instruction}\n"
                f"{pose_text}\n{vision_text}\n{readiness}"
            )
        )
        markers = [clear]
        if root is not None:
            markers.extend(coordinate_frame_markers(
                frame_id=self.frames_config.base_frame, stamp=stamp, namespace="manual_root",
                start_id=1, pose=root, alpha=1.0,
            ))
            markers.extend(coordinate_frame_markers(
                frame_id=self.frames_config.base_frame, stamp=stamp, namespace="manual_target",
                start_id=4, pose=target, alpha=0.35,
            ))
        if current is not None:
            markers.extend(coordinate_frame_markers(
                frame_id=self.frames_config.base_frame, stamp=stamp, namespace="manual_current",
                start_id=7, pose=current, alpha=0.95,
            ))
        markers.append(text)
        self._guidance_publisher.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = ManualCalibrationAssistant()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.stop_without_save("Ctrl-C")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
