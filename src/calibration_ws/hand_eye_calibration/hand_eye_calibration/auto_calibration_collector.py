#!/usr/bin/env python3
"""Four-layer Fairino fixed-joint eye-in-hand collector.

该节点是手眼标定采集的核心 ROS 2 节点，负责：
  1. 订阅相机图像、相机内参和关节状态；
  2. 通过独立线程进行 ArUco 检测与视觉质量门控（VisionQualityGate）；
  3. 按配置文件中的预置关节位姿（20个槽位）依次运动；
  4. 在每个位姿下采样稳定的视觉观测和机器人末端位姿（base_T_ee）；
  5. 判断样本多样性，保存或丢弃样本；
  6. 采集完成后调用 solver 进行标定求解并存储结果。
"""

from __future__ import annotations

import math
from pathlib import Path
import select
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger

# 路径处理，确保可以导入同包内的模块
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "hand_eye_calibration"

# 导入配置、求解器和视觉模块
from hand_eye_calibration.config import JOINT_WAYPOINT_SLOTS, JointWaypointSpec, load_collector_config, yaml_use_sim_time
from hand_eye_calibration.solver import (
    CalibrationSample,
    TransformMatrix,
    coverage_status,
    finalize_calibration,
    rotation_delta_deg,
    robot_pose_for_calibration,
    save_samples,
)
from hand_eye_calibration.sampling_runtime import SamplingRuntime


class AutoCalibrationCollector(Node, SamplingRuntime):
    """ROS 2 节点：固定关节动作、WVCSC 视觉门控、本地求解与保存。"""

    def __init__(self):
        super().__init__("auto_calibration_collector")
        self._yaml_use_sim_time = yaml_use_sim_time()
        result = self.set_parameters([
            Parameter("use_sim_time", Parameter.Type.BOOL, self._yaml_use_sim_time),
        ])[0]
        if not result.successful:
            raise RuntimeError(f"Cannot apply YAML use_sim_time: {result.reason}")
        # 从 YAML 加载配置：frames_config, motion_config, sampling_config
        self.frames_config, self.motion_config, self.sampling_config = load_collector_config(self)
        self._use_sim_time = bool(self.get_parameter("use_sim_time").value)

        # 互斥回调组，确保关键路径（如运动执行）不被并发打断
        self._callback_group = MutuallyExclusiveCallbackGroup()

        # 事件标志（线程间同步）
        self._collection_active = threading.Event()   # 采集会话正在进行
        self._start_requested = threading.Event()     # 请求开始采集
        self._step_continue = threading.Event()       # 单步执行模式下的“继续”信号
        self._return_initial_requested = threading.Event()  # 请求返回初始位姿
        self._stop_requested = threading.Event()      # 请求停止（中止）
        self._quit_requested = threading.Event()      # 请求退出节点

        # 采集结果
        self._accepted: list[CalibrationSample] = []   # 已接受的样本
        self._results = []                             # 每个位姿的执行结果 (index, accepted, reason)
        self._collector_output_stem = None             # 存储路径前缀

        # 会话状态机状态
        self.session_state = "STANDBY"

        # 键盘轮询定时器与服务工作状态
        self._keyboard_timer = None
        self._keyboard_clock = None
        self._keyboard_stream = sys.stdin
        self._keyboard_enabled = False
        self._service_subs_ready = False

        self._initialize_sampling_runtime()

        # 设置手动控制（服务、键盘监听）
        self._setup_manual_control()

        # 打印配置信息
        self._log_configuration()

    def _log_configuration(self):
        """打印当前加载的配置概要。"""
        self.get_logger().info(
            "Fairino collector configured: "
            f"image={self.frames_config.image_topic} camera_info={self.frames_config.camera_info_topic} "
            f"intrinsics={self.frames_config.camera_intrinsics_source} "
            f"real_waypoints={len(self.sampling_config.waypoint_specs)}/20 min_samples=15 "
            f"use_sim_time={self._use_sim_time} "
            f"move_group_ns={self.motion_config.move_group_ns_fairino or '/'} "
            f"keyboard_control={self._keyboard_enabled}"
        )
        command = (
            "ros2 run hand_eye_calibration auto_calibration_collector.py --ros-args -p use_sim_time:=true"
            if self._use_sim_time else
            "ros2 run hand_eye_calibration auto_calibration_collector.py --ros-args -p use_sim_time:=false"
        )
        environment = "Gazebo simulation" if self._use_sim_time else "real hardware"
        self.get_logger().info(
            f"Time source from YAML: use_sim_time={self._use_sim_time}; Environment: {environment}. "
            "注意：仿真运行前将 auto_calibration_collector_params.yaml 的 use_sim_time 设为 true；"
            "实机运行前设为 false。"
        )
        self.get_logger().info(
            "YAML controls direct collector startup; do not mix it with --ros-args -p use_sim_time. "
            f"Equivalent ROS parameter syntax: {command}"
        )

    def _setup_manual_control(self):
        """设置手动控制入口：服务 + 键盘输入。"""
        # 创建 /auto_calibration_collector/start 服务（Trigger 类型）
        self.create_service(Trigger, "/auto_calibration_collector/start", self._on_start_request)

        # 如果配置了 auto_start=true，则自动触发开始信号
        auto_start = bool(self.declare_parameter("auto_start", False).value)
        if auto_start:
            self._start_requested.set()

        # ros2 launch commonly gives child processes a non-TTY stdin even when
        # the launch command itself runs in a terminal.  Prefer the controlling
        # terminal so Enter/s is read from the same terminal as the launch shell.
        self._keyboard_stream = sys.stdin
        try:
            self._keyboard_enabled = bool(self._keyboard_stream.isatty())
        except (AttributeError, OSError, ValueError):
            self._keyboard_enabled = False
        if not self._keyboard_enabled:
            try:
                terminal = open("/dev/tty", "r", buffering=1)
                self._keyboard_stream = terminal
                self._keyboard_enabled = True
            except OSError:
                try:
                    self._keyboard_stream.fileno()
                    self._keyboard_enabled = True
                except (AttributeError, OSError, ValueError):
                    self._keyboard_enabled = False
        if self._keyboard_enabled:
            # Keyboard control must continue working when simulated ROS time is paused.
            self._keyboard_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self._keyboard_timer = self.create_timer(
                self.motion_config.keyboard_poll_period,
                self.poll_keyboard_once,
                clock=self._keyboard_clock,
            )
        elif self.motion_config.step_between_actions:
            raise RuntimeError("step_between_actions=true requires an interactive TTY")

    def _on_start_request(self, _request, response):
        """服务回调：处理 /start 请求。"""
        if self._collection_active.is_set() or self._start_requested.is_set():
            response.success, response.message = False, "collection already queued or running"
        else:
            self._start_requested.set()
            response.success, response.message = True, "collection start accepted"
        return response

    def poll_keyboard_once(self):
        """定时器回调：非阻塞读取键盘输入（适用于交互式终端）。"""
        if not self._keyboard_enabled:
            return
        try:
            ready, _, _ = select.select([self._keyboard_stream], [], [], 0.0)
        except (OSError, ValueError):
            self._keyboard_enabled = False
            return
        if not ready:
            return
        line = self._keyboard_stream.readline()
        if line == "":
            self._keyboard_enabled = False
            return
        command = line.strip().lower()
        if command in ("", "s", "start"):
            # 如果采集活跃，则触发单步继续；否则触发开始采集
            (self._step_continue if self._collection_active.is_set() else self._start_requested).set()
        elif command in ("h", "return"):
            if self.session_state == "RETURN_INITIAL":
                self._return_initial_requested.set()
            else:
                self.get_logger().warn("h is only valid in RETURN_INITIAL")
        elif command in ("q", "quit", "exit"):
            self._request_stop("keyboard command")
        else:
            self.get_logger().warn("use Enter/s to start or step, h to return initial, q to stop")

    def _request_stop(self, reason: str):
        """请求停止当前采集会话（触发中止）。"""
        self.get_logger().info(f"Collection stop requested: {reason}")
        self._stop_requested.set()
        if self._abort is not None:
            self._abort.request_abort(reason)
            try:
                self._abort.cancel_all_motion_now()
            except Exception as exc:
                self.get_logger().warn(f"motion cancellation failed: {exc}")

    def _clear_stop(self):
        """清除停止标志。"""
        self._stop_requested.clear()
        if self._abort is not None:
            self._abort.clear()

    def _should_exit(self) -> bool:
        """检查是否应该退出整个节点。"""
        return not rclpy.ok() or self._quit_requested.is_set()

    def _should_stop(self) -> bool:
        """检查是否应该停止当前会话（包括外部退出请求）。"""
        return self._should_exit() or self._stop_requested.is_set() or bool(self._abort is not None and self._abort.is_set())

    def _wait_for_start(self) -> bool:
        """等待用户或服务触发开始信号。"""
        self.get_logger().info("Standby. Press Enter/s or call /auto_calibration_collector/start.")
        while not self._should_exit():
            if self._start_requested.wait(self.motion_config.start_wait_poll_period):
                self._start_requested.clear()
                return True
        return False

    def _wait_for_step(self, message: str) -> bool:
        """单步模式：等待用户按 Enter 继续。"""
        if not self.motion_config.step_between_actions:
            return True
        self._step_continue.clear()
        self.get_logger().info(message)
        while not self._should_stop():
            if self._step_continue.wait(self.motion_config.start_wait_poll_period):
                self._step_continue.clear()
                return True
        return False

    def _setup_services(self):
        """订阅相机信息、图像、关节状态等 ROS 话题。"""
        if self._service_subs_ready:
            return
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, self.frames_config.camera_info_topic, self._on_camera_info, sensor_qos)
        self.create_subscription(Image, self.frames_config.image_topic, self._on_image, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self._service_subs_ready = True

    def _is_diverse(self, pose: TransformMatrix):
        """检查当前机器人的末端位姿是否与已接受样本足够不同（多样性门控）。"""
        for sample in self._accepted:
            translation = float(np.linalg.norm(np.asarray(pose.translation) - np.asarray(sample.robot_pose.translation)))
            rotation = rotation_delta_deg(pose.rotation, sample.robot_pose.rotation)
            if translation < self.sampling_config.minimum_translation_delta_m and rotation < self.sampling_config.minimum_rotation_delta_deg:
                return False, f"nearest waypoint={sample.waypoint_index} translation={translation:.4f}m rotation={rotation:.2f}deg"
        return True, "diverse" if self._accepted else "first sample"

    def _record_result(self, spec: JointWaypointSpec, accepted: bool, reason: str):
        """记录单个位点的执行结果（用于日志和统计）。"""
        self._results.append((spec.index, accepted, reason))
        # 注意：info 与 warn 必须各自使用独立的调用点（不同行号），否则 rclpy
        # 会在同一 CallerId 上检测到 severity 变化并抛出
        # "Logger severity cannot be changed between calls."，导致会话中止。
        if accepted:
            self.get_logger().info(f"WAYPOINT {spec.index:02d} ACCEPTED: {reason}")
        else:
            self.get_logger().warn(f"WAYPOINT {spec.index:02d} REJECTED: {reason}")

    def _execute_waypoint(self, spec: JointWaypointSpec, total: int):
        """执行单个关节位姿：
           1. 运动到目标关节；
           2. 等待关节静止；
           3. 采集稳定视觉样本；
           4. 多样性检测；
           5. 接受或拒绝并记录。
        """
        self.get_logger().info(f"WAYPOINT {spec.index:02d}/{total:02d} target={spec.joints_deg}")
        self._clear_joint_history()
        try:
            moved = self._motion.move_to_joints(
                spec.joints_rad, action_name=f"Calibration waypoint {spec.index:02d}",
                max_velocity=self.motion_config.max_velocity, max_acceleration=self.motion_config.max_acceleration,
                allowed_planning_time=self.motion_config.allowed_planning_time,
                allowed_start_tolerance=self.motion_config.allowed_start_tolerance,
            )
        except Exception as exc:
            self._record_result(spec, False, f"MOVE_FAILED: {exc}")
            return
        if not moved:
            self._record_result(spec, False, "MOVE_FAILED")
            return
        time.sleep(self.motion_config.settle_time_sec)
        stationary, reason = self._wait_for_joint_stationary()
        if not stationary:
            self._record_result(spec, False, f"JOINT_STATIONARY: {reason}")
            return
        robot, tracking, reason = self._stable_sample()
        if robot is None:
            self._record_result(spec, False, f"VISION_GATE: {reason}")
            return
        diverse, note = self._is_diverse(robot)
        if not diverse:
            self._record_result(spec, False, f"DIVERSITY_GATE: {note}")
            return
        self._accepted.append(CalibrationSample(spec.index, spec.joints_deg, robot, tracking))
        self._record_result(spec, True, f"{reason}; {note}")

    def _wait_for_camera_info(self) -> bool:
        deadline = time.monotonic() + self.sampling_config.stable_marker_timeout_sec
        while not self.vision_gate.camera_info_snapshot().ready and time.monotonic() < deadline:
            time.sleep(self.motion_config.start_wait_poll_period)
        return self.vision_gate.camera_info_snapshot().ready

    def _precheck(self) -> bool:
        """采集前检查：ArUco 就绪、MoveIt 就绪、相机内参就绪、关节状态存在、位姿不越限。"""
        if not self._cv_ready:
            self.get_logger().error("PRECHECK: OpenCV ArUco detector is unavailable")
            return False
        if not self._wait_for_moveit():
            return False
        if not self._wait_for_execution_controller():
            return False
        if not self._wait_for_camera_info():
            self.get_logger().error("PRECHECK: CameraInfo projection matrix is not ready")
            return False
        if not self._wait_for_joint_state_stream():
            return False
        for spec in self.sampling_config.waypoint_specs:
            for axis, (value, limits) in enumerate(zip(spec.joints_deg, self.sampling_config.joint_limits_deg), 1):
                if not limits[0] <= value <= limits[1]:
                    self.get_logger().error(f"PRECHECK: W{spec.index} joint {axis} outside configured limits")
                    return False
        return True

    def _wait_return_initial(self):
        """等待用户按 'h' 回到第一个位姿（用于结束会话后复位）。"""
        self.session_state = "RETURN_INITIAL"
        self._return_initial_requested.clear()
        if not self._keyboard_enabled:
            return
        self.get_logger().info("RETURN_INITIAL: press h+Enter to return to waypoint 1; q+Enter stays here.")
        while not self._should_stop() and not self._should_exit():
            if self._return_initial_requested.wait(0.05):
                root = self.sampling_config.waypoint_specs[0]
                try:
                    self._motion.move_to_joints(
                        root.joints_rad, action_name="Return initial pose", max_velocity=self.motion_config.max_velocity,
                        max_acceleration=self.motion_config.max_acceleration,
                        allowed_planning_time=self.motion_config.allowed_planning_time,
                        allowed_start_tolerance=self.motion_config.allowed_start_tolerance,
                    )
                except Exception as exc:
                    self.get_logger().error(f"RETURN_INITIAL failed: {exc}")
                return

    def _run_session(self):
        """执行一次完整的采集会话：
           1. 复位状态；
           2. 预检查；
           3. 依次执行每个关节位姿；
           4. 检查覆盖度（coverage）；
           5. 若满足条件则调用 solver 求解并保存。
        """
        self._accepted = []
        self._results = []
        self._collector_output_stem = None
        self.vision_gate.reset_window()
        self._clear_stop()
        self.session_state = "PRECHECK"
        if not self._precheck():
            return False, False
        self.session_state = "JOINT_SEQUENCE"
        waypoints = self.sampling_config.waypoint_specs
        for spec in waypoints:
            if self._should_stop() or not self._wait_for_step(f"[step] W{spec.index:02d}/{JOINT_WAYPOINT_SLOTS:02d}: press Enter to execute"):
                break
            # 单个位姿的任何未预期异常都只记为 REJECTED 并继续下一个位姿，
            # 不允许因单个位姿失败而中止整个标定会话。
            try:
                self._execute_waypoint(spec, JOINT_WAYPOINT_SLOTS)
            except Exception as exc:
                self.get_logger().error(f"WAYPOINT {spec.index:02d} unexpected error, skipping: {exc}")
                self._record_result(spec, False, f"SKIPPED: {exc}")
        self.get_logger().info(f"JOINT SEQUENCE complete: attempted={len(self._results)} accepted={len(self._accepted)}")
        if self._should_stop():
            return False, False
        complete, coverage_note = coverage_status(
            self._accepted, self.sampling_config, minimum_count=self.sampling_config.minimum_samples,
        )
        self.get_logger().info(f"SAMPLE COVERAGE: {coverage_note}")
        if not complete:
            save_samples(self, self._accepted, "incomplete")
            self._wait_return_initial()
            return False, True
        self.session_state = "SOLVE"
        saved = finalize_calibration(self, self._accepted)
        self._wait_return_initial()
        return saved, True

    def _validate_time_base(self) -> bool:
        """检查 use_sim_time 与是否存在 /clock 话题的一致性。"""
        has_clock = any(name == "/clock" for name, _ in self.get_topic_names_and_types())
        if has_clock and not self._use_sim_time:
            self.get_logger().error(
                "Time-source mismatch: Gazebo /clock is present but YAML use_sim_time is false. "
                "Set use_sim_time: true in auto_calibration_collector_params.yaml and restart collector. "
                "If using a ROS parameter instead, the syntax is --ros-args -p use_sim_time:=true (not a remap)."
            )
            return False
        if not has_clock and self._use_sim_time:
            self.get_logger().error(
                "Time-source mismatch: YAML use_sim_time is true but no /clock publisher exists. "
                "Set use_sim_time: false in auto_calibration_collector_params.yaml for real hardware and restart collector. "
                "If using a ROS parameter instead, the syntax is --ros-args -p use_sim_time:=false (not a remap)."
            )
            return False
        return True

    def run(self):
        """节点主循环：等待开始信号，执行采集，循环。"""
        if not self._validate_time_base():
            return
        while not self._should_exit():
            self.session_state = "WAIT_START"
            if not self._wait_for_start():
                return
            self._collection_active.set()
            persisted = False
            try:
                _saved, persisted = self._run_session()
            except Exception as exc:
                self.get_logger().error(f"Collection session aborted: {exc}")
            finally:
                if not persisted:
                    try:
                        save_samples(self, self._accepted, "stopped" if self._should_stop() else "failed")
                    except Exception as exc:
                        self.get_logger().error(f"Cannot save samples: {exc}")
                self._collection_active.clear()
            self.session_state = "STANDBY"


def main():
    """ROS 2 节点入口函数。"""
    print(f"[auto_calibration_collector runtime] file={__file__}", flush=True)
    rclpy.init()
    node = AutoCalibrationCollector()
    try:
        node._setup_services()
        node._setup_motion()
    except Exception as exc:
        node.get_logger().error(f"Setup failed: {exc}")
        rclpy.shutdown()
        raise SystemExit(1)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    started = False
    clock = Clock(clock_type=ClockType.STEADY_TIME)

    def start():
        """延时启动 run()，确保 executor 已经 spin。"""
        nonlocal started
        if started:
            return
        started = True
        try:
            node.run()
        finally:
            node._quit_requested.set()
            if rclpy.ok():
                rclpy.shutdown()

    # 使用定时器触发启动（避免在 executor.spin() 之前直接调用 run 导致阻塞）
    timer = node.create_timer(0.5, start, callback_group=MutuallyExclusiveCallbackGroup(), clock=clock)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node._request_stop("KeyboardInterrupt")
    finally:
        timer.cancel()
        node._quit_requested.set()
        executor.shutdown()


if __name__ == "__main__":
    main()
