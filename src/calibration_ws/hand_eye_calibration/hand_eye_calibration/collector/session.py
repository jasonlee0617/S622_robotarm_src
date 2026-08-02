"""CollectorExecutionSession: main collection orchestration facade.

本模块是手眼标定自动采集流程的总编排类。
它将各个阶段的逻辑委托给 session_checks、session_motion、
session_finalize 中的模块级辅助函数，而将所有状态集中在这个类中管理。

主要职责：
- 初始化并持有所有子模块（配置、几何、运动、视觉门控、样本管理等）
- 提供 TF 查询、图像标记状态检查等便捷方法
- 执行完整的采集会话：移动到原位 → 生成候选 → 逐个执行 → 本地标定收尾
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from .vision import QUALITY_SAMPLING

# 导入三个辅助模块，并将它们的函数绑定为类的静态方法/实例方法
from . import session_checks as _checks
from . import session_motion as _motion
from . import session_finalize as _finalize


class CollectorExecutionSession:
    """
    采集执行会话类。
    通过将外部辅助函数绑定为类属性，使得在会话中可以直接调用它们，
    同时又保持了代码的模块化分离。
    """

    # 将会话检查模块的函数直接映射为类属性，便于在实例方法中通过 self._xxx 调用
    _post_move_recenter_requirement = _checks.post_move_recenter_requirement
    _is_xy_coverage_candidate = staticmethod(_checks.is_xy_coverage_candidate)
    _recenter_weak_allowance = _motion.recenter_weak_allowance
    _recenter_budget_for_family = _motion.recenter_budget_for_family
    _projection_metrics = _checks.projection_metrics
    _check_projected_marker = _checks.check_projected_marker
    _marker_status = _checks.marker_status
    _camera_model_metrics = _checks.camera_model_metrics
    _check_marker_visible = _checks.check_marker_visible
    _wait_for_stable_marker = _checks.wait_for_stable_marker
    _capture_direct_sample = _checks.capture_direct_sample
    _candidate_quality_snapshot = _checks.candidate_quality_snapshot
    _precision_sample_status = _checks.precision_sample_status
    _wait_for_moveit = _motion.wait_for_moveit
    _moveit_ready_status = _motion.moveit_ready_status
    _workspace_status = _motion.workspace_status
    _preplan_pose = _motion.preplan_pose
    _original_place_pose = _motion.original_place_pose
    _go_original_place = _motion.go_original_place
    _recover_last_good_pose = _motion.recover_last_good_pose
    _fresh_successful_observation_after_motion = _motion.fresh_successful_observation_after_motion
    _move_with_visibility_guard = _motion.move_with_visibility_guard
    _recenter_marker = _motion.recenter_marker
    _actual_pose_diverse = _motion.actual_pose_diverse
    _move_candidate_and_sample = _motion.move_candidate_and_sample
    _precision_recenter_budget = _motion.precision_recenter_budget
    _maybe_precision_recenter = _motion.maybe_precision_recenter
    _stable_center_limit = staticmethod(_motion.stable_center_limit)
    _is_gate_deficit_critical = staticmethod(_checks.is_gate_deficit_critical)
    _finalize_calibration = _finalize.finalize_calibration

    def __init__(
        self,
        *,
        node,
        frames_config,
        motion_config,
        sampling_config,
        geometry,
        tf_buffer,
        motion,
        vision_gate,
        sample_manager,
        calibration_validator,
    ):
        """
        初始化采集会话。
        接收所有必要的子模块引用和配置对象。

        参数说明：
        - node: ROS 2 节点 (AutoCalibrationCollector)
        - frames_config: 坐标系配置
        - motion_config: 运动相关配置
        - sampling_config: 采样/视觉相关配置
        - geometry: CollectorGeometry 实例
        - tf_buffer: tf2_ros.Buffer 实例
        - motion: MoveItMotion 实例
        - vision_gate: VisionQualityGate 实例
        - sample_manager: SampleManager 实例
        - calibration_validator: CalibrationValidator 实例
        """
        self.node = node
        self.frames = frames_config
        self.motion_cfg = motion_config
        self.sampling_cfg = sampling_config
        self.geometry = geometry
        self.tf_buffer = tf_buffer
        self.motion = motion
        self.vision_gate = vision_gate
        self.sample_manager = sample_manager
        self.calibration_validator = calibration_validator
        # The image-centering Jacobian is measured online in the robot base frame.
        self.centering_jacobian = None
        # 采集结果列表，每项为 (候选ID, 描述, 是否成功, 备注)
        self.results = []
        # 记录最后一个标记可见时的末端位姿，用于丢失后恢复
        self.last_good_pose = None

    def _reset_session_state(self):
        """重置会话状态：清空结果列表、上次良好位姿、样本管理器，并清除停止采集标志。"""
        self.results = []
        self.last_good_pose = None
        self.sample_manager.reset()
        self.node._clear_collection_stop()

    def _logger(self):
        """快捷获取 ROS 日志记录器。"""
        return self.node.get_logger()

    def _cv_ready(self) -> bool:
        """检查 OpenCV/图像检测是否已就绪。"""
        return bool(getattr(self.node, "_cv_ready", False))

    def _lookup_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 1.0):
        """
        查询两个坐标系之间的 TF 变换，返回 TransformMatrix。
        若超时或不可用则抛出异常。
        """
        return self.geometry.tf_to_matrix(
            self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=Duration(seconds=timeout_sec),
            )
        )

    def _lookup_tf_at_ns(self, target_frame: str, source_frame: str, stamp_ns: int, timeout_sec: float = 1.0):
        """Look up a robot transform at the PnP image stamp; no latest-TF substitution."""
        stamp = Time(nanoseconds=int(stamp_ns))
        return self.geometry.tf_to_matrix(
            self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, timeout=Duration(seconds=timeout_sec),
            )
        )

    def _current_transform(self, target_frame: str, source_frame: str):
        """
        尝试获取当前 TF 变换，失败时返回 None 并记录警告。
        """
        try:
            return self._lookup_tf(target_frame, source_frame, timeout_sec=1.0)
        except Exception as exc:
            self._logger().warn(f"Cannot lookup {target_frame}->{source_frame}: {exc}")
            return None

    def _image_marker_status(self, require_center: bool = False, quality_level: str = QUALITY_SAMPLING,
                             center_error_limit_px: Optional[float] = None):
        """
        封装对视觉门控的图像标记状态检查。
        如果图像检测器未就绪，返回失败。
        """
        if not self._cv_ready():
            return False, "image-level ArUco detector is unavailable"
        return self.vision_gate.image_marker_status(
            require_center=require_center, quality_level=quality_level,
            center_error_limit_px=center_error_limit_px,
        )

    def _capture_base_pose(self) -> bool:
        """
        通过 TF 捕获并记录当前基座到末端的位姿，用于初始化候选生成。
        返回 True 表示成功获取并打印了位姿日志。
        """
        try:
            t = self.tf_buffer.lookup_transform(
                self.frames.base_frame, self.frames.ee_frame, Time(), timeout=Duration(seconds=2.0),
            )
            p = t.transform.translation
            q = t.transform.rotation
            euler = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=True)
            self._logger().info(
                f"Captured base pose {self.frames.base_frame}->{self.frames.ee_frame}: "
                f"xyz=({float(p.x):.4f}, {float(p.y):.4f}, {float(p.z):.4f}), "
                f"rpy=({float(euler[0]):.1f}, {float(euler[1]):.1f}, {float(euler[2]):.1f}) deg"
            )
            return True
        except Exception as exc:
            self._logger().error(f"Cannot lookup {self.frames.base_frame}->{self.frames.ee_frame}: {exc}")
            return False

    def _record_candidate_failure(self, candidate, note: str, *, recover: bool = False) -> None:
        """
        将候选采集失败记录到结果列表，并在需要时触发恢复到上次良好位姿。
        """
        self.results.append((candidate.idx, candidate.description, False, note))
        if recover:
            self._recover_last_good_pose()

    # ------------------------------------------------------------------
    # 进度摘要日志
    # ------------------------------------------------------------------

    def _log_coverage_summary(self):
        """打印当前样本集的覆盖度摘要（样本数、XYZ 跨度、最大旋转差等）。"""
        m = self.sample_manager.coverage_metrics()
        if m is None:
            self._logger().warn("Coverage summary: no accepted samples.")
            return
        self._logger().info(
            f"Coverage summary: samples={m['count']}, "
            f"xyz_span=({m['xyz_span'][0]:.3f},{m['xyz_span'][1]:.3f},{m['xyz_span'][2]:.3f})m, "
            f"xy_span={m['xy_span']:.3f}m, z_span={m['z_span']:.3f}m, "
            f"max_rot_delta={m['max_rot_delta_deg']:.1f}deg"
        )

    def _log_observability_summary(self):
        """打印当前样本集的可观测性摘要（各欧拉角跨度、锚点/高度/外壳样本数等）。"""
        m = self.sample_manager.observability_metrics()
        if m is None:
            self._logger().warn("Observability summary: no accepted samples.")
            return
        self._logger().info(
            f"Observability summary: pitch_span={m['pitch_span_deg']:.1f}deg, "
            f"yaw_span={m['yaw_span_deg']:.1f}deg, roll_span={m['roll_span_deg']:.1f}deg, "
            f"sphere_anchor={m['sphere_anchor_count']}, sphere_height={m['sphere_height_count']}, "
            f"sphere_shell={m['sphere_shell_count']}"
        )

    # ------------------------------------------------------------------
    # 采集目标达成检查（双重门控）
    # ------------------------------------------------------------------

    def _collection_goal_reached(self) -> Tuple[bool, str]:
        """
        判断是否已达到采集目标。
        需要同时满足：
        1. 成功样本数 >= min_successful_samples
        2. 覆盖度和可观测性双重门控均通过
        """
        count = len(self.sample_manager.accepted_sample_poses)
        if count < self.sampling_cfg.min_successful_samples:
            return False, f"count {count}/{self.sampling_cfg.min_successful_samples} below minimum"

        ok, note, _, _ = self.sample_manager.dual_gate_status()
        if not ok:
            return False, note
        return True, f"collection goal satisfied: {note}"

    # ------------------------------------------------------------------
    # 主要采集会话流程
    # ------------------------------------------------------------------

    def _run_collection_session(self):
        """
        执行一次完整的自动采集会话。

        流程概览：
        1. 重置本地状态
        2. 捕获原位姿并检查视觉就绪
        3. 初始标记检查与采样质量门控
        4. 等待标记稳定 + 相机模型自检
        5. 记录参考旋转、设置 last_good_pose
        6. 生成候选位姿列表并打印概要
        7. 遍历候选，依次执行 move_candidate_and_sample，直到满足条件或超出上限
        8. 输出结果统计、诊断信息、覆盖度/可观测性状态
        9. 调用 finalize_calibration 进行子集选择、求解、保存
        """
        self._reset_session_state()
        # 捕获当前 base->ee 位姿
        if not self._capture_base_pose():
            return
        # 确保图像级 ArUco 检测已就绪
        if not self._cv_ready():
            self._logger().error("Image-level ArUco quality gate is not available.")
            return

        # 检查原位姿下标记是否可见
        marker_ok, marker_note = self._check_marker_visible(timeout=self.sampling_cfg.marker_timeout)
        if not marker_ok:
            self._logger().warn(f"Initial marker check failed: {marker_note}.")
            return
        self._logger().info(f"Initial marker check ok: {marker_note}")

        # 检查原位姿下采样质量是否达标
        sampling_ok, sampling_note = self._image_marker_status(require_center=True, quality_level=QUALITY_SAMPLING)
        if not sampling_ok:
            obs = self.vision_gate.latest_successful_observation()
            info = self.vision_gate.camera_info_snapshot()
            if obs is not None and info.ready:
                du = obs.center_px[0] - info.cx
                dv = obs.center_px[1] - info.cy
                self._logger().error(
                    f"Original place does not satisfy sampling quality. "
                    f"marker_center=({obs.center_px[0]:.1f},{obs.center_px[1]:.1f}) "
                    f"image_center=({info.cx:.0f},{info.cy:.0f}) "
                    f"center_error=({du:.1f},{dv:.1f})px; {sampling_note}"
                )
                recentered, recenter_note, _, _ = self._recenter_marker(
                    max_total_translation=self.motion_cfg.recenter_max_total_translation_m,
                    center_error_limit_px=self.sampling_cfg.precision_max_center_error_px,
                )
                if not recentered:
                    self._logger().error(f"Seed-free initial recenter failed: {recenter_note}")
                    return
                sampling_ok, sampling_note = self._image_marker_status(
                    require_center=True, quality_level=QUALITY_SAMPLING,
                )
                if not sampling_ok:
                    self._logger().error(f"Initial sampling quality remains invalid after recenter: {sampling_note}")
                    return
            else:
                self._logger().error(f"Original place does not satisfy sampling quality. {sampling_note}")
            return
        self._logger().info(f"Initial sampling-quality gate passed: {sampling_note}")

        # 等待标记稳定
        stable_ok, stable_note = self._wait_for_stable_marker()
        if not stable_ok:
            self._logger().error(f"Initial marker is not stable enough: {stable_note}")
            return
        # 相机模型一致性检查
        model_ok, model_note, _ = self._camera_model_metrics()
        if not model_ok:
            self._logger().error(f"Initial camera model self-check failed: {model_note}")
            return
        self._logger().info(f"Initial {model_note}")

        # 记录参考旋转（用于可观测性计算）
        initial_base_T_ee = self._current_transform(self.frames.base_frame, self.frames.ee_frame)
        if initial_base_T_ee is not None:
            self.last_good_pose = self.geometry.matrix_to_pose_stamped(
                initial_base_T_ee, self.frames.base_frame, self.node.get_clock().now().to_msg(),
            )
            self.sample_manager.set_reference_rotation(initial_base_T_ee.rotation)

        # 绝对最大采样数上限
        abs_max = getattr(self.sampling_cfg, "absolute_max_successful_samples", 24)
        self._logger().info(
            "Starting base-offset collection: target "
            f"{self.sampling_cfg.min_successful_samples} good samples, "
            f"soft cap {self.sampling_cfg.max_successful_samples}, "
            f"absolute cap {abs_max}, "
            "spherical-shell deterministic sweep."
        )
        if initial_base_T_ee is None:
            self._logger().error("Cannot capture actual original_place EE pose for candidate generation.")
            return

        # 生成所有候选规范并构建候选位姿列表
        all_specs = self.sample_manager.build_candidate_specs()
        try:
            all_candidates = self.geometry.build_visibility_candidates(
                reference_base_T_ee=initial_base_T_ee,
                candidate_specs=all_specs,
                workspace_status=self._workspace_status,
                now_msg=lambda: self.node.get_clock().now().to_msg(),
            )
        except RuntimeError as exc:
            self._logger().error(str(exc))
            return
        if not all_candidates:
            self._logger().error("No fixed-offset calibration candidates generated.")
            return

        # 统计各家族候选数量
        family_counts = {}
        for c in all_candidates:
            family_counts[c.family] = family_counts.get(c.family, 0) + 1
        self._logger().info(
            f"Candidate sweep: {len(all_candidates)} total — "
            + ", ".join(f"{fam}={cnt}" for fam, cnt in sorted(family_counts.items()))
        )

        # 构建 label -> family 的映射，用于赤字检查
        spec_family_map = _checks.build_spec_family_map(self.sampling_cfg.base_offsets)

        # 主循环：遍历所有候选
        for order_idx, candidate in enumerate(all_candidates, start=1):
            if self.node._should_stop():
                break

            # 达到软上限时，若双重门控已通过则停止
            if len(self.sample_manager.accepted_sample_poses) >= self.sampling_cfg.max_successful_samples:
                cov_ok, _ = self.sample_manager.coverage_status()
                obs_ok, _ = self.sample_manager.observability_status()
                if cov_ok and obs_ok:
                    self._logger().info(
                        f"Stopping candidate sweep: reached soft cap "
                        f"{self.sampling_cfg.max_successful_samples} and dual gate PASS"
                    )
                    break
                # 否则，检查当前候选是否对解决赤字至关重要，若不是且已达绝对上限，则停止
                source = spec_family_map.get(candidate.spec.source, "")
                deficits = self.sample_manager.gate_deficits()
                if self._is_gate_deficit_critical(candidate, source, deficits):
                    active = [k for k, v in deficits.items() if v]
                    self._logger().info(
                        f"[{order_idx:02d}/{len(all_candidates):02d}] soft-cap override: "
                        f"deficits={active} candidate={candidate.idx:02d} src={source}"
                    )
                elif len(self.sample_manager.accepted_sample_poses) >= getattr(
                    self.sampling_cfg, "absolute_max_successful_samples", 24
                ):
                    self._logger().warn(
                        "Stopping: absolute_max_successful_samples reached "
                        f"with active deficits={[k for k, v in deficits.items() if v]}"
                    )
                    break
                else:
                    continue

            # 在进入候选执行前再次检查目标是否已达成
            candidate_source = spec_family_map.get(candidate.spec.source, "unknown")
            goal_reached, goal_note = self._collection_goal_reached()
            if goal_reached:
                self._logger().info(f"Stopping candidate sweep early: {goal_note}")
                break
            if len(self.sample_manager.accepted_sample_poses) >= self.sampling_cfg.min_successful_samples:
                self._logger().info(f"Continue sweep for coverage/observability: {goal_note}")

            self._logger().info(
                f"[{order_idx:02d}/{len(all_candidates):02d}] candidate {candidate.idx:02d} "
                f"family={candidate.family} src={candidate_source} {candidate.description}"
            )
            # 执行单个候选的运动和采样
            self._move_candidate_and_sample(candidate, self.sampling_cfg.min_successful_samples)

        # 如果用户手动停止了采集，则直接返回，不进行后续计算/保存
        if self.node._stop_collection_requested.is_set():
            self._logger().warn("Collection session interrupted; skip compute/save and return to standby.")
            return

        # 汇总结果
        ok_count = sum(1 for _, _, ok, _ in self.results if ok)
        self._logger().info("=" * 60)
        self._logger().info(f"Collection complete: {ok_count}/{self.sampling_cfg.min_successful_samples} required samples succeeded")
        for idx, desc, ok, note in self.results:
            status = "OK" if ok else "FAIL"
            self._logger().info(f"  [{idx:02d}] {status} {desc}: {note}")

        # 外壳样本诊断
        shell_ok = sum(1 for _, desc, ok, _ in self.results if ok and "sphere_shell" in desc)
        shell_fail = sum(1 for _, desc, ok, _ in self.results if not ok and "sphere_shell" in desc)
        shell_fail_reasons = {}
        for _, desc, ok, note in self.results:
            if not ok and "sphere_shell" in desc:
                reason = note.split(":")[0] if ":" in note else note[:60]
                shell_fail_reasons[reason] = shell_fail_reasons.get(reason, 0) + 1
        yaw_ok = sum(1 for _, desc, ok, _ in self.results if ok and "yaw" in desc.lower() and "sphere_anchor" in desc)
        yaw_fail = sum(1 for _, desc, ok, _ in self.results if not ok and "yaw" in desc.lower() and "sphere_anchor" in desc)
        self._logger().info(
            f"Shell diagnostics: sphere_shell OK={shell_ok} FAIL={shell_fail} "
            + (f"reasons={shell_fail_reasons}" if shell_fail_reasons else "")
        )
        self._logger().info(f"Yaw diagnostics: yaw OK={yaw_ok} FAIL={yaw_fail}")

        # 输出覆盖度/可观测性摘要
        self._log_coverage_summary()
        self._log_observability_summary()
        cov_ok, cov_note = self.sample_manager.coverage_status()
        obs_ok, obs_note = self.sample_manager.observability_status()
        self._logger().info(f"Coverage gate: {'PASS' if cov_ok else 'FAIL'}: {cov_note}")
        self._logger().info(f"Observability gate: {'PASS' if obs_ok else 'FAIL'}: {obs_note}")

        # 若旋转跨度不足，给出诊断信息和可能被拒的 roll 候选
        cov_m = self.sample_manager.coverage_metrics()
        if cov_m and cov_m["max_rot_delta_deg"] < self.sampling_cfg.min_coverage_rotation_span_deg:
            self._logger().warn(
                f"COVERAGE ROTATION DEFICIT: rot_span={cov_m['max_rot_delta_deg']:.1f}deg "
                f"< {self.sampling_cfg.min_coverage_rotation_span_deg:.1f}deg. "
                "sphere_roll_coverage candidates may have been rejected as too-close. "
                "Check orientation_sample_min_rotation_delta_deg and candidate angles."
            )
            roll_rejects = [(idx, desc, note) for idx, desc, ok, note in self.results
                           if not ok and ("sphere_roll_coverage" in desc or "orientation_too_close" in note)]
            if roll_rejects:
                self._logger().warn("Rejected roll/coverage candidates:")
                for idx, desc, note in roll_rejects:
                    self._logger().warn(f"  [{idx:02d}] {desc}: {note}")
        if ok_count < self.sampling_cfg.min_successful_samples:
            self._logger().warn("Not enough samples succeeded.")

        # 进入标定最终化流程
        self._finalize_calibration(ok_count)

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def run(self):
        """
        启动采集器的运行循环。
        1. 等待 MoveIt2 就绪
        2. 解析种子 ee_T_cam
        3. 进入主循环：等待启动指令 → 移动到原位 → 执行一次采集会话 → 循环
        """
        if not self._wait_for_moveit():
            return
        while not self.node._should_exit():
            self.node._clear_collection_stop()
            if not self.node._wait_for_start_request():
                return
            self.node._collection_active.set()
            try:
                if not self._go_original_place():
                    if self.node._should_exit():
                        return
                    self._logger().error("Original place failed. Returning to standby.")
                    continue
                self._run_collection_session()
            finally:
                self.node._collection_active.clear()
            if self.node._should_exit():
                return
            if self.node._stop_collection_requested.is_set():
                self._logger().info("Back to standby after manual stop request.")
            else:
                self._logger().info("Collection session finished. Returning to standby.")
