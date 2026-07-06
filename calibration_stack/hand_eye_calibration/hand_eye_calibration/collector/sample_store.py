"""SampleManager: sample store, diversity checks, candidate generation, subset selection.

本模块负责管理已采集的样本集：
- 存储和访问已接受的样本记录
- 根据基本偏移配置生成候选采集规范（CandidateSpec）
- 进行多样性检查，防止重复或过于接近的样本
- 评估样本子集的质量，并为标定求解器选择最优子集
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .sample_types import (
    FAMILY_EXECUTION_ORDER,
    AcceptedSampleQuality,
    AcceptedSampleRecord,
    BaseOffsetPose,
    CandidateFamily,
    CandidateSpec,
)
from .sample_governor import SampleSetGovernor


class SampleManager:
    """
    样本管理器。
    封装了样本数据的存储、候选位姿生成、多样性门控、子集优化等功能，
    并与 SampleSetGovernor 协作进行覆盖度和可观测性评估。
    """

    def __init__(
        self,
        *,
        base_offsets: Dict[str, List[BaseOffsetPose]],
        governor: SampleSetGovernor,
        nominal_translation_delta_scale: float,
        nominal_rotation_delta_scale: float,
        rotation_delta_deg: Callable,
    ):
        # 基本偏移配置字典，键为族名（如 "sphere_anchor"），值为偏移位姿列表
        self._base_offsets = base_offsets
        # 样本治理器实例，用于覆盖度/可观测性判断
        self.governor = governor
        # 标称平移增量缩放因子（用于多样性判定）
        self.nominal_translation_delta_scale = float(nominal_translation_delta_scale)
        # 标称旋转增量缩放因子
        self.nominal_rotation_delta_scale = float(nominal_rotation_delta_scale)
        # 旋转角度差计算函数
        self._rotation_delta_deg = rotation_delta_deg

        # 已接受的样本记录列表
        self._accepted_samples: List[AcceptedSampleRecord] = []
        # 参考旋转（通常为第一个样本的末端姿态，用于计算相对姿态）
        self._reference_rotation: Optional[R] = None

    # ------------------------------------------------------------------
    # 已接受样本的访问接口
    # ------------------------------------------------------------------

    @property
    def accepted_samples(self) -> List[AcceptedSampleRecord]:
        """返回所有已接受样本的完整记录列表。"""
        return self._accepted_samples

    @property
    def accepted_sample_poses(self):
        """返回所有已接受样本的机器人末端变换（TransformMatrix）列表。"""
        return [r.robot_pose for r in self._accepted_samples]

    @property
    def accepted_tracking_poses(self):
        """返回所有已接受样本的跟踪标记变换列表，排除为 None 的记录。"""
        return [r.tracking_pose for r in self._accepted_samples if r.tracking_pose is not None]

    @property
    def reference_rotation(self):
        """返回当前的参考旋转，用于可观测性计算。"""
        return self._reference_rotation

    def reset(self):
        """清空所有样本及参考旋转，恢复到初始状态。"""
        self._accepted_samples.clear()
        self._reference_rotation = None

    def set_reference_rotation(self, rotation: R):
        """手动设置参考旋转（例如原位姿）。"""
        self._reference_rotation = rotation

    def record_accepted_sample(
        self,
        *,
        robot_pose,
        tracking_pose,
        family: str,
        spec: CandidateSpec,
        quality: AcceptedSampleQuality,
        candidate_idx: int,
        candidate_description: str,
        recenter_attempted: bool,
        recenter_strict_converged: bool,
    ):
        """
        添加一个通过质量门控的样本到内部列表。
        如果尚未设置参考旋转，则将当前样本的末端姿态设为参考旋转。
        """
        if self._reference_rotation is None:
            self._reference_rotation = robot_pose.rotation
        self._accepted_samples.append(
            AcceptedSampleRecord(
                robot_pose=robot_pose,
                tracking_pose=tracking_pose,
                family=family,
                spec=spec,
                quality=quality,
                candidate_idx=int(candidate_idx),
                candidate_description=candidate_description,
                recenter_attempted=bool(recenter_attempted),
                recenter_strict_converged=bool(recenter_strict_converged),
                removable=bool(spec.removable),
            )
        )

    def remove_accepted_sample(self, index: int):
        """按索引移除一个已接受的样本。"""
        self._accepted_samples.pop(index)

    def subset_records(self, keep_indices: Sequence[int]) -> List[AcceptedSampleRecord]:
        """
        根据保留的索引集合返回对应的样本记录子集。
        用于子集优化后的样本选取。
        """
        keep = set(int(idx) for idx in keep_indices)
        return [r for idx, r in enumerate(self._accepted_samples) if idx in keep]

    # ------------------------------------------------------------------
    # 候选规范生成（基于族顺序，生成时去重）
    # ------------------------------------------------------------------

    @staticmethod
    def _make_spec(offset: BaseOffsetPose) -> CandidateSpec:
        """将基础偏移配置对象转换为 CandidateSpec 候选规范。"""
        return CandidateSpec(
            source=offset.label,
            base_x=offset.base_x,
            base_y=offset.base_y,
            base_z=offset.base_z,
            pitch=offset.pitch,
            yaw=offset.yaw,
            roll=offset.roll,
            family=offset.family,
            removable=offset.removable,
            intent=offset.intent,
            observability_axis=offset.observability_axis,
            dedup_protected=offset.dedup_protected,
        )

    def build_candidate_specs(self) -> List[CandidateSpec]:
        """
        按族执行顺序生成候选规范列表，并进行去重处理。

        - 对于有 dedup_protected=True 的定向候选（纯方向样本），仅通过完全键值去重，
          避免因平移/旋转接近而被提前丢弃。
        - 对于覆盖度候选，额外进行平移和旋转的接近度去重，防止生成在运行时会被
          actual_too_close 拒绝的候选。
        """
        exact_seen = set()  # 记录已见过的精确键值，用于完全去重
        specs: List[CandidateSpec] = []

        for family_name in FAMILY_EXECUTION_ORDER:
            offsets = self._base_offsets.get(family_name, [])
            for offset in offsets:
                spec = self._make_spec(offset)
                exact_k = spec.exact_key()  # 计算唯一标识键

                # 完全去重：相同键值的候选只保留第一个
                if exact_k in exact_seen:
                    continue
                exact_seen.add(exact_k)

                # 接近度去重：仅对非保护（即覆盖类）候选进行
                if not spec.dedup_protected:
                    too_close = False
                    for prev_spec in specs:
                        prev_t = np.array([prev_spec.base_x, prev_spec.base_y, prev_spec.base_z])
                        this_t = np.array([spec.base_x, spec.base_y, spec.base_z])
                        dt = float(np.linalg.norm(this_t - prev_t))
                        # 计算姿态欧拉角分量的最大差值作为旋转差距
                        dr = max(
                            abs(spec.pitch - prev_spec.pitch),
                            abs(spec.yaw - prev_spec.yaw),
                            abs(spec.roll - prev_spec.roll),
                        )
                        # 若平移和旋转均小于治理器规定的最小增量，视为太近
                        if dt < self.governor.sample_min_translation_delta and dr < self.governor.sample_min_rotation_delta_deg:
                            too_close = True
                            break
                    if too_close:
                        continue

                specs.append(spec)

        return specs

    # ------------------------------------------------------------------
    # 候选特征判断辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _has_orientation_component(spec: CandidateSpec) -> bool:
        """判断候选规范是否包含非零的姿态分量（pitch/yaw/roll）。"""
        return any(abs(v) > 1.0e-6 for v in (spec.pitch, spec.yaw, spec.roll))

    @staticmethod
    def _has_translation_component(spec: CandidateSpec) -> bool:
        """判断候选规范是否包含非零的平移分量（base_x/y/z）。"""
        return any(abs(v) > 1.0e-6 for v in (spec.base_x, spec.base_y, spec.base_z))

    @staticmethod
    def _is_pure_orientation(spec: CandidateSpec) -> bool:
        """
        判断是否为纯方向候选：有姿态分量但无平移分量。
        这类候选通常用于 roll 覆盖等场景，位置保持在原位。
        """
        return (
            SampleManager._has_orientation_component(spec)
            and not SampleManager._has_translation_component(spec)
        )

    @staticmethod
    def _is_yaw_coupled_shell_record(record: AcceptedSampleRecord) -> bool:
        """
        判断一个已接受样本是否属于 "yaw 耦合外壳" 记录：
        族为 SPHERE_SHELL，具有显著的 yaw 分量且同时包含平移分量。
        """
        return (
            record.family == CandidateFamily.SPHERE_SHELL
            and abs(record.spec.yaw) > 1.0e-6
            and SampleManager._has_translation_component(record.spec)
        )

    @staticmethod
    def _optional_quality_remove_key(record: AcceptedSampleRecord):
        """
        为可移除样本生成一个排序键，值越小越倾向于被移除。
        逻辑：
        - yaw 耦合外壳样本优先级最高（0），其次是最近重新居中但未严格收敛的（0），
          然后再按质量指标排序（取负值使高质量值排在前面，即移除时优先丢弃低质量样本）。
        """
        yaw_coupled_shell = SampleManager._is_yaw_coupled_shell_record(record)
        non_strict_recenter = (
            record.recenter_attempted and not record.recenter_strict_converged
        )
        quality = record.quality
        return (
            0 if yaw_coupled_shell else 1,  # 优先考虑移除 yaw 耦合外壳样本
            0 if non_strict_recenter else 1,  # 其次考虑未严格收敛的重新居中样本
            -quality.camera_model_error_px,   # 以下各项取负，使数值越大越不被移除
            -quality.center_error_px,
            -quality.center_std_px,
            -quality.depth_std_m,
            -quality.angle_std_deg,
            quality.marker_side_px,
            quality.margin_px,
        )

    def is_coupled_shell_record(self, record: AcceptedSampleRecord) -> bool:
        """
        检查样本是否为耦合外壳记录：同时包含平移和姿态分量的 SPHERE_SHELL 样本。
        """
        return (
            record.family == CandidateFamily.SPHERE_SHELL
            and self._has_translation_component(record.spec)
            and self._has_orientation_component(record.spec)
        )

    def is_yaw_coupled_shell_record(self, record: AcceptedSampleRecord) -> bool:
        """委托到静态方法，判断是否为 yaw 耦合外壳记录。"""
        return self._is_yaw_coupled_shell_record(record)

    # ------------------------------------------------------------------
    # 多样性检查（避免样本过于集中）
    # ------------------------------------------------------------------

    def _diversity_status(
        self,
        sample_pose,
        *,
        translation_threshold: float,
        rotation_threshold_deg: float,
        prefix: str = "",
    ) -> Tuple[bool, str]:
        """
        核心多样性检查：将候选位姿与所有已接受样本比较，
        若平移和旋转均小于各自阈值，则判定为太近，返回 False。
        否则返回 True 并附带描述。
        """
        if not self._accepted_samples:
            return True, f"{prefix}first sample".strip()
        c_t = np.array(sample_pose.translation, dtype=float)
        for prev in self.accepted_sample_poses:
            p_t = np.array(prev.translation, dtype=float)
            trans_delta = float(np.linalg.norm(c_t - p_t))
            rot_delta_deg = self._rotation_delta_deg(prev.rotation, sample_pose.rotation)
            if trans_delta < translation_threshold and rot_delta_deg < rotation_threshold_deg:
                return False, (
                    f"{prefix}too close to accepted sample "
                    f"(dt={trans_delta:.3f}m, dr={rot_delta_deg:.1f}deg)"
                )
        return True, f"{prefix}diverse".strip()

    def is_diverse_transform(self, sample_pose) -> Tuple[bool, str]:
        """严格多样性检查：使用治理器的最小平移/旋转增量阈值。"""
        return self._diversity_status(
            sample_pose,
            translation_threshold=self.governor.sample_min_translation_delta,
            rotation_threshold_deg=self.governor.sample_min_rotation_delta_deg,
        )

    def nominal_diversity_status(self, sample_pose) -> Tuple[bool, str]:
        """
        标称多样性检查：阈值按标称缩放因子放大，用于初步筛选。
        如果标称检查不通过，则返回 'nominal_too_close' 状态。
        """
        nominal_trans = max(1e-6, self.governor.sample_min_translation_delta * self.nominal_translation_delta_scale)
        nominal_rot = max(1e-6, self.governor.sample_min_rotation_delta_deg * self.nominal_rotation_delta_scale)
        ok, note = self._diversity_status(sample_pose, translation_threshold=nominal_trans, rotation_threshold_deg=nominal_rot)
        if ok:
            return True, note
        return False, f"nominal_too_close: {note}"

    def nominal_orientation_diversity_status(self, sample_pose, observability_axis: str) -> Tuple[bool, str]:
        """
        标称方向多样性检查：仅与同 observability_axis 的已接受样本比较旋转差异，
        忽略平移，适用于纯方向候选。
        """
        if not self._accepted_samples:
            return True, f"nominal_orientation_diverse axis={observability_axis} (first sample)"
        nominal_rot = max(
            1.0e-6,
            self.governor.orientation_sample_min_rotation_delta_deg * self.nominal_rotation_delta_scale,
        )
        for prev_rec in self._accepted_samples:
            prev_axis = getattr(prev_rec.spec, "observability_axis", "none")
            if prev_axis != observability_axis:
                continue
            rot_delta_deg = self._rotation_delta_deg(prev_rec.robot_pose.rotation, sample_pose.rotation)
            if rot_delta_deg < nominal_rot:
                return False, (
                    f"nominal_orientation_too_close axis={observability_axis}: "
                    f"dr={rot_delta_deg:.1f}deg < {nominal_rot:.1f}deg"
                )
        return True, (
            f"nominal_orientation_diverse axis={observability_axis} "
            f"(dr_thresh={nominal_rot:.1f}deg)"
        )

    def nominal_diversity_for_spec(self, sample_pose, spec: CandidateSpec) -> Tuple[bool, str]:
        """
        根据候选规范类型选择适当的标称多样性检查：
        - 对于纯方向且受保护且指定了 observability_axis 的候选，
          仅执行方向多样性检查；否则执行常规标称多样性检查。
        """
        obs_axis = getattr(spec, "observability_axis", "none")
        if getattr(spec, "dedup_protected", False) and obs_axis != "none" and self._is_pure_orientation(spec):
            return self.nominal_orientation_diversity_status(sample_pose, obs_axis)
        return self.nominal_diversity_status(sample_pose)

    def is_orientation_diverse_transform(self, sample_pose, observability_axis: str) -> Tuple[bool, str]:
        """
        严格方向多样性检查（用于受保护的方向候选）。
        仅比较与同 axis 的已接受样本的旋转增量，忽略平移。
        """
        if not self._accepted_samples:
            return True, f"orientation_diverse axis={observability_axis} (first sample)"
        orient_rot = self.governor.orientation_sample_min_rotation_delta_deg
        for prev_rec in self._accepted_samples:
            prev_axis = getattr(prev_rec.spec, "observability_axis", "none")
            if prev_axis != observability_axis:
                continue
            rot_delta_deg = self._rotation_delta_deg(
                prev_rec.robot_pose.rotation, sample_pose.rotation)
            if rot_delta_deg < orient_rot:
                return False, (
                    f"orientation_too_close axis={observability_axis}: "
                    f"dr={rot_delta_deg:.1f}deg < {orient_rot:.1f}deg"
                )
        return True, f"orientation_diverse axis={observability_axis} (dr_thresh={orient_rot:.1f}deg)"

    # ------------------------------------------------------------------
    # 子集质量评估与构建
    # ------------------------------------------------------------------

    def subset_quality_metrics(
        self,
        records: Sequence[AcceptedSampleRecord],
    ) -> Optional[dict]:
        """
        计算给定样本子集的质量度量字典，用于子集优选。
        包括高度符号不平衡、yaw 耦合外壳样本数、重新居中未收敛数，
        以及各种质量指标的最大/最小值。
        """
        if not records:
            return None
        # 统计正负高度样本数
        height_pos = sum(
            1 for r in records
            if r.family == CandidateFamily.SPHERE_HEIGHT and r.spec.base_z > 1.0e-6
        )
        height_neg = sum(
            1 for r in records
            if r.family == CandidateFamily.SPHERE_HEIGHT and r.spec.base_z < -1.0e-6
        )
        # 若存在高度样本但缺失某一方向，设置一个巨大的不平衡值，避免该子集被优选
        if (height_pos + height_neg) > 0 and (height_pos == 0 or height_neg == 0):
            height_imbalance = 999
        else:
            height_imbalance = abs(height_pos - height_neg)
        return {
            "height_positive_count": height_pos,
            "height_negative_count": height_neg,
            "height_sign_imbalance": height_imbalance,
            "yaw_coupled_shell_count": sum(
                1 for record in records
                if self._is_yaw_coupled_shell_record(record)
            ),
            "non_strict_recenter_count": sum(
                1
                for record in records
                if record.recenter_attempted and not record.recenter_strict_converged
            ),
            "max_camera_model_error_px": max(
                record.quality.camera_model_error_px for record in records
            ),
            "max_center_error_px": max(record.quality.center_error_px for record in records),
            "max_center_std_px": max(record.quality.center_std_px for record in records),
            "max_depth_std_m": max(record.quality.depth_std_m for record in records),
            "max_angle_std_deg": max(record.quality.angle_std_deg for record in records),
            "min_marker_side_px": min(record.quality.marker_side_px for record in records),
            "min_margin_px": min(record.quality.margin_px for record in records),
        }

    @staticmethod
    def _append_unique_keep_set(keep_sets: List[Tuple[int, ...]], keep) -> None:
        """将排序后的索引元组加入列表，避免重复。"""
        keep_tuple = tuple(sorted(keep))
        if keep_tuple not in keep_sets:
            keep_sets.append(keep_tuple)

    def solver_subset_keep_sets(self, min_keep: int, max_keep: int) -> List[Tuple[int, ...]]:
        """
        为标定求解器生成一组推荐的样本索引子集（保留集）。

        策略：
        - 强制保留不可移除的样本（removable=False）。
        - 按多种优先级序列（考虑方向、高度、外壳覆盖等）逐步添加可选样本。
        - 基于质量评分反向剔除最差的可选样本，同时保证最小外壳样本数。
        - 最终返回一系列候选保留索引集（已去重）。
        """
        records = self._accepted_samples
        if not records:
            return []

        # 分离强制保留和可选样本
        mandatory = [idx for idx, rec in enumerate(records) if not rec.removable]
        optional = [idx for idx, rec in enumerate(records) if rec.removable]

        # 分类外壳样本
        coupled_shell = [
            idx for idx in optional
            if self.is_coupled_shell_record(records[idx])
        ]
        yaw_coupled_shell = [
            idx for idx in coupled_shell
            if self.is_yaw_coupled_shell_record(records[idx])
        ]
        yaw_coupled_set = set(yaw_coupled_shell)
        stable_coupled_shell = [
            idx for idx in coupled_shell
            if idx not in yaw_coupled_set
        ]
        essential = sorted(set(mandatory))
        min_keep = max(len(essential), int(min_keep))
        max_keep = min(len(records), max(int(max_keep), min_keep))

        if len(essential) > max_keep:
            return [tuple(essential)]

        # 辅助：按指定键值降序排序
        def _best_by(items, key_fn):
            return sorted(items, key=key_fn, reverse=True)

        # 辅助：获取候选规范的某个绝对分量
        def _abs_component(spec: CandidateSpec, axis: str) -> float:
            return abs(getattr(spec, axis))

        # 按方向/位置分量大小分组排序
        shell_x_pos = _best_by(
            [idx for idx in optional if records[idx].family == CandidateFamily.SPHERE_SHELL and records[idx].spec.base_x > 1.0e-6],
            lambda idx: _abs_component(records[idx].spec, "base_x"),
        )
        shell_x_neg = _best_by(
            [idx for idx in optional if records[idx].family == CandidateFamily.SPHERE_SHELL and records[idx].spec.base_x < -1.0e-6],
            lambda idx: _abs_component(records[idx].spec, "base_x"),
        )
        shell_y_pos = _best_by(
            [idx for idx in optional if records[idx].family == CandidateFamily.SPHERE_SHELL and records[idx].spec.base_y > 1.0e-6],
            lambda idx: _abs_component(records[idx].spec, "base_y"),
        )
        shell_y_neg = _best_by(
            [idx for idx in optional if records[idx].family == CandidateFamily.SPHERE_SHELL and records[idx].spec.base_y < -1.0e-6],
            lambda idx: _abs_component(records[idx].spec, "base_y"),
        )
        shell_z = _best_by(
            [idx for idx in optional if records[idx].family == CandidateFamily.SPHERE_SHELL and abs(records[idx].spec.base_z) > 1.0e-6],
            lambda idx: _abs_component(records[idx].spec, "base_z"),
        )
        roll_cov = _best_by(
            [idx for idx in optional if records[idx].family == CandidateFamily.SPHERE_ROLL_COVERAGE],
            lambda idx: abs(records[idx].spec.roll),
        )

        # 多种优先级添加顺序，以覆盖不同偏好
        priority_sequences = [
            stable_coupled_shell + shell_x_pos[:1] + shell_x_neg[:1] + shell_y_pos[:1] + shell_y_neg[:1] + shell_z[:1] + roll_cov,
            stable_coupled_shell + shell_z[:2] + shell_x_pos[:1] + shell_x_neg[:1] + shell_y_pos[:1] + shell_y_neg[:1] + roll_cov,
            stable_coupled_shell + roll_cov + shell_x_pos[:1] + shell_x_neg[:1] + shell_y_pos[:1] + shell_y_neg[:1] + shell_z[:1],
            stable_coupled_shell + shell_x_pos + shell_x_neg + shell_y_pos + shell_y_neg + shell_z + roll_cov,
        ]

        keep_sets: List[Tuple[int, ...]] = []
        for sequence in priority_sequences:
            keep = list(essential)
            for idx in sequence:
                if idx in keep:
                    continue
                if len(keep) >= max_keep:
                    break
                keep.append(idx)
                # 每达到 min_keep 就记录一次子集
                if len(keep) >= min_keep:
                    self._append_unique_keep_set(keep_sets, keep)
            self._append_unique_keep_set(keep_sets, keep)

        # 基于质量反向剔除：从可选样本中逐步移除质量最差的，同时保持最低外壳样本数
        quality_optional = [
            idx for idx in optional
            if idx not in essential
        ]
        worst_optional = sorted(
            quality_optional,
            key=lambda idx: self._optional_quality_remove_key(records[idx]),
        )
        quality_keep = list(range(len(records)))
        for idx in worst_optional:
            if idx not in quality_keep:
                continue
            trial_keep = [keep_idx for keep_idx in quality_keep if keep_idx != idx]
            if len(trial_keep) < min_keep:
                break
            shell_count = sum(
                1 for keep_idx in trial_keep
                if records[keep_idx].family == CandidateFamily.SPHERE_SHELL
            )
            if shell_count < self.governor.min_sphere_shell_samples:
                continue
            quality_keep = trial_keep
            if len(quality_keep) <= max_keep:
                self._append_unique_keep_set(keep_sets, quality_keep)

        # 最后添加全量索引集作为备选
        self._append_unique_keep_set(keep_sets, range(len(records)))
        return keep_sets

    # ------------------------------------------------------------------
    # 委托给治理器的便捷方法
    # ------------------------------------------------------------------

    def coverage_metrics(self):
        """调用治理器计算当前样本集的覆盖度量。"""
        return self.governor.coverage_metrics(self._accepted_samples)

    def coverage_status(self) -> Tuple[bool, str]:
        """判断当前样本集的覆盖度是否满足要求。"""
        return self.governor.coverage_status(self._accepted_samples)

    def observability_metrics(self):
        """调用治理器计算当前样本集的可观测性度量。"""
        return self.governor.observability_metrics(self._accepted_samples, self._reference_rotation)

    def observability_status(self) -> Tuple[bool, str]:
        """判断当前样本集的可观测性是否满足要求。"""
        return self.governor.observability_status(self._accepted_samples, self._reference_rotation)

    def dual_gate_status(self) -> Tuple[bool, str, dict, dict]:
        """同时获取覆盖度和可观测性的综合状态。"""
        return self.governor.dual_gate_status(self._accepted_samples, self._reference_rotation)

    def gate_deficits(self) -> dict:
        """返回当前样本集的指标赤字字典（哪些维度尚未达标）。"""
        return self.governor.gate_deficits(self._accepted_samples, self._reference_rotation)