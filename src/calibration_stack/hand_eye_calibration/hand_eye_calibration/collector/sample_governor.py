"""SampleSetGovernor: unified coverage + observability + subset governance.

本模块提供统一的样本集治理器，负责评估已采集样本的空间覆盖度（coverage）
和姿态可观测性（observability），并判断是否满足标定所需的最低要求。
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .sample_types import (
    AcceptedSampleRecord,
    CandidateFamily,
)


class SampleSetGovernor:
    """
    样本集统一治理器。

    将原本分散的覆盖度检查、可观测性检查、高风险族跳过逻辑
    和子集优化入口整合到一个类中，拥有完整的门控逻辑和子集搜索编排能力。
    """

    def __init__(
        self,
        *,
        min_successful_samples: int,
        sample_min_translation_delta: float,
        sample_min_rotation_delta_deg: float,
        orientation_sample_min_rotation_delta_deg: float,
        min_coverage_xy_span_m: float,
        min_coverage_z_span_m: float,
        min_coverage_rotation_span_deg: float,
        min_pitch_span_deg: float,
        min_yaw_span_deg: float,
        min_roll_span_deg: float,
        min_sphere_anchor_samples: int,
        min_sphere_height_samples: int,
        min_sphere_shell_samples: int,
        rotation_delta_deg: Callable,
    ):
        # 最少成功样本数
        self.min_successful_samples = int(min_successful_samples)
        # 样本间最小平移增量（米）
        self.sample_min_translation_delta = float(sample_min_translation_delta)
        # 样本间最小旋转增量（度）
        self.sample_min_rotation_delta_deg = float(sample_min_rotation_delta_deg)
        # 方向样本的最小旋转增量（度）
        self.orientation_sample_min_rotation_delta_deg = float(orientation_sample_min_rotation_delta_deg)
        # 最小 XY 平面跨度（米）
        self.min_coverage_xy_span_m = float(min_coverage_xy_span_m)
        # 最小 Z 方向跨度（米）
        self.min_coverage_z_span_m = float(min_coverage_z_span_m)
        # 最小旋转跨度（度）
        self.min_coverage_rotation_span_deg = float(min_coverage_rotation_span_deg)
        # 可观测性要求：最小俯仰角跨度（度）
        self.min_pitch_span_deg = float(min_pitch_span_deg)
        # 可观测性要求：最小偏航角跨度（度）
        self.min_yaw_span_deg = float(min_yaw_span_deg)
        # 可观测性要求：最小翻滚角跨度（度）
        self.min_roll_span_deg = float(min_roll_span_deg)
        # 最少球体锚点样本数
        self.min_sphere_anchor_samples = int(min_sphere_anchor_samples)
        # 最少球体高度变化样本数
        self.min_sphere_height_samples = int(min_sphere_height_samples)
        # 最少球体壳层样本数
        self.min_sphere_shell_samples = int(min_sphere_shell_samples)
        # 外部传入的旋转角度差计算函数
        self._rotation_delta_deg = rotation_delta_deg

    # ------------------------------------------------------------------
    # 覆盖度评估（Coverage）
    # 关注的是机器人末端执行器在工作空间中的分布跨度
    # ------------------------------------------------------------------

    def coverage_metrics(self, records: Sequence[AcceptedSampleRecord]):
        """
        计算当前已接受样本的覆盖度量。
        若无样本则返回 None，否则返回包含数量、XYZ 跨度、XY 平面跨度、
        Z 跨度以及最大旋转差值的字典。
        """
        if not records:
            return None

        # 提取所有样本的末端位置
        translations = np.array([r.robot_pose.translation for r in records], dtype=float)
        # 计算各轴的极差（peak-to-peak）
        xyz_span = np.ptp(translations, axis=0)  # 形状 (3,)
        # XY 平面的欧氏距离跨度
        xy_span = float(np.linalg.norm(xyz_span[:2]))

        # 遍历所有样本对，找出最大旋转角度差
        max_rot_delta = 0.0
        poses = [r.robot_pose for r in records]
        for i, a in enumerate(poses):
            for b in poses[i + 1:]:
                max_rot_delta = max(max_rot_delta, self._rotation_delta_deg(a.rotation, b.rotation))

        return {
            "count": len(records),
            "xyz_span": xyz_span,  # numpy 数组，元素分别对应 x、y、z 跨度
            "xy_span": xy_span,
            "z_span": float(xyz_span[2]),
            "max_rot_delta_deg": max_rot_delta,
        }

    def coverage_status(
        self,
        records: Sequence[AcceptedSampleRecord],
        *,
        min_count: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        判断覆盖度是否满足要求。返回 (是否通过, 描述字符串)。
        可通过 min_count 临时覆写最小样本数要求。
        """
        m = self.coverage_metrics(records)
        if m is None:
            return False, "no accepted samples"

        required_count = self.min_successful_samples if min_count is None else int(min_count)
        # 逐项检查
        count_ok = m["count"] >= required_count
        xy_ok = m["xy_span"] >= self.min_coverage_xy_span_m
        z_ok = m["z_span"] >= self.min_coverage_z_span_m
        rot_ok = m["max_rot_delta_deg"] >= self.min_coverage_rotation_span_deg
        ok = count_ok and xy_ok and z_ok and rot_ok

        # 构造可读的判定结果字符串
        note = (
            f"count {m['count']}/{required_count} {'PASS' if count_ok else 'FAIL'}, "
            f"xy_span {m['xy_span']:.3f}/{self.min_coverage_xy_span_m:.3f} {'PASS' if xy_ok else 'FAIL'}, "
            f"z_span {m['z_span']:.3f}/{self.min_coverage_z_span_m:.3f} {'PASS' if z_ok else 'FAIL'}, "
            f"rot_span {m['max_rot_delta_deg']:.1f}/{self.min_coverage_rotation_span_deg:.1f} {'PASS' if rot_ok else 'FAIL'}, "
            f"xyz_span=({m['xyz_span'][0]:.3f},{m['xyz_span'][1]:.3f},{m['xyz_span'][2]:.3f})m"
        )
        return ok, note

    # ------------------------------------------------------------------
    # 可观测性评估（Observability）
    # 基于实际末端姿态相对于参考姿态的分解，而非规划时的名义偏移
    # ------------------------------------------------------------------

    def _actual_rotation_deltas(
        self, records: Sequence[AcceptedSampleRecord], reference_rotation: R
    ):
        """
        将每个样本的实际末端旋转相对于参考旋转（如原位姿）分解为
        末端局部坐标系下的 pitch / yaw / roll 三个欧拉角序列。
        旋转顺序为 'xyz'（外旋），返回三个列表。
        """
        pitches, yaws, rolls = [], [], []
        for rec in records:
            # 计算相对旋转：从参考旋转到当前旋转
            delta_r = reference_rotation.inv() * rec.robot_pose.rotation
            # 提取欧拉角（度）
            euler = delta_r.as_euler("xyz", degrees=True)
            pitches.append(float(euler[0]))
            yaws.append(float(euler[1]))
            rolls.append(float(euler[2]))
        return pitches, yaws, rolls

    def observability_metrics(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ):
        """
        计算可观测性度量。
        若未提供参考旋转，则默认使用第一个样本的末端姿态作为参考。
        返回包含各欧拉角跨度、各球体族样本数量的字典。
        """
        if not records:
            return None
        if reference_rotation is None and records:
            reference_rotation = records[0].robot_pose.rotation

        # 获取实际姿态相对参考的欧拉角序列
        pitches, yaws, rolls = self._actual_rotation_deltas(records, reference_rotation)

        # 统计各候选族（family）的数量
        family_counts = {
            CandidateFamily.SPHERE_ANCHOR: 0,
            CandidateFamily.SPHERE_HEIGHT: 0,
            CandidateFamily.SPHERE_SHELL: 0,
            CandidateFamily.SPHERE_ROLL_COVERAGE: 0,
        }
        for r in records:
            family_counts[r.family] = family_counts.get(r.family, 0) + 1

        return {
            "pitch_span_deg": max(pitches) - min(pitches) if pitches else 0.0,
            "yaw_span_deg": max(yaws) - min(yaws) if yaws else 0.0,
            "roll_span_deg": max(rolls) - min(rolls) if rolls else 0.0,
            "sphere_anchor_count": family_counts[CandidateFamily.SPHERE_ANCHOR],
            "sphere_height_count": family_counts[CandidateFamily.SPHERE_HEIGHT],
            "sphere_shell_count": family_counts[CandidateFamily.SPHERE_SHELL],
            "total_count": len(records),
        }

    def observability_status(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> Tuple[bool, str]:
        """
        判断可观测性是否满足要求。
        检查俯仰、偏航、翻滚角跨度以及球体锚点、高度样本数量。
        """
        m = self.observability_metrics(records, reference_rotation)
        if m is None:
            return False, "no accepted samples for observability check"

        pitch_ok = m["pitch_span_deg"] >= self.min_pitch_span_deg
        yaw_ok = m["yaw_span_deg"] >= self.min_yaw_span_deg
        roll_ok = m["roll_span_deg"] >= self.min_roll_span_deg
        sphere_anchor_ok = m["sphere_anchor_count"] >= self.min_sphere_anchor_samples
        sphere_height_ok = m["sphere_height_count"] >= self.min_sphere_height_samples
        ok = pitch_ok and yaw_ok and roll_ok and sphere_anchor_ok and sphere_height_ok

        parts = [
            f"pitch_span {m['pitch_span_deg']:.1f}/{self.min_pitch_span_deg:.1f}deg {'PASS' if pitch_ok else 'FAIL'}",
            f"yaw_span {m['yaw_span_deg']:.1f}/{self.min_yaw_span_deg:.1f}deg {'PASS' if yaw_ok else 'FAIL'}",
            f"roll_span {m['roll_span_deg']:.1f}/{self.min_roll_span_deg:.1f}deg {'PASS' if roll_ok else 'FAIL'}",
            f"sphere_anchor {m['sphere_anchor_count']}/{self.min_sphere_anchor_samples} {'PASS' if sphere_anchor_ok else 'FAIL'}",
            f"sphere_height {m['sphere_height_count']}/{self.min_sphere_height_samples} {'PASS' if sphere_height_ok else 'FAIL'}",
        ]
        return ok, ", ".join(parts)

    # ------------------------------------------------------------------
    # 综合门控
    # ------------------------------------------------------------------

    def dual_gate_status(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> Tuple[bool, str, dict, dict]:
        """
        一次性获取覆盖度和可观测性两方面的判定结果。
        返回 (是否全部通过, 简要描述, 覆盖度量字典, 可观测性度量字典)。
        """
        cov_ok, cov_note = self.coverage_status(records)
        obs_ok, obs_note = self.observability_status(records, reference_rotation)
        cov_m = self.coverage_metrics(records)
        obs_m = self.observability_metrics(records, reference_rotation)
        ok = cov_ok and obs_ok
        note = f"coverage={'PASS' if cov_ok else 'FAIL'} observability={'PASS' if obs_ok else 'FAIL'}"
        return ok, note, cov_m, obs_m

    def gate_deficits(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> dict:
        """
        返回一个布尔值字典，指示当前哪些具体指标未达标。
        便于后续定向补充缺失的样本类型。
        """
        cov_m = self.coverage_metrics(records)
        obs_m = self.observability_metrics(records, reference_rotation)
        if cov_m is None or obs_m is None:
            return {"count": True}  # 样本数为零，必然所有指标都不满足

        return {
            "count": cov_m["count"] < self.min_successful_samples,
            "xy": cov_m["xy_span"] < self.min_coverage_xy_span_m,
            "z": cov_m["z_span"] < self.min_coverage_z_span_m,
            "rot": cov_m["max_rot_delta_deg"] < self.min_coverage_rotation_span_deg,
            "pitch": obs_m["pitch_span_deg"] < self.min_pitch_span_deg,
            "yaw": obs_m["yaw_span_deg"] < self.min_yaw_span_deg,
            "roll": obs_m["roll_span_deg"] < self.min_roll_span_deg,
            "anchor": obs_m["sphere_anchor_count"] < self.min_sphere_anchor_samples,
            "height": obs_m["sphere_height_count"] < self.min_sphere_height_samples,
            "shell": obs_m["sphere_shell_count"] < self.min_sphere_shell_samples,
        }