from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateSpec:
    source: str
    right: float
    up: float
    dist: float
    roll: float
    tilt_x: float
    tilt_y: float

    def key(self):
        return (
            round(self.right, 4),
            round(self.up, 4),
            round(self.dist, 4),
            round(self.roll, 2),
            round(self.tilt_x, 2),
            round(self.tilt_y, 2),
        )


class SampleManager:
    def __init__(
        self,
        *,
        min_successful_samples: int,
        sample_min_translation_delta: float,
        sample_min_rotation_delta_deg: float,
        min_coverage_xy_span_m: float,
        min_coverage_z_span_m: float,
        min_coverage_rotation_span_deg: float,
        rank_visibility_margin_cap_px: float,
        rank_visibility_margin_scale_px: float,
        rank_visibility_side_cap_px: float,
        rank_visibility_side_scale_px: float,
        rank_center_penalty_weight: float,
        rank_right_coverage_deficit_weight: float,
        rank_right_coverage_base_weight: float,
        rank_up_coverage_deficit_weight: float,
        rank_up_coverage_base_weight: float,
        rank_dist_coverage_deficit_weight: float,
        rank_dist_coverage_base_weight: float,
        rank_rot_coverage_deficit_weight: float,
        rank_rot_coverage_base_weight: float,
        rank_path_segment_penalty_weight: float,
        rank_recenter_cost_penalty_weight: float,
        rotation_delta_deg: Callable,
    ):
        self.min_successful_samples = int(min_successful_samples)
        self.sample_min_translation_delta = float(sample_min_translation_delta)
        self.sample_min_rotation_delta_deg = float(sample_min_rotation_delta_deg)
        self.min_coverage_xy_span_m = float(min_coverage_xy_span_m)
        self.min_coverage_z_span_m = float(min_coverage_z_span_m)
        self.min_coverage_rotation_span_deg = float(min_coverage_rotation_span_deg)
        self.rank_visibility_margin_cap_px = float(rank_visibility_margin_cap_px)
        self.rank_visibility_margin_scale_px = max(1.0e-6, float(rank_visibility_margin_scale_px))
        self.rank_visibility_side_cap_px = float(rank_visibility_side_cap_px)
        self.rank_visibility_side_scale_px = max(1.0e-6, float(rank_visibility_side_scale_px))
        self.rank_center_penalty_weight = float(rank_center_penalty_weight)
        self.rank_right_coverage_deficit_weight = float(rank_right_coverage_deficit_weight)
        self.rank_right_coverage_base_weight = float(rank_right_coverage_base_weight)
        self.rank_up_coverage_deficit_weight = float(rank_up_coverage_deficit_weight)
        self.rank_up_coverage_base_weight = float(rank_up_coverage_base_weight)
        self.rank_dist_coverage_deficit_weight = float(rank_dist_coverage_deficit_weight)
        self.rank_dist_coverage_base_weight = float(rank_dist_coverage_base_weight)
        self.rank_rot_coverage_deficit_weight = float(rank_rot_coverage_deficit_weight)
        self.rank_rot_coverage_base_weight = float(rank_rot_coverage_base_weight)
        self.rank_path_segment_penalty_weight = float(rank_path_segment_penalty_weight)
        self.rank_recenter_cost_penalty_weight = float(rank_recenter_cost_penalty_weight)
        self._rotation_delta_deg = rotation_delta_deg
        self.accepted_sample_poses = []
        self.accepted_tracking_poses = []

    def reset(self):
        self.accepted_sample_poses.clear()
        self.accepted_tracking_poses.clear()

    def record_accepted_sample(self, robot_pose, tracking_pose):
        self.accepted_sample_poses.append(robot_pose)
        if tracking_pose is not None:
            self.accepted_tracking_poses.append(tracking_pose)

    def is_diverse_transform(self, sample_pose) -> Tuple[bool, str]:
        if not self.accepted_sample_poses:
            return True, "first sample"
        c_t = np.array(sample_pose.translation, dtype=float)
        for prev in self.accepted_sample_poses:
            p_t = np.array(prev.translation, dtype=float)
            trans_delta = float(np.linalg.norm(c_t - p_t))
            rot_delta_deg = self._rotation_delta_deg(prev.rotation, sample_pose.rotation)
            if (
                trans_delta < self.sample_min_translation_delta
                and rot_delta_deg < self.sample_min_rotation_delta_deg
            ):
                return (
                    False,
                    f"too close to accepted sample "
                    f"(dt={trans_delta:.3f}m, dr={rot_delta_deg:.1f}deg)",
                )
        return True, "diverse"

    def coverage_metrics(self):
        if not self.accepted_sample_poses:
            return None
        translations = np.array([p.translation for p in self.accepted_sample_poses], dtype=float)
        xyz_span = np.ptp(translations, axis=0)
        xy_span = float(np.linalg.norm(xyz_span[:2]))
        max_rot_delta = 0.0
        for i, a in enumerate(self.accepted_sample_poses):
            for b in self.accepted_sample_poses[i + 1:]:
                max_rot_delta = max(max_rot_delta, self._rotation_delta_deg(a.rotation, b.rotation))
        return {
            "count": len(self.accepted_sample_poses),
            "xyz_span": xyz_span,
            "xy_span": xy_span,
            "z_span": float(xyz_span[2]),
            "max_rot_delta_deg": max_rot_delta,
        }

    def coverage_status(self) -> Tuple[bool, str]:
        metrics = self.coverage_metrics()
        if metrics is None:
            return False, "no accepted samples"
        count_ok = metrics["count"] >= self.min_successful_samples
        xy_ok = metrics["xy_span"] >= self.min_coverage_xy_span_m
        z_ok = metrics["z_span"] >= self.min_coverage_z_span_m
        rot_ok = metrics["max_rot_delta_deg"] >= self.min_coverage_rotation_span_deg
        ok = count_ok and xy_ok and z_ok and rot_ok
        note = (
            f"count {metrics['count']}/{self.min_successful_samples} "
            f"{'PASS' if count_ok else 'FAIL'}, "
            f"xy_span {metrics['xy_span']:.3f}/{self.min_coverage_xy_span_m:.3f} "
            f"{'PASS' if xy_ok else 'FAIL'}, "
            f"z_span {metrics['z_span']:.3f}/{self.min_coverage_z_span_m:.3f} "
            f"{'PASS' if z_ok else 'FAIL'}, "
            f"rot_span {metrics['max_rot_delta_deg']:.1f}/{self.min_coverage_rotation_span_deg:.1f} "
            f"{'PASS' if rot_ok else 'FAIL'}, "
            f"xyz_span=({metrics['xyz_span'][0]:.3f},{metrics['xyz_span'][1]:.3f},{metrics['xyz_span'][2]:.3f})m"
        )
        return ok, note

    def rank_candidates(self, candidates, *, danger_penalty_fn: Callable[[CandidateSpec], float]):
        metrics = self.coverage_metrics() or {}
        xy_span = float(metrics.get("xy_span", 0.0))
        z_span = float(metrics.get("z_span", 0.0))
        rot_span = float(metrics.get("max_rot_delta_deg", 0.0))
        xy_deficit = max(0.0, self.min_coverage_xy_span_m - xy_span)
        z_deficit = max(0.0, self.min_coverage_z_span_m - z_span)
        rot_deficit = max(0.0, self.min_coverage_rotation_span_deg - rot_span)

        ranked = []
        for candidate in candidates:
            spec = candidate.spec
            visibility_margin_score = max(
                0.0,
                min(candidate.projected_margin_px, self.rank_visibility_margin_cap_px)
                / self.rank_visibility_margin_scale_px,
            )
            visibility_side_score = max(
                0.0,
                min(candidate.projected_marker_px, self.rank_visibility_side_cap_px)
                / self.rank_visibility_side_scale_px,
            )
            center_penalty = self.rank_center_penalty_weight * max(
                0.0, candidate.projected_center_error_px
            )
            coverage_gain = 0.0
            if spec is not None:
                coverage_gain += min(1.0, abs(spec.right) / max(self.min_coverage_xy_span_m, 1.0e-6)) * (
                    self.rank_right_coverage_deficit_weight
                    if xy_deficit > 1.0e-6
                    else self.rank_right_coverage_base_weight
                )
                coverage_gain += min(1.0, abs(spec.up) / max(self.min_coverage_xy_span_m, 1.0e-6)) * (
                    self.rank_up_coverage_deficit_weight
                    if xy_deficit > 1.0e-6
                    else self.rank_up_coverage_base_weight
                )
                coverage_gain += min(1.0, abs(spec.dist) / max(self.min_coverage_z_span_m, 1.0e-6)) * (
                    self.rank_dist_coverage_deficit_weight
                    if z_deficit > 1.0e-6
                    else self.rank_dist_coverage_base_weight
                )
                rot_mag = max(abs(spec.roll), abs(spec.tilt_x), abs(spec.tilt_y))
                coverage_gain += min(1.0, rot_mag / max(self.min_coverage_rotation_span_deg, 1.0e-6)) * (
                    self.rank_rot_coverage_deficit_weight
                    if rot_deficit > 1.0e-6
                    else self.rank_rot_coverage_base_weight
                )
            path_risk_penalty = self.rank_path_segment_penalty_weight * max(
                0, candidate.segment_count - 1
            )
            recenter_cost_penalty = self.rank_recenter_cost_penalty_weight * max(
                0.0, candidate.projected_center_error_px
            )
            danger_axis_penalty = danger_penalty_fn(spec) if spec is not None else 0.0
            score = (
                visibility_margin_score
                + visibility_side_score
                + coverage_gain
                - center_penalty
                - path_risk_penalty
                - recenter_cost_penalty
                - danger_axis_penalty
            )
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked


class AdaptiveCandidatePlanner:
    def __init__(
        self,
        *,
        right_levels_m: Sequence[float],
        up_levels_m: Sequence[float],
        dist_levels_m: Sequence[float],
        roll_levels_deg: Sequence[float],
        tilt_levels_deg: Sequence[float],
        max_candidate_attempts: int,
        axis_expand_success_streak: int,
        pair_enable_success_count: int,
        corner_enable_success_count: int,
        corner_cooldown_steps: int,
        axis_failure_penalty_increment: float,
        axis_failure_penalty_decay: float,
        axis_failure_penalty_max: float,
        pair_risk_penalty: float,
        corner_risk_penalty: float,
    ):
        self.right_levels_m = self._levels(right_levels_m)
        self.up_levels_m = self._levels(up_levels_m)
        self.dist_levels_m = self._levels(dist_levels_m)
        self.roll_levels_deg = self._levels(roll_levels_deg)
        self.tilt_levels_deg = self._levels(tilt_levels_deg)
        self.max_candidate_attempts = int(max_candidate_attempts)
        self.axis_expand_success_streak = max(1, int(axis_expand_success_streak))
        self.pair_enable_success_count = max(0, int(pair_enable_success_count))
        self.corner_enable_success_count = max(0, int(corner_enable_success_count))
        self.corner_cooldown_steps = max(0, int(corner_cooldown_steps))
        self.axis_failure_penalty_increment = float(axis_failure_penalty_increment)
        self.axis_failure_penalty_decay = float(axis_failure_penalty_decay)
        self.axis_failure_penalty_max = float(axis_failure_penalty_max)
        self.pair_risk_penalty = float(pair_risk_penalty)
        self.corner_risk_penalty = float(corner_risk_penalty)

        self._axis_index: Dict[str, int] = {
            "right": 0,
            "up": 0,
            "dist": 0,
            "roll": 0,
            "tilt_x": 0,
            "tilt_y": 0,
        }
        self._axis_success_streak = {name: 0 for name in self._axis_index}
        self._attempted = set()
        self._total_success = 0
        self._corner_cooldown = 0
        self._axis_failure_score = {name: 0.0 for name in self._axis_index}

    def reset(self):
        for axis in self._axis_index:
            self._axis_index[axis] = 0
            self._axis_success_streak[axis] = 0
        self._attempted.clear()
        self._total_success = 0
        self._corner_cooldown = 0
        for axis in self._axis_failure_score:
            self._axis_failure_score[axis] = 0.0

    @staticmethod
    def _levels(values: Sequence[float]) -> List[float]:
        levels = sorted({abs(float(v)) for v in values if abs(float(v)) > 1.0e-9})
        return levels or [0.0]

    def _axis_radius(self, axis_name: str) -> float:
        levels_map = {
            "right": self.right_levels_m,
            "up": self.up_levels_m,
            "dist": self.dist_levels_m,
            "roll": self.roll_levels_deg,
            "tilt_x": self.tilt_levels_deg,
            "tilt_y": self.tilt_levels_deg,
        }
        levels = levels_map[axis_name]
        idx = min(self._axis_index[axis_name], len(levels) - 1)
        return levels[idx]

    def status_note(self) -> str:
        return (
            "adaptive_radii="
            f"right:{self._axis_radius('right'):.3f}m, "
            f"up:{self._axis_radius('up'):.3f}m, "
            f"dist:{self._axis_radius('dist'):.3f}m, "
            f"roll:{self._axis_radius('roll'):.1f}deg, "
            f"tilt:{self._axis_radius('tilt_x'):.1f}deg, "
            f"successes={self._total_success}, cooldown={self._corner_cooldown}"
        )

    def axis_risk_penalty(self, spec: CandidateSpec) -> float:
        penalty = 0.0
        for axis, value in (
            ("right", spec.right),
            ("up", spec.up),
            ("dist", spec.dist),
            ("roll", spec.roll),
            ("tilt_x", spec.tilt_x),
            ("tilt_y", spec.tilt_y),
        ):
            if abs(value) > 1.0e-9:
                penalty += self._axis_failure_score[axis]
        if "corner" in spec.source:
            penalty += self.corner_risk_penalty
        elif "pair" in spec.source:
            penalty += self.pair_risk_penalty
        return penalty

    def build_specs(self) -> List[CandidateSpec]:
        r = self._axis_radius("right")
        u = self._axis_radius("up")
        d = self._axis_radius("dist")
        roll = self._axis_radius("roll")
        tilt_x = self._axis_radius("tilt_x")
        tilt_y = self._axis_radius("tilt_y")

        specs = [
            CandidateSpec("seed-center", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            CandidateSpec("seed-right+", r, 0.0, 0.0, 0.0, 0.0, 0.0),
            CandidateSpec("seed-right-", -r, 0.0, 0.0, 0.0, 0.0, 0.0),
            CandidateSpec("seed-up+", 0.0, u, 0.0, 0.0, 0.0, 0.0),
            CandidateSpec("seed-up-", 0.0, -u, 0.0, 0.0, 0.0, 0.0),
            CandidateSpec("seed-dist+", 0.0, 0.0, d, 0.0, 0.0, 0.0),
            CandidateSpec("seed-dist-", 0.0, 0.0, -d, 0.0, 0.0, 0.0),
            CandidateSpec("seed-roll+", 0.0, 0.0, 0.0, roll, 0.0, 0.0),
            CandidateSpec("seed-roll-", 0.0, 0.0, 0.0, -roll, 0.0, 0.0),
            CandidateSpec("seed-tilt-x+", 0.0, 0.0, 0.0, 0.0, tilt_x, 0.0),
            CandidateSpec("seed-tilt-x-", 0.0, 0.0, 0.0, 0.0, -tilt_x, 0.0),
            CandidateSpec("seed-tilt-y+", 0.0, 0.0, 0.0, 0.0, 0.0, tilt_y),
            CandidateSpec("seed-tilt-y-", 0.0, 0.0, 0.0, 0.0, 0.0, -tilt_y),
        ]

        if self._total_success >= self.pair_enable_success_count and self._corner_cooldown == 0:
            specs.extend(
                [
                    CandidateSpec("pair-right-dist+", r, 0.0, d, 0.0, 0.0, 0.0),
                    CandidateSpec("pair-right-dist-", -r, 0.0, d, 0.0, 0.0, 0.0),
                    CandidateSpec("pair-up-dist+", 0.0, u, d, 0.0, 0.0, 0.0),
                    CandidateSpec("pair-up-dist-", 0.0, -u, d, 0.0, 0.0, 0.0),
                    CandidateSpec("pair-roll-tilt-x", 0.0, 0.0, 0.0, roll, tilt_x, 0.0),
                    CandidateSpec("pair-roll-tilt-y", 0.0, 0.0, 0.0, roll, 0.0, tilt_y),
                ]
            )
        if self._total_success >= self.corner_enable_success_count and self._corner_cooldown == 0:
            specs.extend(
                [
                    CandidateSpec("corner-1", r, u, d, roll, tilt_x, tilt_y),
                    CandidateSpec("corner-2", -r, u, d, -roll, -tilt_x, tilt_y),
                    CandidateSpec("corner-3", r, -u, -d, roll, tilt_x, -tilt_y),
                    CandidateSpec("corner-4", -r, -u, d, -roll, -tilt_x, -tilt_y),
                ]
            )

        result = []
        seen = set()
        for spec in specs:
            if spec.key() in seen or spec.key() in self._attempted:
                continue
            seen.add(spec.key())
            result.append(spec)
            if len(result) >= self.max_candidate_attempts:
                break
        return result

    def feedback(self, spec: CandidateSpec, ok: bool, reason: str):
        self._attempted.add(spec.key())
        if self._corner_cooldown > 0:
            self._corner_cooldown -= 1

        axes = []
        if abs(spec.right) > 1.0e-9:
            axes.append("right")
        if abs(spec.up) > 1.0e-9:
            axes.append("up")
        if abs(spec.dist) > 1.0e-9:
            axes.append("dist")
        if abs(spec.roll) > 1.0e-9:
            axes.append("roll")
        if abs(spec.tilt_x) > 1.0e-9:
            axes.append("tilt_x")
        if abs(spec.tilt_y) > 1.0e-9:
            axes.append("tilt_y")

        if ok:
            self._total_success += 1
            for axis in axes:
                self._axis_failure_score[axis] = max(
                    0.0, self._axis_failure_score[axis] - self.axis_failure_penalty_decay
                )
                self._axis_success_streak[axis] += 1
                if self._axis_success_streak[axis] >= self.axis_expand_success_streak:
                    self._axis_index[axis] += 1
                    self._axis_success_streak[axis] = 0
            return

        lower_reason = reason.lower()
        shrink = (
            "marker_lost" in lower_reason
            or "cannot_reacquire" in lower_reason
            or "no markers" in lower_reason
            or "stale" in lower_reason
            or "cannot recenter" in lower_reason
            or "recenter limit" in lower_reason
            or "recenter_sign_failed" in lower_reason
            or "recenter_error_not_decreasing" in lower_reason
            or "preplan_failed" in lower_reason
        )
        if "too close" in lower_reason:
            return

        if shrink:
            for axis in axes:
                self._axis_index[axis] = max(0, self._axis_index[axis] - 1)
                self._axis_success_streak[axis] = 0
                self._axis_failure_score[axis] = min(
                    self.axis_failure_penalty_max,
                    self._axis_failure_score[axis] + self.axis_failure_penalty_increment,
                )
            if "corner" in spec.source or "pair" in spec.source:
                self._corner_cooldown = max(self._corner_cooldown, self.corner_cooldown_steps)
