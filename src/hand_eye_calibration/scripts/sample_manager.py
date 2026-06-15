from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Family constants
# ---------------------------------------------------------------------------

class CandidateFamily:
    ANCHOR = "anchor_pose"
    SOLVER_CORE = "solver_core"
    DEPTH = "depth_span"
    SAFE_LATERAL = "safe_lateral"
    COVERAGE_ROLL = "coverage_roll"
    RISKY = "risky_recovery"


# Removal priority (higher = more preferred to remove).
_FAMILY_REMOVE_PRIORITY = {
    CandidateFamily.RISKY: 3,
    CandidateFamily.COVERAGE_ROLL: 2,
    CandidateFamily.SAFE_LATERAL: 2,
    CandidateFamily.DEPTH: 1,
    CandidateFamily.SOLVER_CORE: 0,
    CandidateFamily.ANCHOR: 0,
}


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseOffsetPose:
    """A single family-based base-offset candidate record.

    Canonical definition — imported by collector_config for YAML parsing.
    """

    label: str = ""
    family: str = ""
    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    removable: bool = False
    intent: str = ""
    observability_axis: str = "none"  # pitch | yaw | roll | none
    dedup_protected: bool = False     # skip normal dt/dr dedup


@dataclass(frozen=True)
class CandidateSpec:
    """Resolved candidate with a unique key for dedup."""

    source: str
    base_x: float
    base_y: float
    base_z: float
    pitch: float
    yaw: float
    roll: float
    family: str
    removable: bool
    intent: str = ""
    observability_axis: str = "none"
    dedup_protected: bool = False

    def exact_key(self):
        """Exact-pose key for protected dedup."""
        return (
            round(self.base_x, 4), round(self.base_y, 4), round(self.base_z, 4),
            round(self.pitch, 2), round(self.yaw, 2), round(self.roll, 2),
        )

    def dedup_key(self):
        """Coarse key for normal dedup (ignores small differences)."""
        return self.exact_key()


@dataclass(frozen=True)
class AcceptedSampleQuality:
    center_error_px: float
    margin_px: float
    marker_side_px: float
    distance_m: float
    marker_note: str
    model_note: str
    stable_note: str


@dataclass(frozen=True)
class AcceptedSampleRecord:
    robot_pose: object
    tracking_pose: object
    family: str
    spec: CandidateSpec
    quality: AcceptedSampleQuality
    candidate_idx: int
    candidate_description: str
    recenter_attempted: bool
    recenter_strict_converged: bool
    removable: bool


# ---------------------------------------------------------------------------
# SampleSetGovernor — unified coverage + observability + subset governance
# ---------------------------------------------------------------------------


class SampleSetGovernor:
    """Unified governance over the accepted sample set.

    Replaces the previously scattered coverage_status*, observability_status*,
    should_skip_risky_family, and subset-optimizer entry points with a single
    class that owns both gate logic and subset-search orchestration.
    """

    def __init__(
        self,
        *,
        min_successful_samples: int,
        sample_min_translation_delta: float,
        sample_min_rotation_delta_deg: float,
        min_coverage_xy_span_m: float,
        min_coverage_z_span_m: float,
        min_coverage_rotation_span_deg: float,
        min_pitch_span_deg: float,
        min_yaw_span_deg: float,
        min_roll_span_deg: float,
        min_anchor_pose_samples: int,
        min_depth_span_samples: int,
        min_lateral_samples: int,
        rotation_delta_deg: Callable,
    ):
        self.min_successful_samples = int(min_successful_samples)
        self.sample_min_translation_delta = float(sample_min_translation_delta)
        self.sample_min_rotation_delta_deg = float(sample_min_rotation_delta_deg)
        self.min_coverage_xy_span_m = float(min_coverage_xy_span_m)
        self.min_coverage_z_span_m = float(min_coverage_z_span_m)
        self.min_coverage_rotation_span_deg = float(min_coverage_rotation_span_deg)
        self.min_pitch_span_deg = float(min_pitch_span_deg)
        self.min_yaw_span_deg = float(min_yaw_span_deg)
        self.min_roll_span_deg = float(min_roll_span_deg)
        self.min_anchor_pose_samples = int(min_anchor_pose_samples)
        self.min_depth_span_samples = int(min_depth_span_samples)
        self.min_lateral_samples = int(min_lateral_samples)
        self._rotation_delta_deg = rotation_delta_deg

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def coverage_metrics(self, records: Sequence[AcceptedSampleRecord]):
        if not records:
            return None
        translations = np.array([r.robot_pose.translation for r in records], dtype=float)
        xyz_span = np.ptp(translations, axis=0)
        xy_span = float(np.linalg.norm(xyz_span[:2]))
        max_rot_delta = 0.0
        poses = [r.robot_pose for r in records]
        for i, a in enumerate(poses):
            for b in poses[i + 1:]:
                max_rot_delta = max(max_rot_delta, self._rotation_delta_deg(a.rotation, b.rotation))
        return {
            "count": len(records),
            "xyz_span": xyz_span,
            "xy_span": xy_span,
            "z_span": float(xyz_span[2]),
            "max_rot_delta_deg": max_rot_delta,
        }

    def coverage_status(self, records: Sequence[AcceptedSampleRecord]) -> Tuple[bool, str]:
        m = self.coverage_metrics(records)
        if m is None:
            return False, "no accepted samples"
        count_ok = m["count"] >= self.min_successful_samples
        xy_ok = m["xy_span"] >= self.min_coverage_xy_span_m
        z_ok = m["z_span"] >= self.min_coverage_z_span_m
        rot_ok = m["max_rot_delta_deg"] >= self.min_coverage_rotation_span_deg
        ok = count_ok and xy_ok and z_ok and rot_ok
        note = (
            f"count {m['count']}/{self.min_successful_samples} {'PASS' if count_ok else 'FAIL'}, "
            f"xy_span {m['xy_span']:.3f}/{self.min_coverage_xy_span_m:.3f} {'PASS' if xy_ok else 'FAIL'}, "
            f"z_span {m['z_span']:.3f}/{self.min_coverage_z_span_m:.3f} {'PASS' if z_ok else 'FAIL'}, "
            f"rot_span {m['max_rot_delta_deg']:.1f}/{self.min_coverage_rotation_span_deg:.1f} {'PASS' if rot_ok else 'FAIL'}, "
            f"xyz_span=({m['xyz_span'][0]:.3f},{m['xyz_span'][1]:.3f},{m['xyz_span'][2]:.3f})m"
        )
        return ok, note

    # ------------------------------------------------------------------
    # Observability (based on ACTUAL pose deltas, not spec offsets)
    # ------------------------------------------------------------------

    def _actual_rotation_deltas(
        self, records: Sequence[AcceptedSampleRecord], reference_rotation: R
    ):
        """Decompose each sample's actual EE rotation relative to the
        reference (original_place) into pitch / yaw / roll in the EE
        local frame (XYZ Euler order)."""
        pitches, yaws, rolls = [], [], []
        for rec in records:
            delta_r = reference_rotation.inv() * rec.robot_pose.rotation
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
        if not records:
            return None
        if reference_rotation is None and records:
            reference_rotation = records[0].robot_pose.rotation

        pitches, yaws, rolls = self._actual_rotation_deltas(records, reference_rotation)

        family_counts = {
            CandidateFamily.ANCHOR: 0,
            CandidateFamily.SOLVER_CORE: 0,
            CandidateFamily.DEPTH: 0,
            CandidateFamily.SAFE_LATERAL: 0,
            CandidateFamily.COVERAGE_ROLL: 0,
            CandidateFamily.RISKY: 0,
        }
        for r in records:
            family_counts[r.family] = family_counts.get(r.family, 0) + 1

        return {
            "pitch_span_deg": max(pitches) - min(pitches) if pitches else 0.0,
            "yaw_span_deg": max(yaws) - min(yaws) if yaws else 0.0,
            "roll_span_deg": max(rolls) - min(rolls) if rolls else 0.0,
            "anchor_count": family_counts[CandidateFamily.ANCHOR],
            "depth_count": family_counts[CandidateFamily.DEPTH],
            "lateral_count": family_counts[CandidateFamily.SAFE_LATERAL],
            "risky_count": family_counts[CandidateFamily.RISKY],
            "total_count": len(records),
        }

    def observability_status(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> Tuple[bool, str]:
        m = self.observability_metrics(records, reference_rotation)
        if m is None:
            return False, "no accepted samples for observability check"

        pitch_ok = m["pitch_span_deg"] >= self.min_pitch_span_deg
        yaw_ok = m["yaw_span_deg"] >= self.min_yaw_span_deg
        roll_ok = m["roll_span_deg"] >= self.min_roll_span_deg
        anchor_ok = m["anchor_count"] >= self.min_anchor_pose_samples
        depth_ok = m["depth_count"] >= self.min_depth_span_samples
        lateral_ok = m["lateral_count"] >= self.min_lateral_samples
        ok = pitch_ok and yaw_ok and roll_ok and anchor_ok and depth_ok and lateral_ok

        parts = [
            f"pitch_span {m['pitch_span_deg']:.1f}/{self.min_pitch_span_deg:.1f}deg {'PASS' if pitch_ok else 'FAIL'}",
            f"yaw_span {m['yaw_span_deg']:.1f}/{self.min_yaw_span_deg:.1f}deg {'PASS' if yaw_ok else 'FAIL'}",
            f"roll_span {m['roll_span_deg']:.1f}/{self.min_roll_span_deg:.1f}deg {'PASS' if roll_ok else 'FAIL'}",
            f"anchor {m['anchor_count']}/{self.min_anchor_pose_samples} {'PASS' if anchor_ok else 'FAIL'}",
            f"depth {m['depth_count']}/{self.min_depth_span_samples} {'PASS' if depth_ok else 'FAIL'}",
            f"lateral {m['lateral_count']}/{self.min_lateral_samples} {'PASS' if lateral_ok else 'FAIL'}",
        ]
        return ok, ", ".join(parts)

    # ------------------------------------------------------------------
    # Combined gate
    # ------------------------------------------------------------------

    def dual_gate_status(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> Tuple[bool, str, dict, dict]:
        cov_ok, cov_note = self.coverage_status(records)
        obs_ok, obs_note = self.observability_status(records, reference_rotation)
        cov_m = self.coverage_metrics(records)
        obs_m = self.observability_metrics(records, reference_rotation)
        ok = cov_ok and obs_ok
        note = f"coverage={'PASS' if cov_ok else 'FAIL'} observability={'PASS' if obs_ok else 'FAIL'}"
        return ok, note, cov_m, obs_m

    # ------------------------------------------------------------------
    # Subset search (geometric-first)
    # ------------------------------------------------------------------

    def find_best_geometric_subset(
        self,
        records: List[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> Tuple[Optional[Tuple[int, ...]], str]:
        """Return (best_remove_indices, note).

        Searches for a removable subset whose removal preserves both
        coverage and observability.  Preferentially removes RISKY
        samples, then SAFE_LATERAL, then DEPTH.
        """
        if not records:
            return None, "no records"

        removable = [
            idx for idx, rec in enumerate(records)
            if rec.removable
        ]
        if not removable:
            return None, "no removable samples"

        # Score a candidate subset.
        def _eval(keep_indices: Sequence[int]):
            subset = [records[i] for i in keep_indices]
            cov_ok, cov_note = self.coverage_status(subset)
            obs_ok, obs_note = self.observability_status(subset, reference_rotation)
            removed = [i for i in range(len(records)) if i not in keep_indices]
            fam_score = sum(
                _FAMILY_REMOVE_PRIORITY.get(records[i].family, 0)
                for i in removed
            )
            # Lower score is better.
            layer = 0 if (cov_ok and obs_ok) else (1 if (cov_ok or obs_ok) else 2)
            return (layer, -fam_score, len(removed)), cov_ok, obs_ok, cov_note, obs_note

        all_idx = list(range(len(records)))
        base_score, base_cov, base_obs, base_cov_note, base_obs_note = _eval(all_idx)
        best_score = base_score
        best_remove: Optional[Tuple[int, ...]] = None
        best_cov_ok = base_cov
        best_obs_ok = base_obs

        max_remove = min(len(removable), 3)

        # Greedy backward elimination.
        greedy_removed = []
        for _ in range(max_remove):
            step_best = None
            for idx in removable:
                if idx in greedy_removed:
                    continue
                trial_removed = tuple(sorted(greedy_removed + [idx]))
                trial_keep = [i for i in all_idx if i not in trial_removed]
                score, cov_ok, obs_ok, _, _ = _eval(trial_keep)
                if step_best is None or score < step_best[0]:
                    step_best = (score, trial_removed, cov_ok, obs_ok)
            if step_best is None or step_best[0] >= best_score if best_remove is None else step_best[0] >= best_score:
                break
            best_score = step_best[0]
            best_remove = step_best[1]
            best_cov_ok = step_best[2]
            best_obs_ok = step_best[3]
            greedy_removed = list(best_remove)

        # Small combinatorial check.
        from itertools import combinations
        for size in range(1, max_remove + 1):
            for combo in combinations(removable, size):
                keep = [i for i in all_idx if i not in combo]
                score, cov_ok, obs_ok, _, _ = _eval(keep)
                if best_remove is not None and score >= best_score:
                    continue
                if score < best_score if best_remove is not None else True:
                    best_score = score
                    best_remove = combo
                    best_cov_ok = cov_ok
                    best_obs_ok = obs_ok

        if best_remove is None:
            return None, "full_set is already optimal (no removable improvement found)"

        note = (
            f"remove={list(best_remove)}; "
            f"coverage={'PASS' if best_cov_ok else 'FAIL'} "
            f"observability={'PASS' if best_obs_ok else 'FAIL'}"
        )
        return best_remove, note


# ---------------------------------------------------------------------------
# SampleManager
# ---------------------------------------------------------------------------


class SampleManager:
    def __init__(
        self,
        *,
        base_offsets: Dict[str, List[BaseOffsetPose]],
        governor: SampleSetGovernor,
        nominal_translation_delta_scale: float,
        nominal_rotation_delta_scale: float,
        rotation_delta_deg: Callable,
    ):
        self._base_offsets = base_offsets
        self.governor = governor
        self.nominal_translation_delta_scale = float(nominal_translation_delta_scale)
        self.nominal_rotation_delta_scale = float(nominal_rotation_delta_scale)
        self._rotation_delta_deg = rotation_delta_deg
        self._accepted_samples: List[AcceptedSampleRecord] = []
        self._reference_rotation: Optional[R] = None

    # ------------------------------------------------------------------
    # Accepted sample access
    # ------------------------------------------------------------------

    @property
    def accepted_samples(self) -> List[AcceptedSampleRecord]:
        return self._accepted_samples

    @property
    def accepted_sample_poses(self):
        return [r.robot_pose for r in self._accepted_samples]

    @property
    def accepted_tracking_poses(self):
        return [r.tracking_pose for r in self._accepted_samples if r.tracking_pose is not None]

    def reset(self):
        self._accepted_samples.clear()
        self._reference_rotation = None

    def set_reference_rotation(self, rotation: R):
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
        self._accepted_samples.pop(index)

    def subset_records(self, keep_indices: Sequence[int]) -> List[AcceptedSampleRecord]:
        keep = set(int(idx) for idx in keep_indices)
        return [r for idx, r in enumerate(self._accepted_samples) if idx in keep]

    def removable_indices(self) -> List[int]:
        return [idx for idx, r in enumerate(self._accepted_samples) if r.removable]

    # ------------------------------------------------------------------
    # Candidate generation (family-based, dedup at generation)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_spec(offset: BaseOffsetPose) -> CandidateSpec:
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
        """Generate candidates in family order with protected-orientation dedup.

        - Orientation candidates (dedup_protected=True) are only deduped
          by exact pose key match, never by dt/dr proximity.
        - Coverage candidates use normal dt/dr dedup to avoid generating
          candidates that will be rejected as actual_too_close at runtime.
        """
        exact_seen = set()
        specs: List[CandidateSpec] = []

        family_order = [
            "anchor_roll", "anchor_pitch", "anchor_yaw",
            "anchor_yaw_expansion",
            "solver_core",
            "depth_span", "lateral_span", "coverage_roll", "risky_recovery",
        ]

        for family_name in family_order:
            offsets = self._base_offsets.get(family_name, [])
            for offset in offsets:
                spec = self._make_spec(offset)
                exact_k = spec.exact_key()

                # Exact dedup always applies.
                if exact_k in exact_seen:
                    continue
                exact_seen.add(exact_k)

                # Proximity dedup only for non-protected (coverage) candidates.
                if not spec.dedup_protected:
                    too_close = False
                    for prev_spec in specs:
                        prev_t = np.array([prev_spec.base_x, prev_spec.base_y, prev_spec.base_z])
                        this_t = np.array([spec.base_x, spec.base_y, spec.base_z])
                        dt = float(np.linalg.norm(this_t - prev_t))
                        dr = max(
                            abs(spec.pitch - prev_spec.pitch),
                            abs(spec.yaw - prev_spec.yaw),
                            abs(spec.roll - prev_spec.roll),
                        )
                        if dt < self.governor.sample_min_translation_delta and dr < self.governor.sample_min_rotation_delta_deg:
                            too_close = True
                            break
                    if too_close:
                        continue

                specs.append(spec)

        return specs

    # ------------------------------------------------------------------
    # Diversity checks
    # ------------------------------------------------------------------

    def _diversity_status(
        self,
        sample_pose,
        *,
        translation_threshold: float,
        rotation_threshold_deg: float,
        prefix: str = "",
    ) -> Tuple[bool, str]:
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
        return self._diversity_status(
            sample_pose,
            translation_threshold=self.governor.sample_min_translation_delta,
            rotation_threshold_deg=self.governor.sample_min_rotation_delta_deg,
        )

    def nominal_diversity_status(self, sample_pose) -> Tuple[bool, str]:
        nominal_trans = max(1e-6, self.governor.sample_min_translation_delta * self.nominal_translation_delta_scale)
        nominal_rot = max(1e-6, self.governor.sample_min_rotation_delta_deg * self.nominal_rotation_delta_scale)
        ok, note = self._diversity_status(sample_pose, translation_threshold=nominal_trans, rotation_threshold_deg=nominal_rot)
        if ok:
            return True, note
        return False, f"nominal_too_close: {note}"

    # ------------------------------------------------------------------
    # Convenience delegates to governor
    # ------------------------------------------------------------------

    def coverage_metrics(self):
        return self.governor.coverage_metrics(self._accepted_samples)

    def coverage_status(self) -> Tuple[bool, str]:
        return self.governor.coverage_status(self._accepted_samples)

    def observability_metrics(self):
        return self.governor.observability_metrics(self._accepted_samples, self._reference_rotation)

    def observability_status(self) -> Tuple[bool, str]:
        return self.governor.observability_status(self._accepted_samples, self._reference_rotation)

    def dual_gate_status(self) -> Tuple[bool, str, dict, dict]:
        return self.governor.dual_gate_status(self._accepted_samples, self._reference_rotation)

    def find_best_geometric_subset(self) -> Tuple[Optional[Tuple[int, ...]], str]:
        return self.governor.find_best_geometric_subset(self._accepted_samples, self._reference_rotation)
