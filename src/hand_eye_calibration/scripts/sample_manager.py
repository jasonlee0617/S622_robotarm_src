from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Family constants
# ---------------------------------------------------------------------------

class CandidateFamily:
    SPHERE_ANCHOR = "sphere_anchor"
    SPHERE_HEIGHT = "sphere_height"
    SPHERE_SHELL = "sphere_shell"
    SPHERE_ROLL_COVERAGE = "sphere_roll_coverage"


FAMILY_EXECUTION_ORDER = [
    "sphere_anchor",
    "sphere_height",
    "sphere_shell",
    "sphere_roll_coverage",
]


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
        orientation_sample_min_rotation_delta_deg: float,
        min_coverage_xy_span_m: float,
        min_coverage_z_span_m: float,
        min_coverage_rotation_span_deg: float,
        min_pitch_span_deg: float,
        min_yaw_span_deg: float,
        min_roll_span_deg: float,
        min_sphere_anchor_samples: int,
        min_sphere_height_samples: int,
        rotation_delta_deg: Callable,
    ):
        self.min_successful_samples = int(min_successful_samples)
        self.sample_min_translation_delta = float(sample_min_translation_delta)
        self.sample_min_rotation_delta_deg = float(sample_min_rotation_delta_deg)
        self.orientation_sample_min_rotation_delta_deg = float(orientation_sample_min_rotation_delta_deg)
        self.min_coverage_xy_span_m = float(min_coverage_xy_span_m)
        self.min_coverage_z_span_m = float(min_coverage_z_span_m)
        self.min_coverage_rotation_span_deg = float(min_coverage_rotation_span_deg)
        self.min_pitch_span_deg = float(min_pitch_span_deg)
        self.min_yaw_span_deg = float(min_yaw_span_deg)
        self.min_roll_span_deg = float(min_roll_span_deg)
        self.min_sphere_anchor_samples = int(min_sphere_anchor_samples)
        self.min_sphere_height_samples = int(min_sphere_height_samples)
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

    def gate_deficits(
        self,
        records: Sequence[AcceptedSampleRecord],
        reference_rotation: Optional[R] = None,
    ) -> dict:
        """Return a dict of booleans indicating which gates are currently failing."""
        cov_m = self.coverage_metrics(records)
        obs_m = self.observability_metrics(records, reference_rotation)
        if cov_m is None or obs_m is None:
            return {"count": True}
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
            "shell": obs_m["sphere_shell_count"] < 1,  # conservative: at least 1 shell sample
        }

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

    @property
    def reference_rotation(self):
        return self._reference_rotation

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

        for family_name in FAMILY_EXECUTION_ORDER:
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

    @staticmethod
    def _has_orientation_component(spec: CandidateSpec) -> bool:
        return any(abs(v) > 1.0e-6 for v in (spec.pitch, spec.yaw, spec.roll))

    @staticmethod
    def _has_translation_component(spec: CandidateSpec) -> bool:
        return any(abs(v) > 1.0e-6 for v in (spec.base_x, spec.base_y, spec.base_z))

    @staticmethod
    def _is_pure_orientation(spec: CandidateSpec) -> bool:
        """True when the candidate has orientation but NO translation offset."""
        return (
            SampleManager._has_orientation_component(spec)
            and not SampleManager._has_translation_component(spec)
        )

    def is_coupled_shell_record(self, record: AcceptedSampleRecord) -> bool:
        return (
            record.family == CandidateFamily.SPHERE_SHELL
            and self._has_translation_component(record.spec)
            and self._has_orientation_component(record.spec)
        )

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

    def nominal_orientation_diversity_status(self, sample_pose, observability_axis: str) -> Tuple[bool, str]:
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
        obs_axis = getattr(spec, "observability_axis", "none")
        if getattr(spec, "dedup_protected", False) and obs_axis != "none" and self._is_pure_orientation(spec):
            return self.nominal_orientation_diversity_status(sample_pose, obs_axis)
        return self.nominal_diversity_status(sample_pose)

    def is_orientation_diverse_transform(self, sample_pose, observability_axis: str) -> Tuple[bool, str]:
        """Diversity check for dedup_protected orientation candidates.

        Only compares the actual rotation delta against accepted samples that
        share the same observability_axis.  Translation delta is NOT checked,
        so pure-rotation candidates (e.g. sphere_roll_coverage ±18° at the
        original_place position) are never rejected because of dt≈0.
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

    def solver_subset_keep_sets(self, min_keep: int, max_keep: int) -> List[Tuple[int, ...]]:
        records = self._accepted_samples
        if not records:
            return []

        mandatory = [idx for idx, rec in enumerate(records) if not rec.removable]
        optional = [idx for idx, rec in enumerate(records) if rec.removable]
        min_keep = max(len(mandatory), int(min_keep))
        max_keep = min(len(records), max(int(max_keep), min_keep))

        if len(mandatory) > max_keep:
            return [tuple(sorted(mandatory))]

        def _best_by(items, key_fn):
            return sorted(items, key=key_fn, reverse=True)

        def _abs_component(spec: CandidateSpec, axis: str) -> float:
            return abs(getattr(spec, axis))

        coupled_shell = [
            idx for idx in optional
            if self.is_coupled_shell_record(records[idx])
        ]
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
        priority_sequences = [
            coupled_shell + shell_x_pos[:1] + shell_x_neg[:1] + shell_y_pos[:1] + shell_y_neg[:1] + shell_z[:1] + roll_cov,
            coupled_shell + shell_z[:2] + shell_x_pos[:1] + shell_x_neg[:1] + shell_y_pos[:1] + shell_y_neg[:1] + roll_cov,
            coupled_shell + roll_cov + shell_x_pos[:1] + shell_x_neg[:1] + shell_y_pos[:1] + shell_y_neg[:1] + shell_z[:1],
            coupled_shell + shell_x_pos + shell_x_neg + shell_y_pos + shell_y_neg + shell_z + roll_cov,
        ]

        keep_sets: List[Tuple[int, ...]] = []
        for sequence in priority_sequences:
            keep = list(mandatory)
            for idx in sequence:
                if idx in keep:
                    continue
                if len(keep) >= max_keep:
                    break
                keep.append(idx)
                if len(keep) >= min_keep and tuple(sorted(keep)) not in keep_sets:
                    keep_sets.append(tuple(sorted(keep)))
            if tuple(sorted(keep)) not in keep_sets:
                keep_sets.append(tuple(sorted(keep)))

        full_set = tuple(range(len(records)))
        if full_set not in keep_sets:
            keep_sets.append(full_set)
        return keep_sets

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

    def gate_deficits(self) -> dict:
        return self.governor.gate_deficits(self._accepted_samples, self._reference_rotation)
