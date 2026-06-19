"""SampleSetGovernor: unified coverage + observability + subset governance."""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .sample_types import (
    AcceptedSampleRecord,
    CandidateFamily,
)


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
        min_sphere_shell_samples: int,
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
        self.min_sphere_shell_samples = int(min_sphere_shell_samples)
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

    def coverage_status(
        self,
        records: Sequence[AcceptedSampleRecord],
        *,
        min_count: Optional[int] = None,
    ) -> Tuple[bool, str]:
        m = self.coverage_metrics(records)
        if m is None:
            return False, "no accepted samples"
        required_count = self.min_successful_samples if min_count is None else int(min_count)
        count_ok = m["count"] >= required_count
        xy_ok = m["xy_span"] >= self.min_coverage_xy_span_m
        z_ok = m["z_span"] >= self.min_coverage_z_span_m
        rot_ok = m["max_rot_delta_deg"] >= self.min_coverage_rotation_span_deg
        ok = count_ok and xy_ok and z_ok and rot_ok
        note = (
            f"count {m['count']}/{required_count} {'PASS' if count_ok else 'FAIL'}, "
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
            "shell": obs_m["sphere_shell_count"] < self.min_sphere_shell_samples,
        }
