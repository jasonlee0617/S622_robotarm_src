from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


class CandidateFamily:
    CORE = "core_translation_roll"
    COVERAGE = "coverage_expansion"
    RISKY = "provisional_risky"


@dataclass(frozen=True)
class CandidateSpec:
    source: str
    base_x: float
    base_y: float
    base_z: float
    tilt_x: float
    tilt_y: float
    roll: float
    family: str
    removable: bool

    def key(self):
        return (
            self.source,
            round(self.base_x, 4),
            round(self.base_y, 4),
            round(self.base_z, 4),
            round(self.tilt_x, 2),
            round(self.tilt_y, 2),
            round(self.roll, 2),
            self.family,
            self.removable,
        )


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


class SampleManager:
    def __init__(
        self,
        *,
        min_successful_samples: int,
        sample_min_translation_delta: float,
        sample_min_rotation_delta_deg: float,
        nominal_translation_delta_scale: float,
        nominal_rotation_delta_scale: float,
        min_coverage_xy_span_m: float,
        min_coverage_z_span_m: float,
        min_coverage_rotation_span_deg: float,
        sampling_base_x_offsets_m: Sequence[float],
        sampling_base_y_offsets_m: Sequence[float],
        sampling_base_z_offsets_m: Sequence[float],
        sampling_tilt_x_offsets_deg: Sequence[float],
        sampling_tilt_y_offsets_deg: Sequence[float],
        sampling_roll_offsets_deg: Sequence[float],
        rotation_delta_deg: Callable,
    ):
        self.min_successful_samples = int(min_successful_samples)
        self.sample_min_translation_delta = float(sample_min_translation_delta)
        self.sample_min_rotation_delta_deg = float(sample_min_rotation_delta_deg)
        self.nominal_translation_delta_scale = float(nominal_translation_delta_scale)
        self.nominal_rotation_delta_scale = float(nominal_rotation_delta_scale)
        self.min_coverage_xy_span_m = float(min_coverage_xy_span_m)
        self.min_coverage_z_span_m = float(min_coverage_z_span_m)
        self.min_coverage_rotation_span_deg = float(min_coverage_rotation_span_deg)
        self.sampling_base_x_offsets_m = [float(v) for v in sampling_base_x_offsets_m]
        self.sampling_base_y_offsets_m = [float(v) for v in sampling_base_y_offsets_m]
        self.sampling_base_z_offsets_m = [float(v) for v in sampling_base_z_offsets_m]
        self.sampling_tilt_x_offsets_deg = [float(v) for v in sampling_tilt_x_offsets_deg]
        self.sampling_tilt_y_offsets_deg = [float(v) for v in sampling_tilt_y_offsets_deg]
        self.sampling_roll_offsets_deg = [float(v) for v in sampling_roll_offsets_deg]
        self._rotation_delta_deg = rotation_delta_deg
        self._accepted_samples: List[AcceptedSampleRecord] = []

    @property
    def accepted_samples(self) -> List[AcceptedSampleRecord]:
        return self._accepted_samples

    @property
    def accepted_sample_poses(self):
        return [record.robot_pose for record in self._accepted_samples]

    @property
    def accepted_tracking_poses(self):
        return [
            record.tracking_pose
            for record in self._accepted_samples
            if record.tracking_pose is not None
        ]

    def reset(self):
        self._accepted_samples.clear()

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
        return [record for idx, record in enumerate(self._accepted_samples) if idx in keep]

    def removable_indices(self) -> List[int]:
        return [
            idx
            for idx, record in enumerate(self._accepted_samples)
            if record.removable
        ]

    def _make_spec(
        self,
        source: str,
        *,
        base_x: float = 0.0,
        base_y: float = 0.0,
        base_z: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        roll: float = 0.0,
        family: str,
        removable: bool,
    ) -> CandidateSpec:
        return CandidateSpec(
            source=source,
            base_x=float(base_x),
            base_y=float(base_y),
            base_z=float(base_z),
            tilt_x=float(tilt_x),
            tilt_y=float(tilt_y),
            roll=float(roll),
            family=family,
            removable=bool(removable),
        )

    def build_candidate_specs(self) -> List[CandidateSpec]:
        specs: List[CandidateSpec] = [
            self._make_spec("center", family=CandidateFamily.CORE, removable=False)
        ]

        for value in self.sampling_roll_offsets_deg:
            if abs(value) <= 1.0e-9:
                continue
            family = CandidateFamily.CORE if abs(value) <= 12.0 else CandidateFamily.COVERAGE
            removable = family != CandidateFamily.CORE
            specs.append(
                self._make_spec(
                    f"roll {value:+.1f}deg",
                    roll=value,
                    family=family,
                    removable=removable,
                )
            )

        for value in self.sampling_tilt_x_offsets_deg:
            if abs(value) <= 1.0e-9:
                continue
            specs.append(
                self._make_spec(
                    f"tilt_x {value:+.1f}deg",
                    tilt_x=value,
                    family=CandidateFamily.RISKY,
                    removable=True,
                )
            )

        for value in self.sampling_tilt_y_offsets_deg:
            if abs(value) <= 1.0e-9:
                continue
            specs.append(
                self._make_spec(
                    f"tilt_y {value:+.1f}deg",
                    tilt_y=value,
                    family=CandidateFamily.RISKY,
                    removable=True,
                )
            )

        for value in self.sampling_base_z_offsets_m:
            if abs(value) <= 1.0e-9:
                continue
            family = CandidateFamily.CORE if abs(value) <= 0.030 else CandidateFamily.COVERAGE
            removable = family != CandidateFamily.CORE
            specs.append(
                self._make_spec(
                    f"base_z {value:+.3f}m",
                    base_z=value,
                    family=family,
                    removable=removable,
                )
            )

        for axis_name, values in (
            ("base_x", self.sampling_base_x_offsets_m),
            ("base_y", self.sampling_base_y_offsets_m),
        ):
            for value in values:
                if abs(value) <= 1.0e-9:
                    continue
                if value < 0.0:
                    family = CandidateFamily.RISKY
                    removable = True
                elif abs(value) <= 0.020:
                    family = CandidateFamily.CORE
                    removable = False
                else:
                    family = CandidateFamily.COVERAGE
                    removable = True
                specs.append(
                    self._make_spec(
                        f"{axis_name} {value:+.3f}m",
                        **{axis_name: value},
                        family=family,
                        removable=removable,
                    )
                )
        return specs

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
            if (
                trans_delta < translation_threshold
                and rot_delta_deg < rotation_threshold_deg
            ):
                return (
                    False,
                    f"{prefix}too close to accepted sample "
                    f"(dt={trans_delta:.3f}m, dr={rot_delta_deg:.1f}deg)",
                )
        return True, f"{prefix}diverse".strip()

    def is_diverse_transform(self, sample_pose) -> Tuple[bool, str]:
        return self._diversity_status(
            sample_pose,
            translation_threshold=self.sample_min_translation_delta,
            rotation_threshold_deg=self.sample_min_rotation_delta_deg,
        )

    def nominal_diversity_status(self, sample_pose) -> Tuple[bool, str]:
        nominal_translation = max(
            1.0e-6, self.sample_min_translation_delta * self.nominal_translation_delta_scale
        )
        nominal_rotation = max(
            1.0e-6, self.sample_min_rotation_delta_deg * self.nominal_rotation_delta_scale
        )
        ok, note = self._diversity_status(
            sample_pose,
            translation_threshold=nominal_translation,
            rotation_threshold_deg=nominal_rotation,
        )
        if ok:
            return True, note
        return False, f"nominal_too_close: {note}"

    def _coverage_metrics_from_records(self, records: Sequence[AcceptedSampleRecord]):
        if not records:
            return None
        translations = np.array([record.robot_pose.translation for record in records], dtype=float)
        xyz_span = np.ptp(translations, axis=0)
        xy_span = float(np.linalg.norm(xyz_span[:2]))
        max_rot_delta = 0.0
        poses = [record.robot_pose for record in records]
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

    def coverage_metrics(self):
        return self._coverage_metrics_from_records(self._accepted_samples)

    def _coverage_status_from_metrics(self, metrics) -> Tuple[bool, str]:
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

    def coverage_status(self) -> Tuple[bool, str]:
        return self._coverage_status_from_metrics(self.coverage_metrics())

    def coverage_status_for_indices(self, keep_indices: Sequence[int]):
        records = self.subset_records(keep_indices)
        metrics = self._coverage_metrics_from_records(records)
        ok, note = self._coverage_status_from_metrics(metrics)
        return ok, note, metrics

    def coverage_status_after_removal(self, index: int):
        if index < 0 or index >= len(self._accepted_samples):
            return False, "invalid sample index", None
        keep_indices = [idx for idx in range(len(self._accepted_samples)) if idx != index]
        return self.coverage_status_for_indices(keep_indices)
