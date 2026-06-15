from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional, Sequence, Tuple

from sample_manager import CandidateFamily, SampleManager


@dataclass(frozen=True)
class SubsetEvaluation:
    keep_indices: Tuple[int, ...]
    remove_indices: Tuple[int, ...]
    coverage_ok: bool
    coverage_note: str
    residual: Optional[dict]
    score: Tuple[float, float, float, int]


@dataclass(frozen=True)
class SubsetSearchResult:
    full_set: SubsetEvaluation
    best: SubsetEvaluation
    improved: bool
    local_note: str


class SampleSubsetOptimizer:
    def __init__(
        self,
        *,
        sample_manager: SampleManager,
        calibration_validator,
        compose: Callable,
        rotation_delta_deg: Callable,
        max_remove_count: int,
    ):
        self.sample_manager = sample_manager
        self.calibration_validator = calibration_validator
        self.compose = compose
        self.rotation_delta_deg = rotation_delta_deg
        self.max_remove_count = max(0, int(max_remove_count))

    def _subset_evaluation(self, ee_T_cam, keep_indices: Sequence[int]) -> SubsetEvaluation:
        keep_indices = tuple(sorted(int(idx) for idx in keep_indices))
        remove_indices = tuple(
            idx
            for idx in range(len(self.sample_manager.accepted_samples))
            if idx not in keep_indices
        )
        coverage_ok, coverage_note, _ = self.sample_manager.coverage_status_for_indices(keep_indices)
        residual = None
        if coverage_ok:
            records = self.sample_manager.subset_records(keep_indices)
            if any(record.tracking_pose is None for record in records):
                coverage_ok = False
                coverage_note = "subset contains sample(s) without tracking pose"
            else:
                robot_poses = [record.robot_pose for record in records]
                tracking_poses = [record.tracking_pose for record in records]
                residual, _ = self.calibration_validator.calibration_marker_residual(
                    ee_T_cam,
                    robot_poses,
                    tracking_poses,
                    self.compose,
                    self.rotation_delta_deg,
                )
        score = (
            float("inf"),
            float("inf"),
            float("inf"),
            len(remove_indices),
        )
        if residual is not None:
            score = (
                float(residual["span_norm"]),
                float(residual["rmse"]),
                float(residual["max_error"]),
                len(remove_indices),
            )
        return SubsetEvaluation(
            keep_indices=keep_indices,
            remove_indices=remove_indices,
            coverage_ok=coverage_ok,
            coverage_note=coverage_note,
            residual=residual,
            score=score,
        )

    def _removable_indices_by_priority(self):
        risky = []
        secondary = []
        for idx, record in enumerate(self.sample_manager.accepted_samples):
            if not record.removable:
                continue
            if record.family == CandidateFamily.RISKY:
                risky.append(idx)
            else:
                secondary.append(idx)
        return tuple(risky + secondary)

    def _note(self, label: str, evaluation: SubsetEvaluation) -> str:
        if evaluation.residual is None:
            return f"{label}: {evaluation.coverage_note}"
        residual = evaluation.residual
        return (
            f"{label}: remove={list(evaluation.remove_indices)}; "
            f"span_norm={residual['span_norm']:.3f}m rmse={residual['rmse']:.3f}m "
            f"max_error={residual['max_error']:.3f}m; {evaluation.coverage_note}"
        )

    def find_best_subset(self, ee_T_cam) -> SubsetSearchResult:
        all_indices = tuple(range(len(self.sample_manager.accepted_samples)))
        full_set = self._subset_evaluation(ee_T_cam, all_indices)
        best = full_set

        removable_indices = self._removable_indices_by_priority()
        if not removable_indices or self.max_remove_count <= 0:
            return SubsetSearchResult(
                full_set=full_set,
                best=best,
                improved=False,
                local_note=self._note("full_set", full_set),
            )

        greedy = full_set
        removed_prefix = []
        for _ in range(min(self.max_remove_count, len(removable_indices))):
            step_best = None
            for idx in removable_indices:
                if idx in removed_prefix:
                    continue
                trial_remove = tuple(sorted(removed_prefix + [idx]))
                trial_keep = tuple(i for i in all_indices if i not in trial_remove)
                trial = self._subset_evaluation(ee_T_cam, trial_keep)
                if not trial.coverage_ok:
                    continue
                if step_best is None or trial.score < step_best.score:
                    step_best = trial
            if step_best is None or step_best.score >= greedy.score:
                break
            greedy = step_best
            removed_prefix = list(greedy.remove_indices)
            if greedy.score < best.score:
                best = greedy

        combo_limit = min(self.max_remove_count, 3, len(removable_indices))
        for size in range(1, combo_limit + 1):
            for combo in combinations(removable_indices, size):
                trial_keep = tuple(i for i in all_indices if i not in combo)
                trial = self._subset_evaluation(ee_T_cam, trial_keep)
                if not trial.coverage_ok:
                    continue
                if trial.score < best.score:
                    best = trial

        improved = best.score < full_set.score
        if improved:
            note = f"{self._note('full_set', full_set)} -> {self._note('best_subset', best)}"
        else:
            note = self._note("full_set", full_set)
        return SubsetSearchResult(
            full_set=full_set,
            best=best,
            improved=improved,
            local_note=note,
        )
