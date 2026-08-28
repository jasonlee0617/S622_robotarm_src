"""Deterministic benchmark goal selection and result-file helpers."""

import csv
import math
import statistics

import numpy as np


def obstacle_attr(obstacle, key, default=None):
    return obstacle.get(key, default) if isinstance(obstacle, dict) else getattr(obstacle, key, default)


def obstacle_center(obstacle):
    return tuple(float(value) for value in obstacle_attr(obstacle, "position", (0.0, 0.0, 0.0)))


def obstacle_half_extents(obstacle):
    shape = str(obstacle_attr(obstacle, "shape", "box")).lower()
    if shape == "box":
        return tuple(float(value) * 0.5 for value in obstacle_attr(obstacle, "size", (0.1, 0.1, 0.1)))
    radius = float(obstacle_attr(obstacle, "radius", 0.05))
    if shape == "sphere":
        return (radius, radius, radius)
    height = obstacle_attr(obstacle, "height", None)
    if height is None:
        raise ValueError(f"cylinder obstacle '{obstacle_attr(obstacle, 'name', '')}' missing height")
    return (radius, radius, 0.5 * float(height))


def stratified_candidates(minimum, maximum, count, seed):
    """Return a deterministic, uniformly covered 3-D candidate pool."""
    side = math.ceil(count ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    extent = np.asarray(maximum, dtype=float) - np.asarray(minimum, dtype=float)
    points = []
    for index in range(side ** 3):
        cell = np.asarray((index % side, (index // side) % side, index // (side * side)), dtype=float)
        point = np.asarray(minimum, dtype=float) + (cell + rng.random(3)) / side * extent
        points.append(tuple(float(value) for value in point))
    return points[:count]


def select_farthest_goals(minimum, maximum, count, candidate_count, seed, minimum_separation, validate):
    """Validate a fixed candidate pool then greedily maximize selected spacing."""
    reachable, rejected = [], {"geometry": 0, "ik": 0, "state": 0}
    for point in stratified_candidates(minimum, maximum, candidate_count, seed):
        ok, reason = validate(point)
        if ok:
            reachable.append(point)
        else:
            rejected[reason if reason in rejected else "state"] += 1

    selected, remaining, separation_rejected = [], list(reachable), 0
    while remaining and len(selected) < count:
        if not selected:
            choice_index = 0
        else:
            distances = [min(float(np.linalg.norm(np.asarray(point) - np.asarray(other))) for other in selected)
                         for point in remaining]
            choice_index = max(range(len(remaining)), key=lambda index: (distances[index], -index))
            if distances[choice_index] < minimum_separation:
                separation_rejected += len(remaining)
                break
        selected.append(remaining.pop(choice_index))

    if len(selected) != count:
        raise ValueError(
            "benchmark candidate pool insufficient: "
            f"selected={len(selected)}/{count}, geometry_rejected={rejected['geometry']}, "
            f"ik_rejected={rejected['ik']}, state_invalid={rejected['state']}, "
            f"separation_rejected={separation_rejected}"
        )
    return selected, {**rejected, "separation": separation_rejected, "reachable": len(reachable)}


RESULT_FIELDS = (
    "run_index", "run_mode", "planner_id", "planner_random_seed", "plan_success", "success",
    "failure_phase", "error_code", "goal_pose", "core_planning_time_s",
    "optimized_joint_path_length_rad", "execution_success", "return_home_success",
)


def write_results(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows, expected, run_mode, status="completed", reason=""):
    successful = [row for row in rows if row["success"] == "true"]
    planned = [row for row in rows if row["plan_success"] == "true"]
    values = lambda key: [float(row[key]) for row in planned if float(row[key]) > 0.0]
    percentile = lambda data: sorted(data)[max(0, math.ceil(len(data) * .95) - 1)] if data else 0.0
    phases = {}
    for row in rows:
        if row["failure_phase"] != "none":
            phases[row["failure_phase"]] = phases.get(row["failure_phase"], 0) + 1
    lines = [
        "# Planning benchmark summary", "", f"status: {status}", f"reason: {reason or 'none'}",
        f"run_mode: {run_mode}",
        f"expected_runs: {expected}", f"actual_runs: {len(rows)}",
        f"planning_success_rate: {len(planned)}/{len(rows)}",
        f"failure_phases: {phases or 'none'}",
    ]
    if run_mode == "benchmark_execution":
        lines.insert(7, f"closed_loop_success_rate: {len(successful)}/{len(rows)}")
    for name, data in (("core_planning_time_s", values("core_planning_time_s")),
                       ("path_length_rad", values("optimized_joint_path_length_rad"))):
        lines.append(f"{name}: mean={statistics.mean(data) if data else 0.0:.6f}, "
                     f"median={statistics.median(data) if data else 0.0:.6f}, p95={percentile(data):.6f}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
