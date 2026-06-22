"""Session finalize: subset selection, local solve, compute/save.

Each function takes `session: CollectorExecutionSession` as first parameter.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from easy_handeye2_msgs.srv import (
    ComputeCalibration,
    SaveCalibration,
    SaveSamples,
    SetAlgorithm,
)

from .sample_types import CandidateFamily


# ------------------------------------------------------------------
# Local handeye solve
# ------------------------------------------------------------------

def _get_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def local_handeye_solve(session, records=None):
    """Run local OpenCV hand-eye calibration with multiple algorithms."""
    cv2 = _get_cv2()
    if cv2 is None:
        session._logger().warn("OpenCV not available for local hand-eye solve.")
        return None, None, {}

    if records is None:
        robot_poses = session.sample_manager.accepted_sample_poses
        tracking_poses = session.sample_manager.accepted_tracking_poses
    else:
        robot_poses = [r.robot_pose for r in records]
        tracking_poses = [r.tracking_pose for r in records if r.tracking_pose is not None]
    if len(robot_poses) < 3 or len(tracking_poses) < 3:
        return None, None, {}

    R_cam = np.stack([p.rotation.as_matrix() for p in tracking_poses])
    t_cam = np.stack([np.array(p.translation) for p in tracking_poses])
    R_base = np.stack([p.rotation.as_matrix() for p in robot_poses])
    t_base = np.stack([np.array(p.translation) for p in robot_poses])

    algorithms = list(session.sampling_cfg.calibration_algorithms)
    if not algorithms:
        algorithms = ["Park", "Horaud", "Tsai-Lenz"]

    cv2_alg_map = {
        "Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    best = None
    best_score = (float("inf"), float("inf"), float("inf"))

    for alg_name in algorithms:
        cv2_alg = cv2_alg_map.get(alg_name)
        if cv2_alg is None:
            session._logger().warn(f"Unknown algorithm: {alg_name}")
            continue
        try:
            R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
                R_base, t_base, R_cam, t_cam, method=cv2_alg,
            )
            ee_T = session.geometry.from_matrix(np.eye(4))
            ee_T.rotation = R.from_matrix(R_ee_cam)
            ee_T.translation = (float(t_ee_cam[0]), float(t_ee_cam[1]), float(t_ee_cam[2]))
            t_norm = float(np.linalg.norm(t_ee_cam))
            residual, _ = session.calibration_validator.calibration_marker_residual(
                ee_T, robot_poses, tracking_poses,
                session.geometry.compose, session.geometry.rotation_delta_deg,
            )
            if residual is None:
                results[alg_name] = {"translation_norm": t_norm, "error": "no residual"}
                continue
            r = {
                "translation_norm": t_norm,
                "span_norm": residual["span_norm"],
                "rmse": residual["rmse"],
                "max_error": residual["max_error"],
            }
            results[alg_name] = r
            score = (r["span_norm"], r["rmse"], r["translation_norm"])
            if score < best_score:
                best_score = score
                best = (ee_T, alg_name)
        except Exception as exc:
            session._logger().warn(f"Local solver {alg_name} failed: {exc}")
            results[alg_name] = {"error": str(exc)}

    if best is None:
        return None, None, results
    return best[0], best[1], results


def solver_result_passes_local_gate(session, result_dict) -> Tuple[bool, str]:
    t_ok = result_dict["translation_norm"] <= session.sampling_cfg.max_calibration_translation_norm_m
    span_ok = result_dict["span_norm"] <= session.sampling_cfg.max_calibration_marker_span_m
    rmse_ok = result_dict["rmse"] <= session.sampling_cfg.max_calibration_marker_span_m
    ok = t_ok and span_ok and rmse_ok
    note = (
        f"translation_norm={result_dict['translation_norm']:.3f}/"
        f"{session.sampling_cfg.max_calibration_translation_norm_m:.3f}m {'PASS' if t_ok else 'FAIL'}, "
        f"span_norm={result_dict['span_norm']:.3f}/"
        f"{session.sampling_cfg.max_calibration_marker_span_m:.3f}m {'PASS' if span_ok else 'FAIL'}, "
        f"rmse={result_dict['rmse']:.3f}/"
        f"{session.sampling_cfg.max_calibration_marker_span_m:.3f}m {'PASS' if rmse_ok else 'FAIL'}"
    )
    return ok, note


# ------------------------------------------------------------------
# Solver subset selection
# ------------------------------------------------------------------

def solver_subset_gate_status(session, records):
    cov_ok, cov_note = session.sample_manager.governor.coverage_status(
        records,
        min_count=session.sampling_cfg.solver_subset_min_samples,
    )
    obs_ok, obs_note = session.sample_manager.governor.observability_status(
        records, session.sample_manager.reference_rotation,
    )
    shell_count = sum(
        1 for record in records
        if record.family == CandidateFamily.SPHERE_SHELL
    )
    return cov_ok, cov_note, obs_ok, obs_note, shell_count


def influence_pruned_solver_keep_sets(session) -> List[Tuple[int, ...]]:
    """逐个删除样本、用内部 residual 变化定位高影响样本，生成删除组合。

    不依赖 TF/xacro 真值；只用 local solver 的 rmse/span_norm。
    """
    records = session.sample_manager.accepted_samples
    n = len(records)
    if n <= session.sampling_cfg.solver_subset_min_samples:
        return []

    base_indices = tuple(range(n))
    influence_candidates: List[Tuple[float, float, int]] = []

    for remove_idx in range(n):
        keep = tuple(i for i in base_indices if i != remove_idx)
        subset_records = session.sample_manager.subset_records(keep)
        local_ee_T, local_alg, local_results = local_handeye_solve(session, subset_records)
        if local_ee_T is None or local_alg is None:
            continue
        result = local_results.get(local_alg, {})
        if "error" in result:
            continue
        influence_candidates.append((
            float(result.get("rmse", float("inf"))),
            float(result.get("span_norm", float("inf"))),
            remove_idx,
        ))

    if not influence_candidates:
        return []

    influence_candidates.sort()
    structural_remove_pool = [
        idx for idx, record in enumerate(records)
        if session.sample_manager.is_yaw_coupled_shell_record(record)
    ]
    influence_pool = list(dict.fromkeys(
        structural_remove_pool + [idx for _, _, idx in influence_candidates[:8]]
    ))

    keep_sets: List[Tuple[int, ...]] = []
    max_remove = min(6, n - session.sampling_cfg.solver_subset_min_samples)

    height_pos = [
        idx for idx, record in enumerate(records)
        if record.family == CandidateFamily.SPHERE_HEIGHT and record.spec.base_z > 1.0e-6
    ]
    height_neg = [
        idx for idx, record in enumerate(records)
        if record.family == CandidateFamily.SPHERE_HEIGHT and record.spec.base_z < -1.0e-6
    ]

    def _try_add_keep(remove_indices: Tuple[int, ...]) -> None:
        keep = tuple(i for i in base_indices if i not in set(remove_indices))
        if len(keep) < session.sampling_cfg.solver_subset_min_samples:
            return
        if len(keep) > session.sampling_cfg.solver_subset_max_samples:
            return
        subset_records = session.sample_manager.subset_records(keep)
        cov_ok, _, obs_ok, _, shell_count = solver_subset_gate_status(session, subset_records)
        if not cov_ok or not obs_ok or shell_count < session.sampling_cfg.min_sphere_shell_samples:
            return
        keep_tuple = tuple(sorted(keep))
        if keep_tuple not in keep_sets:
            keep_sets.append(keep_tuple)

    for remove_count in range(1, max_remove + 1):
        for remove_combo in itertools.combinations(influence_pool, remove_count):
            _try_add_keep(tuple(remove_combo))

    if len(height_pos) > 1 and len(height_neg) > 1:
        for height_pair in itertools.product(height_pos, height_neg):
            remaining_pool = [idx for idx in influence_pool if idx not in set(height_pair)]
            max_extra = max_remove - len(height_pair)
            for extra_count in range(0, max_extra + 1):
                for extra_combo in itertools.combinations(remaining_pool, extra_count):
                    _try_add_keep(tuple(height_pair) + tuple(extra_combo))

    return keep_sets


def select_solver_subset(session):
    keep_sets = session.sample_manager.solver_subset_keep_sets(
        session.sampling_cfg.solver_subset_min_samples,
        session.sampling_cfg.solver_subset_max_samples,
    )
    keep_sets.extend(influence_pruned_solver_keep_sets(session))
    if not keep_sets:
        return None, "no solver subset candidates", None, None

    best = None
    best_fail = None
    seen = set()
    for keep in keep_sets:
        keep = tuple(sorted(int(idx) for idx in keep))
        if keep in seen:
            continue
        seen.add(keep)
        records = session.sample_manager.subset_records(keep)
        cov_ok, cov_note, obs_ok, obs_note, shell_count = solver_subset_gate_status(session, records)
        if not cov_ok or not obs_ok or shell_count < session.sampling_cfg.min_sphere_shell_samples:
            note = (
                f"keep={list(keep)} gate_fail: coverage={'PASS' if cov_ok else 'FAIL'} ({cov_note}); "
                f"observability={'PASS' if obs_ok else 'FAIL'} ({obs_note}); "
                f"sphere_shell={shell_count}/{session.sampling_cfg.min_sphere_shell_samples}"
            )
            best_fail = best_fail or note
            continue

        local_ee_T, local_alg, local_results = local_handeye_solve(session, records)
        if local_ee_T is None or local_alg is None or local_alg not in local_results:
            note = f"keep={list(keep)} local_solver_fail"
            best_fail = best_fail or note
            continue
        winner = local_results[local_alg]
        if "error" in winner:
            note = f"keep={list(keep)} local_solver_error={winner['error']}"
            best_fail = best_fail or note
            continue
        local_ok, local_note = solver_result_passes_local_gate(session, winner)
        quality_metrics = session.sample_manager.subset_quality_metrics(records)
        if quality_metrics is None:
            note = f"keep={list(keep)} subset_quality_unavailable"
            best_fail = best_fail or note
            continue

        if len(session.sample_manager.accepted_samples) >= 14:
            if quality_metrics["height_positive_count"] == 0 or quality_metrics["height_negative_count"] == 0:
                note = (
                    f"keep={list(keep)} height_sign_imbalance: "
                    f"+z={quality_metrics['height_positive_count']} "
                    f"-z={quality_metrics['height_negative_count']}"
                )
                best_fail = best_fail or note
                continue

        score = (
            0 if local_ok else 1,
            quality_metrics["height_sign_imbalance"],
            quality_metrics["yaw_coupled_shell_count"],
            winner["span_norm"],
            winner["rmse"],
            quality_metrics["non_strict_recenter_count"],
            quality_metrics["max_camera_model_error_px"],
            quality_metrics["max_center_error_px"],
            quality_metrics["max_center_std_px"],
            -quality_metrics["min_marker_side_px"],
            -quality_metrics["min_margin_px"],
            len(records),
        )
        note = (
            f"keep={list(keep)} alg={local_alg} samples={len(records)} "
            f"{local_note}; "
            f"height_imbalance={quality_metrics['height_sign_imbalance']} "
            f"(+z={quality_metrics['height_positive_count']} "
            f"-z={quality_metrics['height_negative_count']}) "
            f"yaw_coupled={quality_metrics['yaw_coupled_shell_count']} "
            f"non_strict_recenter={quality_metrics['non_strict_recenter_count']} "
            f"max_model_err={quality_metrics['max_camera_model_error_px']:.1f}px "
            f"max_center_err={quality_metrics['max_center_error_px']:.1f}px "
            f"max_center_std={quality_metrics['max_center_std_px']:.2f}px "
            f"min_side={quality_metrics['min_marker_side_px']:.1f}px "
            f"min_margin={quality_metrics['min_margin_px']:.1f}px"
        )
        if best is None or score < best[0]:
            best = (score, keep, local_alg, winner, note, local_ok)
        if local_ok:
            session._logger().info(f"Solver subset candidate PASS: {note}")
        else:
            session._logger().info(f"Solver subset candidate FAIL: {note}")

    if best is None:
        return None, best_fail or "no solver subset candidates survived local solve", None, None
    if not best[5]:
        return None, f"best local subset still failed: {best[4]}", None, None
    return best[1], best[4], best[2], best[3]


# ------------------------------------------------------------------
# Compute calibration
# ------------------------------------------------------------------

def compute_calibration_result(session):
    from .session_checks import call_empty_service
    result, error = call_empty_service(
        session, session.node.compute_cli, ComputeCalibration.Request(),
        session.frames.compute_calibration_service,
        timeout_sec=session.sampling_cfg.compute_calibration_timeout,
    )
    if result is None or not getattr(result, "valid", False):
        return None, f"ComputeCalibration failed: {error or result}"
    return result, ""


def save_current_sample_set(session, context: str = "Sample set"):
    from .session_checks import call_empty_service
    if not session.sampling_cfg.auto_save_samples:
        return
    result, error = call_empty_service(
        session, session.node.save_samples_cli, SaveSamples.Request(),
        session.frames.save_samples_service,
        timeout_sec=session.sampling_cfg.save_samples_timeout,
    )
    if result is None or not getattr(result, "success", False):
        session._logger().warn(f"SaveSamples failed after {context}: {error or result}")
    else:
        session._logger().info(f"{context} saved by easy_handeye2.")


def _log_pose(session, parent_frame: str, child_frame: str, transform):
    xyz = transform.translation
    rpy = transform.rotation.as_euler("xyz", degrees=True)
    session._logger().info(
        f"{parent_frame} -> {child_frame}: xyz,rx,ry,rz="
        f"({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}, "
        f"{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}) [m, deg]"
    )


def _log_tf_mount_error(session, parent_frame: str, tracking_frame: str, estimated_transform):
    truth_transform = session._current_transform(parent_frame, tracking_frame)
    if truth_transform is None:
        session._logger().warn(
            f"Ground-truth comparison skipped: TF {parent_frame} -> {tracking_frame} is unavailable."
        )
        return
    translation_error_mm = 1000.0 * float(np.linalg.norm(
        np.subtract(estimated_transform.translation, truth_transform.translation)
    ))
    rotation_error_deg = session.geometry.rotation_delta_deg(
        truth_transform.rotation, estimated_transform.rotation,
    )
    session._logger().info(
        f"Ground-truth comparison ({parent_frame} -> {tracking_frame}): "
        f"translation_error={translation_error_mm:.2f}mm, "
        f"rotation_error={rotation_error_deg:.2f}deg"
    )


def log_saved_calibration(session, calibration, filepath: str):
    logger = session._logger()
    if filepath:
        try:
            logger.info(f"Calibration result file ({filepath}):\n{Path(filepath).read_text(encoding='utf-8')}")
        except OSError as exc:
            logger.error(f"Cannot read saved calibration file {filepath}: {exc}")
    else:
        logger.error("SaveCalibration succeeded but returned no calibration filepath.")

    parameters = calibration.parameters
    parent_frame = parameters.robot_effector_frame
    tracking_frame = parameters.tracking_base_frame
    parent_T_tracking = session.geometry.transform_to_matrix(calibration.transform)
    _log_pose(session, parent_frame, tracking_frame, parent_T_tracking)
    _log_tf_mount_error(session, parent_frame, tracking_frame, parent_T_tracking)

    tracking_T_camera_link = session._current_transform(tracking_frame, "camera_link")
    if tracking_T_camera_link is None:
        logger.error(
            f"Cannot report {parent_frame} -> camera_link: "
            f"required TF {tracking_frame} -> camera_link is unavailable."
        )
        return
    parent_T_camera_link = session.geometry.compose(parent_T_tracking, tracking_T_camera_link)
    _log_pose(
        session,
        parent_frame,
        "camera_link",
        parent_T_camera_link,
    )
    _log_tf_mount_error(session, parent_frame, "camera_link", parent_T_camera_link)


# ------------------------------------------------------------------
# Finalize calibration
# ------------------------------------------------------------------

def finalize_calibration(session, ok_count: int):
    from .session_checks import apply_remote_removals, call_empty_service
    if ok_count < session.sampling_cfg.min_successful_samples:
        session._logger().warn(f"Skip compute/save: only {ok_count} good samples.")
        return

    ok, note, _, _ = session.sample_manager.dual_gate_status()
    if not ok:
        session._logger().error(f"Skip compute/save calibration: dual gate FAIL: {note}")
        return

    session._logger().info(f"Sample gates passed: {note}")

    sphere_shell_count = sum(
        1 for r in session.sample_manager.accepted_samples
        if r.family == CandidateFamily.SPHERE_SHELL
    )
    if sphere_shell_count < session.sampling_cfg.min_sphere_shell_samples:
        session._logger().error(
            f"Skip compute/save: sphere_shell count {sphere_shell_count} < "
            f"{session.sampling_cfg.min_sphere_shell_samples}. "
            "Insufficient compound multi-axis samples for hand-eye conditioning."
        )
        return
    session._logger().info(
        f"Sphere shell gate: {sphere_shell_count}/{session.sampling_cfg.min_sphere_shell_samples} samples"
    )

    save_current_sample_set(session)

    keep_indices, subset_note, local_alg, local_result = select_solver_subset(session)
    if keep_indices is None:
        session._logger().error(f"Skip compute/save: solver subset selection failed: {subset_note}")
        return
    remove_indices = [
        idx for idx in range(len(session.sample_manager.accepted_samples))
        if idx not in set(keep_indices)
    ]
    if remove_indices:
        applied_ok, applied_note = apply_remote_removals(session, remove_indices)
        if not applied_ok:
            session._logger().error(f"Skip compute/save: cannot apply solver subset: {applied_note}")
            return
        session._logger().info(f"Applied solver subset removals: {applied_note}")
        save_current_sample_set(session, context="Solver subset")
    session._logger().info(f"Solver subset selected: {subset_note}")
    if local_alg is not None and local_result is not None:
        session._logger().info(
            f"Local solver subset winner: {local_alg} "
            f"tnorm={local_result['translation_norm']:.3f}m "
            f"span={local_result['span_norm']:.3f}m "
            f"rmse={local_result['rmse']:.3f}m"
        )
        if session.node.set_algorithm_cli.wait_for_service(timeout_sec=2.0):
            alg_req = SetAlgorithm.Request()
            alg_req.new_algorithm = f"OpenCV/{local_alg}"
            session.node.set_algorithm_cli.call_async(alg_req)
            session._logger().info(f"Switched easy_handeye2 to OpenCV/{local_alg}")

    if not session.sampling_cfg.auto_compute:
        session._logger().info("auto_compute=false: use easy_handeye2 GUI or service to compute.")
        return

    compute_result, error = compute_calibration_result(session)
    if compute_result is None:
        session._logger().error(error)
        return
    session._logger().info("Calibration computed successfully.")

    sanity_ok, sanity_note = session.calibration_validator.calibration_sanity_status(
        compute_result.calibration,
        accepted_sample_poses=session.sample_manager.accepted_sample_poses,
        accepted_tracking_poses=session.sample_manager.accepted_tracking_poses,
        transform_to_matrix=session.geometry.transform_to_matrix,
        lookup_tf=session._lookup_tf, compose=session.geometry.compose,
        rotation_delta_deg=session.geometry.rotation_delta_deg,
        ee_frame=session.frames.ee_frame,
        tracking_base_frame=session.frames.tracking_base_frame,
    )
    if not sanity_ok:
        session._logger().error(
            "Calibration sanity check FAIL after solver-subset selection. "
            "Calibration will NOT be saved. "
            f"Last status: {sanity_note}"
        )
        return

    session._logger().info(f"Calibration sanity check PASS: {sanity_note}")

    if not session.sampling_cfg.auto_save_calibration:
        session._logger().info("auto_save_calibration=false: computed result was not saved.")
        return

    save_result, error = call_empty_service(
        session, session.node.save_calibration_cli, SaveCalibration.Request(),
        session.frames.save_calibration_service,
        timeout_sec=session.sampling_cfg.save_calibration_timeout,
    )
    if save_result is None or not getattr(save_result, "success", False):
        session._logger().error(f"SaveCalibration failed: {error or save_result}")
        return
    filepath = getattr(getattr(save_result, "filepath", None), "data", "")
    session._logger().info(f"Calibration saved: {filepath or '(easy_handeye2 default path)'}")
    log_saved_calibration(session, compute_result.calibration, filepath)
