"""Local direct-PnP hand-eye solve, fixed-marker refinement, and persistence."""

from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def _cv2():
    import cv2
    return cv2


def _se3_increment(values):
    values = np.asarray(values, dtype=float).reshape(6)
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def _rotation_vector(matrix):
    return R.from_matrix(matrix).as_rotvec()


def _marker_metrics(records, ee_T_camera):
    implied = [
        record.robot_pose.matrix() @ ee_T_camera.matrix() @ record.tracking_pose.matrix()
        for record in records
    ]
    positions = np.asarray([matrix[:3, 3] for matrix in implied], dtype=float)
    marker_position = positions.mean(axis=0)
    rotations = R.from_matrix(np.asarray([matrix[:3, :3] for matrix in implied]))
    marker_rotation = rotations.mean()
    translation_residuals = np.linalg.norm(positions - marker_position, axis=1)
    rotation_residuals = np.asarray([
        math.degrees((marker_rotation.inv() * rotation).magnitude())
        for rotation in rotations
    ])
    return {
        "position_rms_m": float(np.sqrt(np.mean(translation_residuals ** 2))),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_residuals ** 2))),
        "position_max_m": float(np.max(translation_residuals)),
        "rotation_max_deg": float(np.max(rotation_residuals)),
        "per_sample_position_m": [float(value) for value in translation_residuals],
        "per_sample_rotation_deg": [float(value) for value in rotation_residuals],
    }


def refine_handeye_fixed_marker(records, initial_transform, *, iterations=25):
    """Refine an OpenCV seed by enforcing one fixed base->marker transform."""
    if len(records) < 3:
        raise ValueError("at least three records are required")
    robots = [record.robot_pose.matrix() for record in records]
    trackings = [record.tracking_pose.matrix() for record in records]
    mount = initial_transform.matrix()
    implied = [robot @ mount @ tracking for robot, tracking in zip(robots, trackings)]
    marker = np.eye(4)
    marker[:3, 3] = np.mean([matrix[:3, 3] for matrix in implied], axis=0)
    marker[:3, :3] = R.from_matrix(
        np.asarray([matrix[:3, :3] for matrix in implied])
    ).mean().as_matrix()
    translation_sigma, rotation_sigma = 0.0005, math.radians(0.30)

    def residual(parameters):
        camera = mount @ _se3_increment(parameters[:6])
        fixed_marker = marker @ _se3_increment(parameters[6:])
        values = []
        for robot, tracking in zip(robots, trackings):
            estimate = robot @ camera @ tracking
            values.extend((estimate[:3, 3] - fixed_marker[:3, 3]) / translation_sigma)
            values.extend(_rotation_vector(fixed_marker[:3, :3].T @ estimate[:3, :3]) / rotation_sigma)
        return np.asarray(values, dtype=float)

    parameters = np.zeros(12)
    initial_cost = float(residual(parameters) @ residual(parameters))
    final_cost = initial_cost
    completed = 0
    for iteration in range(int(iterations)):
        current = residual(parameters)
        current_cost = float(current @ current)
        jacobian = np.empty((len(current), len(parameters)))
        for index in range(len(parameters)):
            epsilon = 1.0e-6 if index % 6 < 3 else 1.0e-5
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            jacobian[:, index] = (residual(plus) - residual(minus)) / (2.0 * epsilon)
        step = np.linalg.lstsq(jacobian, -current, rcond=None)[0]
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = parameters + scale * step
            trial_cost = float(residual(trial) @ residual(trial))
            if trial_cost < current_cost:
                parameters, final_cost, completed, accepted = trial, trial_cost, iteration + 1, True
                break
        if not accepted or float(np.linalg.norm(step)) < 1.0e-8:
            break
    return type(initial_transform)(
        rotation=R.from_matrix((mount @ _se3_increment(parameters[:6]))[:3, :3]),
        translation=tuple(float(value) for value in (mount @ _se3_increment(parameters[:6]))[:3, 3]),
    ), {"initial_cost": initial_cost, "final_cost": final_cost, "iterations": completed}


def _algorithm_methods():
    cv2 = _cv2()
    return {
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


def _transform_distance(left, right):
    return (
        float(np.linalg.norm(np.asarray(left.translation) - np.asarray(right.translation))),
        math.degrees((left.rotation.inv() * right.rotation).magnitude()),
    )


def _consensus(results, max_translation_m=0.003, max_rotation_deg=1.0):
    successful = [(name, result) for name, result in results.items() if "transform" in result]
    best = None
    for name, result in successful:
        neighbours, distances = [], []
        for other_name, other_result in successful:
            dt, dr = _transform_distance(result["transform"], other_result["transform"])
            if dt <= max_translation_m and dr <= max_rotation_deg:
                neighbours.append(other_name)
            distances.append((dt / max_translation_m) + (dr / max_rotation_deg))
        score = (-len(neighbours), sum(distances), name)
        if best is None or score < best[0]:
            best = (score, name, neighbours)
    if best is None or len(best[2]) < 2:
        return None, "no two closed-form algorithms agree within 3mm/1deg"
    return best[1], f"consensus={best[2]}"


def local_handeye_solve(session, records=None):
    """Solve all requested OpenCV methods and select a consensus seed."""
    records = list(records if records is not None else session.sample_manager.accepted_samples)
    if len(records) < 3 or any(record.tracking_pose is None for record in records):
        return None, None, {}
    methods = _algorithm_methods()
    robot_rotations = [record.robot_pose.rotation.as_matrix() for record in records]
    robot_translations = [np.asarray(record.robot_pose.translation) for record in records]
    tracking_rotations = [record.tracking_pose.rotation.as_matrix() for record in records]
    tracking_translations = [np.asarray(record.tracking_pose.translation) for record in records]
    results = {}
    for name in session.sampling_cfg.calibration_algorithms:
        method = methods.get(name)
        if method is None:
            results[name] = {"error": "unknown algorithm"}
            continue
        try:
            rotation, translation = _cv2().calibrateHandEye(
                robot_rotations, robot_translations, tracking_rotations, tracking_translations, method=method,
            )
            transform = session.geometry.from_matrix(np.eye(4))
            transform.rotation = R.from_matrix(rotation)
            transform.translation = tuple(float(value) for value in np.asarray(translation).reshape(3))
            metrics = _marker_metrics(records, transform)
            results[name] = {"transform": transform, "translation_norm": float(np.linalg.norm(translation)), **metrics}
        except Exception as exc:
            results[name] = {"error": str(exc)}
    selected_name, note = _consensus(results)
    if selected_name is None:
        session._logger().warn(f"Closed-form solver consensus failed: {note}")
        return None, None, results
    results[selected_name]["consensus_note"] = note
    return results[selected_name]["transform"], selected_name, results


def _select_records_with_local_prune(session, records):
    kept = list(records)
    removed = []
    for _ in range(2):
        transform, algorithm, results = local_handeye_solve(session, kept)
        if transform is None:
            return None, None, None, removed, results
        metrics = _marker_metrics(kept, transform)
        worst = int(np.argmax(np.asarray(metrics["per_sample_position_m"]) + 0.002 * np.asarray(metrics["per_sample_rotation_deg"])))
        if (
            metrics["position_max_m"] <= 0.003
            and metrics["rotation_max_deg"] <= 1.5
        ) or len(kept) - 1 < session.sampling_cfg.solver_subset_min_samples:
            return kept, transform, algorithm, removed, results
        removed.append(worst)
        kept.pop(worst)
    transform, algorithm, results = local_handeye_solve(session, kept)
    return kept, transform, algorithm, removed, results


def _yaml_transform(transform):
    quaternion = transform.rotation.as_quat()
    return {
        "translation": dict(zip(("x", "y", "z"), (float(value) for value in transform.translation))),
        "rotation": dict(zip(("x", "y", "z", "w"), (float(value) for value in quaternion))),
    }


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def _save_outputs(session, transform, records, report):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(session.sampling_cfg.calibration_output_directory)
    prefix = session.sampling_cfg.calibration_file_prefix
    calibration_path = directory / f"{prefix}_{timestamp}.calib"
    samples_path = directory / f"{prefix}_{timestamp}.samples"
    report_path = directory / f"{prefix}_{timestamp}.report.yaml"
    calibration = {
        "parameters": {
            "name": prefix,
            "calibration_type": "eye_in_hand",
            "robot_base_frame": session.frames.base_frame,
            "robot_effector_frame": session.frames.ee_frame,
            "tracking_base_frame": session.frames.tracking_base_frame,
            "tracking_marker_frame": session.frames.tracking_marker_frame,
            "freehand_robot_movement": True,
            "move_group_namespace": session.motion_cfg.move_group_ns_fairino or "/",
            "move_group": session.motion_cfg.move_group_name,
        },
        "transform": _yaml_transform(transform),
    }
    samples = {
        "parameters": calibration["parameters"],
        "samples": [
            {"robot": _yaml_transform(record.robot_pose), "tracking": _yaml_transform(record.tracking_pose)}
            for record in records
        ],
    }
    _write_yaml(calibration_path, calibration)
    _write_yaml(samples_path, samples)
    _write_yaml(report_path, report)
    return calibration_path, samples_path, report_path


def _log_tf_mount_error(session, parent_frame, tracking_frame, estimated_transform):
    truth_transform = session._current_transform(parent_frame, tracking_frame)
    if truth_transform is None:
        session._logger().warn(
            f"Ground-truth comparison skipped: TF {parent_frame} -> {tracking_frame} is unavailable."
        )
        return None
    translation_error, rotation_error = _transform_distance(estimated_transform, truth_transform)
    session._logger().info(
        f"Ground-truth comparison ({parent_frame} -> {tracking_frame}): "
        f"translation_error={translation_error * 1000.0:.2f}mm, rotation_error={rotation_error:.2f}deg"
    )
    return translation_error, rotation_error


def _tf_mount_error(session, parent_frame, tracking_frame, estimated_transform):
    truth_transform = session._current_transform(parent_frame, tracking_frame)
    if truth_transform is None:
        return None
    return _transform_distance(estimated_transform, truth_transform)


def _log_pose(session, parent_frame, child_frame, transform):
    xyz = transform.translation
    rpy = transform.rotation.as_euler("xyz", degrees=True)
    session._logger().info(
        f"{parent_frame} -> {child_frame}: xyz,rx,ry,rz="
        f"({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}, "
        f"{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}) [m, deg]"
    )


def log_saved_calibration(session, calibration, filepath: str):
    """Compatibility reporting helper for manual/easy_handeye2 callers."""
    if filepath:
        try:
            session._logger().info(f"Calibration result file ({filepath}):\n{Path(filepath).read_text(encoding='utf-8')}")
        except OSError as exc:
            session._logger().error(f"Cannot read saved calibration file {filepath}: {exc}")
    parameters = calibration.parameters
    transform = session.geometry.transform_to_matrix(calibration.transform)
    _log_pose(session, parameters.robot_effector_frame, parameters.tracking_base_frame, transform)
    _log_tf_mount_error(session, parameters.robot_effector_frame, parameters.tracking_base_frame, transform)
    tracking_T_camera_link = session._current_transform(parameters.tracking_base_frame, "camera_link")
    if tracking_T_camera_link is not None:
        camera_link = session.geometry.compose(transform, tracking_T_camera_link)
        _log_pose(session, parameters.robot_effector_frame, "camera_link", camera_link)
        _log_tf_mount_error(
            session, parameters.robot_effector_frame, "camera_link",
            camera_link,
        )
    else:
        session._logger().error(
            "Cannot report end-effector -> camera_link: required TF "
            f"{parameters.tracking_base_frame} -> camera_link is unavailable."
        )


def finalize_calibration(session, ok_count: int):
    """Run local consensus/refinement and save timestamped calibration artifacts."""
    if ok_count < session.sampling_cfg.min_successful_samples:
        session._logger().warn(f"Skip calibration: only {ok_count} good direct samples.")
        return
    gates_ok, gate_note, _coverage, _observability = session.sample_manager.dual_gate_status()
    if not gates_ok:
        session._logger().error(f"Skip calibration: sample gates failed: {gate_note}")
        return
    raw_records = list(session.sample_manager.accepted_samples)
    records, closed_form, algorithm, removed, results = _select_records_with_local_prune(session, raw_records)
    if records is None or closed_form is None:
        session._logger().error("Skip calibration: closed-form solver consensus failed.")
        return
    refined, refinement = refine_handeye_fixed_marker(records, closed_form)
    metrics = _marker_metrics(records, refined)
    if metrics["position_rms_m"] > 0.001 or metrics["rotation_rms_deg"] > 0.70:
        session._logger().error(
            "Skip save: fixed-marker residual gate failed: "
            f"position_rms={metrics['position_rms_m'] * 1000.0:.2f}mm, "
            f"rotation_rms={metrics['rotation_rms_deg']:.2f}deg"
        )
        return
    truth = _tf_mount_error(session, session.frames.ee_frame, session.frames.tracking_base_frame, refined)
    if (
        truth is not None
        and session.sampling_cfg.validate_calibration_against_tf_mount
        and session.sampling_cfg.calibration_tf_mount_check_hard_gate
        and (
            truth[0] > session.sampling_cfg.max_calibration_tf_translation_error_m
            or truth[1] > session.sampling_cfg.max_calibration_tf_rotation_error_deg
        )
    ):
        session._logger().error(
            "Skip save: simulation ground-truth hard gate failed: "
            f"translation_error={truth[0] * 1000.0:.2f}mm, rotation_error={truth[1]:.2f}deg"
        )
        return
    report = {
        "measurement": "direct_ippe_pnp_exact_robot_tf",
        "camera_profile_source": session.sampling_cfg.camera_profile_source or "nominal_fallback",
        "selected_algorithm": algorithm,
        "raw_sample_count": len(raw_records),
        "solver_sample_count": len(records),
        "local_pruned_indices": removed,
        "fixed_marker_refinement": refinement,
        "fixed_marker_residual": metrics,
        "closed_form": {
            name: {key: value for key, value in result.items() if key != "transform"}
            for name, result in results.items()
        },
    }
    calibration_path, samples_path, report_path = _save_outputs(session, refined, raw_records, report)
    session._logger().info(
        f"Calibration saved: {calibration_path}; samples: {samples_path}; report: {report_path}"
    )
    session._logger().info(
        "Fixed-marker residual PASS: "
        f"position_rms={metrics['position_rms_m'] * 1000.0:.2f}mm, "
        f"rotation_rms={metrics['rotation_rms_deg']:.2f}deg"
    )
    _log_tf_mount_error(session, session.frames.ee_frame, session.frames.tracking_base_frame, refined)
