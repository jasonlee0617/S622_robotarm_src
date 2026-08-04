"""Solve one accepted sample pool and write only calibration plus samples."""

from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def _cv2():
    import cv2
    return cv2


def _se3_increment(values):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(np.asarray(values[3:], dtype=float)).as_matrix()
    matrix[:3, 3] = np.asarray(values[:3], dtype=float)
    return matrix


def _rotation_vector(matrix):
    return R.from_matrix(matrix).as_rotvec()


def _marker_reference(records, transform):
    implied = [record.robot_pose.matrix() @ transform.matrix() @ record.tracking_pose.matrix() for record in records]
    return np.mean([matrix[:3, 3] for matrix in implied], axis=0), R.from_matrix([matrix[:3, :3] for matrix in implied]).mean()


def marker_metrics(records, ee_T_camera):
    position, rotation = _marker_reference(records, ee_T_camera)
    implied = [record.robot_pose.matrix() @ ee_T_camera.matrix() @ record.tracking_pose.matrix() for record in records]
    translation = np.asarray([np.linalg.norm(matrix[:3, 3] - position) for matrix in implied], dtype=float)
    angular = np.asarray([math.degrees((rotation.inv() * R.from_matrix(matrix[:3, :3])).magnitude()) for matrix in implied], dtype=float)
    return {
        "position_rms_m": float(np.sqrt(np.mean(translation ** 2))),
        "rotation_rms_deg": float(np.sqrt(np.mean(angular ** 2))),
        "position_max_m": float(np.max(translation)),
        "rotation_max_deg": float(np.max(angular)),
        "per_sample_position_m": [float(value) for value in translation],
        "per_sample_rotation_deg": [float(value) for value in angular],
    }


def refine_handeye_fixed_marker(records, initial_transform, *, iterations=25, translation_sigma_m=0.0005, rotation_sigma_deg=0.30):
    if len(records) < 3:
        raise ValueError("at least three records are required")
    robots = [record.robot_pose.matrix() for record in records]
    trackings = [record.tracking_pose.matrix() for record in records]
    mount = initial_transform.matrix()
    position, rotation = _marker_reference(records, initial_transform)
    marker = np.eye(4)
    marker[:3, 3], marker[:3, :3] = position, rotation.as_matrix()
    translation_sigma, rotation_sigma = float(translation_sigma_m), math.radians(float(rotation_sigma_deg))

    def residual(values):
        camera, fixed_marker = mount @ _se3_increment(values[:6]), marker @ _se3_increment(values[6:])
        rows = []
        for robot, tracking in zip(robots, trackings):
            estimate = robot @ camera @ tracking
            rows.extend((estimate[:3, 3] - fixed_marker[:3, 3]) / translation_sigma)
            rows.extend(_rotation_vector(fixed_marker[:3, :3].T @ estimate[:3, :3]) / rotation_sigma)
        return np.asarray(rows, dtype=float)

    result = least_squares(residual, np.zeros(12), loss="huber", f_scale=1.0, max_nfev=max(100, int(iterations) * 20))
    matrix = mount @ _se3_increment(result.x[:6])
    transform = type(initial_transform)(R.from_matrix(matrix[:3, :3]), tuple(float(value) for value in matrix[:3, 3]))
    return transform, {"cost": float(residual(result.x) @ residual(result.x)), "iterations": int(result.nfev)}


def _solve_method(session, records, method):
    robot_rotations = [record.robot_pose.rotation.as_matrix() for record in records]
    robot_translations = [np.asarray(record.robot_pose.translation) for record in records]
    tracking_rotations = [record.tracking_pose.rotation.as_matrix() for record in records]
    tracking_translations = [np.asarray(record.tracking_pose.translation) for record in records]
    rotation, translation = _cv2().calibrateHandEye(
        robot_rotations,
        robot_translations,
        tracking_rotations,
        tracking_translations,
        method=method,
    )
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, np.asarray(translation, dtype=float).reshape(3)
    transform = session.geometry.from_matrix(matrix)
    return {"transform": transform, **marker_metrics(records, transform)}


def _tsai_diagnostic(session, records, hard_results):
    transforms = [hard_results[name]["transform"] for name in ("Park", "Horaud")]
    consensus_rotation = R.from_quat([transform.rotation.as_quat() for transform in transforms]).mean()
    consensus_translation = np.mean([transform.translation for transform in transforms], axis=0)
    consensus_abs_qw = abs(float(consensus_rotation.as_quat()[3]))
    diagnostic = {"consensus_abs_qw": consensus_abs_qw}
    if consensus_abs_qw < 0.05:
        return {"status": "not_applicable_half_turn", **diagnostic}
    try:
        result = _solve_method(session, records, _cv2().CALIB_HAND_EYE_TSAI)
        transform = result["transform"]
        translation_delta = float(np.linalg.norm(np.asarray(transform.translation) - consensus_translation))
        rotation_delta = float(math.degrees((consensus_rotation.inv() * transform.rotation).magnitude()))
        consistent = (
            translation_delta <= session.sampling_cfg.max_algorithm_translation_delta_m
            and rotation_delta <= session.sampling_cfg.max_algorithm_rotation_delta_deg
        )
        return {
            "status": "consistent" if consistent else "inconsistent_diagnostic",
            **diagnostic,
            "translation_delta_m": translation_delta,
            "rotation_delta_deg": rotation_delta,
        }
    except Exception as exc:
        return {"status": "solver_error", **diagnostic, "error": str(exc)}


def local_handeye_solve(session, records):
    methods = {
        "Park": _cv2().CALIB_HAND_EYE_PARK,
        "Horaud": _cv2().CALIB_HAND_EYE_HORAUD,
    }
    results = {}
    for name, method in methods.items():
        try:
            results[name] = _solve_method(session, records, method)
        except Exception as exc:
            results[name] = {"error": str(exc)}
    valid = [(name, result) for name, result in results.items() if "transform" in result]
    if len(valid) != len(methods):
        return None, None, results, {"status": "not_run_hard_solver"}
    name, result = min(valid, key=lambda item: (item[1]["position_rms_m"], item[1]["rotation_rms_deg"], item[0]))
    spread = _algorithm_spread(results)
    if (
        spread["translation_max_m"] > session.sampling_cfg.max_algorithm_translation_delta_m
        or spread["rotation_max_deg"] > session.sampling_cfg.max_algorithm_rotation_delta_deg
    ):
        tsai = {"status": "not_run_hard_consensus"}
    else:
        tsai = _tsai_diagnostic(session, records, results)
    return result["transform"], name, results, tsai


def _algorithm_spread(results):
    transforms = [value["transform"] for value in results.values() if "transform" in value]
    if len(transforms) < 2:
        return {"translation_max_m": 0.0, "rotation_max_deg": 0.0}
    pairs = [(left, right) for index, left in enumerate(transforms) for right in transforms[index + 1:]]
    return {
        "translation_max_m": max(float(np.linalg.norm(np.asarray(left.translation) - np.asarray(right.translation))) for left, right in pairs),
        "rotation_max_deg": max(float(math.degrees((left.rotation.inv() * right.rotation).magnitude())) for left, right in pairs),
    }


def _coverage(records):
    pairs = [(left.robot_pose, right.robot_pose) for index, left in enumerate(records) for right in records[index + 1:]]
    rotations = [(left.rotation.inv() * right.rotation) for left, right in pairs]
    axes = []
    for rotation in rotations:
        angle = rotation.magnitude()
        if math.radians(17.0) <= angle <= math.radians(120.0):
            axes.append(rotation.as_rotvec() / angle)
    scatter = sum((np.outer(axis, axis) for axis in axes), np.zeros((3, 3)))
    eigenvalues = np.linalg.eigvalsh(scatter)[::-1]
    axis_ratio = float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0.0 else 0.0
    return {
        "translation_span_m": max(float(np.linalg.norm(np.asarray(left.translation) - np.asarray(right.translation))) for left, right in pairs),
        "rotation_span_deg": max(float(math.degrees(rotation.magnitude())) for rotation in rotations),
        "informative_rotation_pairs": len(axes),
        "rotation_axis_eigenvalues": [float(value) for value in eigenvalues],
        "rotation_axis_ratio": axis_ratio,
    }


def _worst_record(records, metrics, sampling_config):
    scores = (
        np.asarray(metrics["per_sample_position_m"]) / sampling_config.max_marker_position_rms_m
        + np.asarray(metrics["per_sample_rotation_deg"]) / sampling_config.max_marker_rotation_rms_deg
    )
    return int(np.argmax(scores))


def _can_prune(failures):
    return set(failures) == {"fixed-marker residual"}


def _yaml_transform(transform):
    q = transform.rotation.as_quat()
    return {"translation": dict(zip(("x", "y", "z"), map(float, transform.translation))), "rotation": dict(zip(("x", "y", "z", "w"), map(float, q)))}


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _parameters(session):
    return {
        "name": session.sampling_cfg.calibration_file_prefix,
        "calibration_type": "eye_in_hand",
        "robot_base_frame": session.frames.base_frame,
        "robot_effector_frame": session.frames.ee_frame,
        "tracking_base_frame": session.frames.tracking_base_frame,
        "tracking_marker_frame": session.frames.tracking_marker_frame,
        "freehand_robot_movement": True,
        "move_group_namespace": session.motion_cfg.move_group_ns_fairino or "/",
        "move_group": session.motion_cfg.move_group_name,
    }


def _samples_data(session, records, *, accepted, rejection=None):
    return {
        "schema_version": 3,
        "accepted": bool(accepted),
        "rejection_reason": rejection,
        "parameters": _parameters(session),
        "samples": [
            {
                "candidate_id": record.candidate_idx,
                "image_stamp_ns": record.image_stamp_ns,
                "tool_delta": vars(record.spec),
                "robot": _yaml_transform(record.robot_pose),
                "tracking": _yaml_transform(record.tracking_pose),
                "quality": vars(record.quality),
            }
            for record in records
        ],
    }


def _timestamped_stem(session):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(session.sampling_cfg.calibration_output_directory) / f"{session.sampling_cfg.calibration_file_prefix}_{stamp}"


def save_partial_samples(session, records):
    path = _timestamped_stem(session).with_suffix(".samples")
    _write_yaml(path, _samples_data(session, records, accepted=False, rejection="collection incomplete"))
    return str(path)


def _simulation_truth(session):
    if not bool(getattr(session.node, "_use_sim_time", False)):
        return None, "real hardware"
    try:
        from rclpy.duration import Duration
        from rclpy.time import Time
        tf = session.tf_buffer.lookup_transform(session.frames.ee_frame, session.frames.tracking_base_frame, Time(), timeout=Duration(seconds=1.0))
        return session.geometry.tf_to_matrix(tf), "live TF"
    except Exception as exc:
        return None, f"TF lookup failed: {exc}"


def _transform_error(estimate, truth):
    return float(np.linalg.norm(np.asarray(estimate.translation) - np.asarray(truth.translation))), float(math.degrees((truth.rotation.inv() * estimate.rotation).magnitude()))


def finalize_calibration(session, records):
    retained = list(records)
    if len(retained) < session.sampling_cfg.minimum_samples:
        return False
    while len(retained) >= session.sampling_cfg.minimum_solution_samples:
        coverage = _coverage(retained)
        if (
            coverage["translation_span_m"] < session.sampling_cfg.min_translation_span_m
            or coverage["rotation_span_deg"] < session.sampling_cfg.min_rotation_span_deg
            or coverage["informative_rotation_pairs"] < session.sampling_cfg.min_informative_rotation_pairs
            or coverage["rotation_axis_ratio"] < session.sampling_cfg.min_rotation_axis_ratio
        ):
            session._logger().error(
                "Skip calibration: motion observability "
                f"(translation_span={coverage['translation_span_m']:.6f}m, "
                f"rotation_span={coverage['rotation_span_deg']:.3f}deg, "
                f"informative_pairs={coverage['informative_rotation_pairs']}, "
                f"axis_ratio={coverage['rotation_axis_ratio']:.6f})"
            )
            break
        seed, algorithm, algorithms, tsai_diagnostic = local_handeye_solve(session, retained)
        if seed is None:
            session._logger().error("Skip calibration: Park/Horaud hard solver failed")
            break
        spread = _algorithm_spread(algorithms)
        if (
            spread["translation_max_m"] > session.sampling_cfg.max_algorithm_translation_delta_m
            or spread["rotation_max_deg"] > session.sampling_cfg.max_algorithm_rotation_delta_deg
        ):
            session._logger().error(
                "Skip calibration: Park/Horaud spread "
                f"({spread['translation_max_m']:.6f}m, {spread['rotation_max_deg']:.3f}deg)"
            )
            break
        refined, refinement = refine_handeye_fixed_marker(retained, seed, translation_sigma_m=session.sampling_cfg.solver_translation_sigma_m, rotation_sigma_deg=session.sampling_cfg.solver_rotation_sigma_deg)
        residual = marker_metrics(retained, refined)
        failures = []
        if not np.all(np.isfinite(refined.translation)) or float(np.linalg.norm(refined.translation)) > session.sampling_cfg.max_calibration_translation_norm_m:
            failures.append("mount transform")
        if residual["position_rms_m"] > session.sampling_cfg.max_marker_position_rms_m or residual["rotation_rms_deg"] > session.sampling_cfg.max_marker_rotation_rms_deg:
            failures.append("fixed-marker residual")
        truth, truth_source = _simulation_truth(session)
        truth_error = None
        if bool(getattr(session.node, "_use_sim_time", False)):
            if truth is None:
                failures.append("simulation ground truth unavailable")
            else:
                truth_error = _transform_error(refined, truth)
                if truth_error[0] >= session.sampling_cfg.simulation_truth_translation_m or truth_error[1] >= session.sampling_cfg.simulation_truth_rotation_deg:
                    failures.append("simulation ground truth")
        if not failures:
            stem = _timestamped_stem(session)
            samples_path, calibration_path = stem.with_suffix(".samples"), stem.with_suffix(".calib")
            _write_yaml(samples_path, _samples_data(session, retained, accepted=True))
            _write_yaml(calibration_path, {
                "parameters": _parameters(session),
                "transform": _yaml_transform(refined),
                "selected_algorithm": algorithm,
                "fixed_marker_refinement": refinement,
                "fixed_marker_residual": residual,
                "algorithm_spread": spread,
                "tsai_lenz_diagnostic": tsai_diagnostic,
                "motion_coverage": coverage,
                "simulation_ground_truth": {"source": truth_source, "translation_error_m": truth_error[0], "rotation_error_deg": truth_error[1]} if truth_error else None,
            })
            session._logger().info(f"Calibration saved: {calibration_path}; samples: {samples_path}; retained={len(retained)}")
            return True
        if not _can_prune(failures):
            session._logger().error("Skip calibration: " + ", ".join(failures))
            break
        if len(retained) == session.sampling_cfg.minimum_solution_samples:
            session._logger().error("Skip calibration: " + ", ".join(failures))
            break
        rejected = retained.pop(_worst_record(retained, residual, session.sampling_cfg))
        session._logger().warn(f"Discard candidate {rejected.candidate_idx} during solve: " + ", ".join(failures))
    path = _timestamped_stem(session).with_suffix(".samples")
    _write_yaml(path, _samples_data(session, retained, accepted=False, rejection="solver quality gate failed"))
    session._logger().error(f"Calibration not saved; retained samples: {path}")
    return False
