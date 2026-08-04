from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parents[1]))

from hand_eye_calibration.collector import solver as solver_module
from hand_eye_calibration.collector.model import CollectorGeometry, TransformMatrix
from hand_eye_calibration.collector.solver import (
    _algorithm_spread,
    _can_prune,
    _coverage,
    finalize_calibration,
    local_handeye_solve,
)


_ROBOT_POSES = (
    ((0.00, 0.00, 0.00), (0, 0, 0)),
    ((0.03, 0.01, 0.01), (25, 0, 0)),
    ((-0.02, 0.04, 0.02), (0, 25, 0)),
    ((0.01, -0.03, 0.03), (0, 0, 25)),
    ((0.04, -0.02, -0.01), (20, 15, 0)),
    ((-0.03, -0.02, 0.015), (-15, 10, 20)),
    ((0.02, 0.03, -0.02), (12, -18, -15)),
    ((-0.04, 0.01, -0.01), (-20, -12, 18)),
)


def _matrix(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = translation
    return matrix


def _transform(matrix):
    return TransformMatrix(
        R.from_matrix(matrix[:3, :3]),
        tuple(float(value) for value in matrix[:3, 3]),
    )


def _synthetic_records(camera_yaw_deg, robot_poses=_ROBOT_POSES):
    ee_T_camera = _matrix(
        R.from_euler("z", camera_yaw_deg, degrees=True),
        (0.0325, -0.034975, -0.0494),
    )
    base_T_marker = _matrix(
        R.from_euler("xyz", (15, -10, 20), degrees=True),
        (0.35, 0.0, 0.0),
    )
    records = []
    for translation, rpy_deg in robot_poses:
        base_T_ee = _matrix(R.from_euler("xyz", rpy_deg, degrees=True), translation)
        camera_T_marker = np.linalg.inv(base_T_ee @ ee_T_camera) @ base_T_marker
        records.append(
            SimpleNamespace(
                robot_pose=_transform(base_T_ee),
                tracking_pose=_transform(camera_T_marker),
            )
        )
    return records, _transform(ee_T_camera)


def _session():
    return SimpleNamespace(
        geometry=CollectorGeometry(base_frame="base_link"),
        sampling_cfg=SimpleNamespace(
            min_informative_rotation_pairs=20,
            min_rotation_axis_ratio=0.20,
            max_algorithm_translation_delta_m=0.003,
            max_algorithm_rotation_delta_deg=1.0,
        ),
    )


def _transform_error(estimate, truth):
    return (
        float(np.linalg.norm(np.asarray(estimate.translation) - np.asarray(truth.translation))),
        float(np.degrees((truth.rotation.inv() * estimate.rotation).magnitude())),
    )


def test_observability_requires_informative_pairs_on_multiple_axes():
    records, _ = _synthetic_records(180.0)
    coverage = _coverage(records)
    assert coverage["translation_span_m"] >= 0.040
    assert coverage["rotation_span_deg"] >= 20.0
    assert coverage["informative_rotation_pairs"] >= 20
    assert coverage["rotation_axis_ratio"] >= 0.20
    assert len(coverage["rotation_axis_eigenvalues"]) == 3

    single_axis = tuple(
        ((0.01 * index, 0.0, 0.0), (0.0, 0.0, 18.0 * index))
        for index in range(8)
    )
    records, _ = _synthetic_records(180.0, single_axis)
    coverage = _coverage(records)
    assert coverage["informative_rotation_pairs"] >= 20
    assert coverage["rotation_axis_ratio"] < 0.20


def test_half_turn_mount_uses_park_horaud_and_skips_tsai():
    for yaw_deg in (180.0, 179.0):
        records, truth = _synthetic_records(yaw_deg)
        estimate, algorithm, hard_results, tsai = local_handeye_solve(_session(), records)
        assert algorithm in {"Park", "Horaud"}
        assert set(hard_results) == {"Park", "Horaud"}
        assert _transform_error(estimate, truth)[0] < 1.0e-9
        assert _transform_error(estimate, truth)[1] < 1.0e-7
        assert _algorithm_spread(hard_results)["rotation_max_deg"] < 1.0e-7
        assert tsai["status"] == "not_applicable_half_turn"
        assert tsai["consensus_abs_qw"] < 0.05


def test_tsai_is_diagnostic_when_mount_is_not_near_half_turn():
    records, truth = _synthetic_records(170.0)
    estimate, algorithm, hard_results, tsai = local_handeye_solve(_session(), records)
    assert algorithm in {"Park", "Horaud"}
    assert set(hard_results) == {"Park", "Horaud"}
    assert _transform_error(estimate, truth)[1] < 1.0e-7
    assert tsai["status"] == "consistent"
    assert tsai["translation_delta_m"] < 1.0e-9
    assert tsai["rotation_delta_deg"] < 1.0e-7


def test_inconsistent_tsai_does_not_replace_hard_consensus(monkeypatch):
    cv2 = solver_module._cv2()

    def calibrate(*args, method, **kwargs):
        if method == cv2.CALIB_HAND_EYE_TSAI:
            return np.eye(3), np.zeros((3, 1))
        return cv2.calibrateHandEye(*args, method=method, **kwargs)

    fake_cv2 = SimpleNamespace(
        CALIB_HAND_EYE_PARK=cv2.CALIB_HAND_EYE_PARK,
        CALIB_HAND_EYE_HORAUD=cv2.CALIB_HAND_EYE_HORAUD,
        CALIB_HAND_EYE_TSAI=cv2.CALIB_HAND_EYE_TSAI,
        calibrateHandEye=calibrate,
    )
    monkeypatch.setattr(solver_module, "_cv2", lambda: fake_cv2)

    records, truth = _synthetic_records(170.0)
    estimate, algorithm, hard_results, tsai = local_handeye_solve(_session(), records)
    assert algorithm in {"Park", "Horaud"}
    assert set(hard_results) == {"Park", "Horaud"}
    assert _transform_error(estimate, truth)[1] < 1.0e-7
    assert tsai["status"] == "inconsistent_diagnostic"


def test_both_hard_solvers_are_required(monkeypatch):
    cv2 = solver_module._cv2()

    def calibrate(*args, method, **kwargs):
        if method == cv2.CALIB_HAND_EYE_HORAUD:
            raise RuntimeError("Horaud failed")
        return cv2.calibrateHandEye(*args, method=method, **kwargs)

    fake_cv2 = SimpleNamespace(
        CALIB_HAND_EYE_PARK=cv2.CALIB_HAND_EYE_PARK,
        CALIB_HAND_EYE_HORAUD=cv2.CALIB_HAND_EYE_HORAUD,
        CALIB_HAND_EYE_TSAI=cv2.CALIB_HAND_EYE_TSAI,
        calibrateHandEye=calibrate,
    )
    monkeypatch.setattr(solver_module, "_cv2", lambda: fake_cv2)

    records, _ = _synthetic_records(170.0)
    estimate, algorithm, hard_results, tsai = local_handeye_solve(_session(), records)
    assert estimate is None
    assert algorithm is None
    assert "transform" in hard_results["Park"]
    assert "error" in hard_results["Horaud"]
    assert tsai["status"] == "not_run_hard_solver"


def test_observability_failure_stops_before_closed_form_solve(monkeypatch, tmp_path):
    single_axis = tuple(
        ((0.01 * index, 0.0, 0.0), (0.0, 0.0, 18.0 * index))
        for index in range(8)
    )
    records, _ = _synthetic_records(180.0, single_axis)
    errors = []
    session = SimpleNamespace(
        sampling_cfg=SimpleNamespace(
            minimum_samples=3,
            minimum_solution_samples=3,
            min_translation_span_m=0.040,
            min_rotation_span_deg=20.0,
            min_informative_rotation_pairs=20,
            min_rotation_axis_ratio=0.20,
            calibration_output_directory=str(tmp_path),
            calibration_file_prefix="test",
        ),
        _logger=lambda: SimpleNamespace(error=errors.append),
    )
    monkeypatch.setattr(
        solver_module,
        "local_handeye_solve",
        lambda *_args: (_ for _ in ()).throw(AssertionError("solver must not run")),
    )
    monkeypatch.setattr(solver_module, "_samples_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(solver_module, "_write_yaml", lambda *_args: None)

    assert not finalize_calibration(session, records)
    assert any("motion observability" in message for message in errors)


def test_only_fixed_marker_residual_is_prunable():
    assert _can_prune(["fixed-marker residual"])
    assert not _can_prune(["closed-form algorithm spread"])
    assert not _can_prune(["fixed-marker residual", "closed-form algorithm spread"])
    assert not _can_prune(["motion observability"])
    assert not _can_prune(["simulation ground truth", "fixed-marker residual"])
