from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from hand_eye_calibration import solver
from hand_eye_calibration.config import CalibrationType


class _Logger:
    def info(self, _message): pass
    def warn(self, _message): pass
    def error(self, _message): pass


def _config(directory):
    return SimpleNamespace(
        minimum_translation_span_m=0.04, minimum_rotation_span_deg=20.0,
        minimum_samples=15, minimum_solution_samples=14,
        algorithm_names=("OpenCV/Park", "OpenCV/Horaud"),
        maximum_algorithm_translation_delta_m=0.003, maximum_algorithm_rotation_delta_deg=1.0,
        maximum_camera_translation_norm_m=0.30,
        maximum_eye_on_base_camera_translation_norm_m=2.0,
        fixed_marker_refinement_translation_sigma_m=0.0005,
        fixed_marker_refinement_rotation_sigma_deg=0.30,
        fixed_marker_refinement_max_iterations=25,
        maximum_marker_position_rms_m=0.002, maximum_marker_rotation_rms_deg=0.70,
        ground_truth_check_enabled=False,
        calibration_output_directory=str(directory), calibration_file_prefix="test",
        ground_truth_max_translation_error_m=0.003, ground_truth_max_axis_error_m=0.002,
        ground_truth_max_rotation_error_deg=1.0,
    )


def _records(count=15):
    """交替绕 x/y/z 轴旋转，位姿多样、覆盖充分。"""
    records = []
    for index in range(count):
        if index % 3 == 0:
            rotation = R.from_euler("x", index * 5.0, degrees=True)
        elif index % 3 == 1:
            rotation = R.from_euler("y", index * 5.0, degrees=True)
        else:
            rotation = R.from_euler("z", index * 5.0, degrees=True)
        records.append(solver.CalibrationSample(
            index + 1, (64.0, -114.0, -34.0, -122.0, 90.0, 64.0),
            solver.TransformMatrix(rotation, (0.005 * index, 0.0, 0.0)),
            solver.TransformMatrix(R.identity(), (0.0, 0.0, 0.4)),
        ))
    return records


def _half_turn_records():
    poses = (
        ((0.004954334824630813, 0.3263515244229137, 0.2596290276156682), (0.9999122321527542, 0.006687741337267737, 0.009893548431627352, -0.005737578455541052)),
        ((0.006249836242408796, 0.29094302553937473, 0.22541330808271362), (0.9955509243205061, 0.009571979003763054, 0.003695987238518431, -0.09366468908222729)),
        ((0.004145299818163195, 0.3617527421619065, 0.22076466756279017), (0.9906352209322942, 0.0023569569221795206, 0.0029760589570531066, 0.13648240500363348)),
        ((0.007729411132412933, 0.32753530969378136, 0.2586564914881788), (0.9761145306626385, -0.21694951149517336, 0.0078084840260485506, -0.008506472152278016)),
    )
    truth = solver.TransformMatrix(R.from_euler("z", 180.0, degrees=True), (0.0325, -0.0375, -0.0794))
    marker = solver.TransformMatrix(R.from_euler("xyz", (15.0, -10.0, 25.0), degrees=True), (0.31, 0.27, 0.02))
    records = []
    for index, (translation, quaternion) in enumerate(poses, 1):
        robot = solver.TransformMatrix(R.from_quat(quaternion), translation)
        tracking = np.linalg.inv(robot.matrix() @ truth.matrix()) @ marker.matrix()
        records.append(solver.CalibrationSample(index, (0.0,) * 6, robot, solver.transform_from_matrix(tracking)))
    return records, truth


class SolverTests(unittest.TestCase):
    def test_hard_algorithm_consensus(self):
        results = {
            "OpenCV/Park": solver.TransformMatrix(R.identity(), (0.000, 0.0, 0.0)),
            "OpenCV/Horaud": solver.TransformMatrix(R.from_euler("z", 0.1, degrees=True), (0.001, 0.0, 0.0)),
        }
        name, _transform, translation, rotation = solver.consensus(results)
        self.assertEqual(name, "OpenCV/Park")
        self.assertEqual(translation, 0.001)
        self.assertAlmostEqual(rotation, 0.1, places=6)

    def test_tsai_diagnostic_cannot_veto_hard_consensus(self):
        config = _config(tempfile.mkdtemp())
        hard = {
            "OpenCV/Park": solver.TransformMatrix(R.identity(), (0.03, -0.04, -0.08)),
            "OpenCV/Horaud": solver.TransformMatrix(R.identity(), (0.0305, -0.04, -0.08)),
        }
        refined = solver.TransformMatrix(R.identity(), (0.03, -0.04, -0.08))
        metrics = {"position_rms_m": 0.0, "rotation_rms_deg": 0.0, "per_sample_position_m": [0.0] * 3, "per_sample_rotation_deg": [0.0] * 3}
        failures = (
            solver.TransformMatrix(R.identity(), (0.4, 0.0, 0.0)),
            solver.TransformMatrix(R.identity(), (float("nan"), 0.0, 0.0)),
            RuntimeError("Tsai-Lenz failed"),
        )
        for tsai in failures:
            with self.subTest(tsai=type(tsai).__name__):
                with patch.object(solver, "solve_algorithms", side_effect=[hard, tsai]), patch.object(solver, "refine_handeye_fixed_marker", return_value=(refined, {"success": True, "iterations": 1})), patch.object(solver, "marker_metrics", return_value=metrics):
                    valid, _transform, _name, _translation, _rotation, _metrics, details = solver._solve_once(_records(3), config)
            self.assertTrue(valid)
            self.assertIn("Tsai-Lenz", details["tsai_diagnostic"])

    def test_half_turn_mount_makes_tsai_diagnostic_only(self):
        records, truth = _half_turn_records()
        results = solver.solve_algorithms(records, ("OpenCV/Park", "OpenCV/Horaud", "OpenCV/Tsai-Lenz"))
        for name in ("OpenCV/Park", "OpenCV/Horaud"):
            self.assertLess(np.linalg.norm(np.asarray(results[name].translation) - np.asarray(truth.translation)), 1.0e-6)
            self.assertLess(solver.rotation_delta_deg(results[name].rotation, truth.rotation), 1.0e-6)
        self.assertGreater(np.linalg.norm(np.asarray(results["OpenCV/Tsai-Lenz"].translation) - np.asarray(truth.translation)), 0.1)

    def test_coverage_and_truth_boundaries(self):
        records = _records()
        config = _config(tempfile.mkdtemp())
        self.assertTrue(solver.coverage_status(records, config, minimum_count=15)[0])
        truth = solver.TransformMatrix(R.identity(), (0.0, 0.0, 0.0))
        estimate = solver.TransformMatrix(R.from_euler("x", 1.0, degrees=True), (0.002, 0.002, 0.001))
        self.assertTrue(solver.truth_status(estimate, truth, config)[0])

    def test_truth_gate_checks_every_translation_axis_and_reports_dz(self):
        config = _config(tempfile.mkdtemp())
        truth = solver.TransformMatrix(R.identity(), (0.0, 0.0, 0.0))
        z_outlier = solver.TransformMatrix(R.identity(), (0.0, 0.0, 0.0021))
        norm_outlier = solver.TransformMatrix(R.identity(), (0.0018, 0.0018, 0.0018))

        self.assertFalse(solver.truth_status(z_outlier, truth, config)[0])
        self.assertFalse(solver.truth_status(norm_outlier, truth, config)[0])
        ok, message = solver.truth_status(
            solver.TransformMatrix(R.identity(), (0.001, -0.001, 0.001)), truth, config,
        )
        self.assertTrue(ok)
        for field in ("dx=", "dy=", "dz=", "translation=", "rotation="):
            self.assertIn(field, message)

    def test_simulation_truth_uses_the_type_specific_parent_frame(self):
        for calibration_type, expected_parent in (
            (CalibrationType.EYE_IN_HAND, "tool0"),
            (CalibrationType.EYE_ON_BASE, "base_link"),
        ):
            calls = []
            session = SimpleNamespace(
                _use_sim_time=True,
                frames_config=SimpleNamespace(
                    calibration_type=calibration_type,
                    ee_frame="tool0",
                    base_frame="base_link",
                    tracking_base_frame="camera_color_optical_frame",
                ),
                tf_buffer=SimpleNamespace(
                    lookup_transform=lambda parent, child, *_args, **_kwargs: calls.append((parent, child)) or object(),
                ),
                tf_to_matrix=lambda _transform: solver.TransformMatrix(R.identity(), (0.0, 0.0, 0.0)),
            )
            truth, _note = solver.freeze_simulation_truth(session)
            self.assertIsNotNone(truth)
            self.assertEqual(calls, [(expected_parent, "camera_color_optical_frame")])

    def test_samples_are_compact_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(
                sampling_config=_config(directory),
                frames_config=SimpleNamespace(calibration_type=CalibrationType.EYE_IN_HAND),
                get_logger=lambda: _Logger(),
            )
            path = Path(solver.save_samples(session, _records(1), "incomplete"))
            data = yaml.safe_load(path.read_text())
            self.assertEqual(set(data), {"calibration_type", "status", "samples"})
            self.assertEqual(set(data["samples"][0]), {"waypoint_index", "target_joints_deg", "base_T_ee", "camera_T_marker"})
            self.assertTrue(path.name.endswith("_eye_in_hand.samples"))

    def test_eye_on_base_robot_pose_and_suffix(self):
        base_T_ee = solver.TransformMatrix(R.from_euler("z", 30.0, degrees=True), (0.2, -0.1, 0.4))
        actual = solver.robot_pose_for_calibration(base_T_ee, CalibrationType.EYE_ON_BASE)
        np.testing.assert_allclose(actual.matrix(), np.linalg.inv(base_T_ee.matrix()), atol=1.0e-12)
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(
                sampling_config=_config(directory),
                frames_config=SimpleNamespace(calibration_type=CalibrationType.EYE_ON_BASE),
                get_logger=lambda: _Logger(),
            )
            path = Path(solver.save_samples(session, _records(1), "incomplete"))
            data = yaml.safe_load(path.read_text())
            self.assertTrue(path.name.endswith("_eye_on_base.samples"))
            self.assertIn("ee_T_base", data["samples"][0])

    def test_eye_on_base_recovers_base_to_camera(self):
        truth = solver.TransformMatrix(
            R.from_euler("xyz", (10.0, -20.0, 30.0), degrees=True),
            (0.4, 0.2, 0.9),
        )
        ee_T_marker = solver.TransformMatrix(
            R.from_euler("xyz", (5.0, 10.0, -15.0), degrees=True),
            (0.03, -0.02, 0.12),
        )
        records = []
        for index in range(12):
            base_T_ee = solver.TransformMatrix(
                R.from_euler(
                    "xyz",
                    ((index % 4) * 8.0, (index % 3) * -7.0, index * 5.0),
                    degrees=True,
                ),
                (0.2 + 0.01 * index, -0.15 + 0.004 * index, 0.4 + 0.006 * (index % 5)),
            )
            camera_T_marker = np.linalg.inv(truth.matrix()) @ base_T_ee.matrix() @ ee_T_marker.matrix()
            records.append(solver.CalibrationSample(
                index + 1,
                (0.0,) * 6,
                solver.robot_pose_for_calibration(base_T_ee, CalibrationType.EYE_ON_BASE),
                solver.transform_from_matrix(camera_T_marker),
            ))
        results = solver.solve_algorithms(records, ("OpenCV/Park", "OpenCV/Horaud"))
        for result in results.values():
            np.testing.assert_allclose(result.translation, truth.translation, atol=1.0e-9)
            self.assertLess(solver.rotation_delta_deg(result.rotation, truth.rotation), 1.0e-8)

    def test_quality_prunes_to_fourteen(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            session = SimpleNamespace(
                sampling_config=config,
                frames_config=SimpleNamespace(calibration_type=CalibrationType.EYE_IN_HAND),
                _use_sim_time=False,
                get_logger=lambda: _Logger(),
            )
            records = _records(15)
            transform = solver.TransformMatrix(R.identity(), (0.0, 0.0, 0.0))
            invalid = (False, transform, "OpenCV/Park", 0.0, 0.0, {"position_rms_m": 0.01, "rotation_rms_deg": 1.0, "per_sample_position_m": [0.0] * 14 + [0.01], "per_sample_rotation_deg": [0.0] * 15}, {"iterations": 1})
            valid = (True, transform, "OpenCV/Park", 0.0, 0.0, {"position_rms_m": 0.0, "rotation_rms_deg": 0.0, "per_sample_position_m": [0.0] * 14, "per_sample_rotation_deg": [0.0] * 14}, {"iterations": 1})
            with patch.object(solver, "_solve_once", side_effect=(invalid, valid)), patch.object(solver, "_save_calibration", return_value="x.calib"), patch.object(solver, "save_samples") as save:
                self.assertTrue(solver.finalize_calibration(session, records))
            self.assertEqual(len(save.call_args.args[1]), 14)

if __name__ == "__main__":
    unittest.main()
