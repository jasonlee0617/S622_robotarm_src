from itertools import combinations
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from hand_eye_calibration.collector.config import ROOT_RELATIVE_TOOL_DELTAS, _tool_deltas


EXPECTED_TOOL_DELTAS = (
    "-0.03,0.02,0.01,5.01,9.58,13.29",
    "-0.04,0.03,0.02,9.58,15.82,16.57",
    "-0.03,0.05,0.03,13.29,16.57,7.38",
    "0.00,0.07,0.02,15.82,11.56,-7.38",
    "0.04,0.06,0.00,16.95,2.53,-16.57",
    "0.06,0.04,-0.01,16.57,-7.38,-13.29",
    "0.07,0.02,0.00,14.72,-14.72,0.00",
    "0.06,0.02,0.02,11.56,-16.95,13.29",
    "0.04,0.01,0.03,7.38,-13.29,16.57",
    "0.02,0.00,0.01,2.53,-5.01,7.38",
    "-0.02,0.00,-0.01,-2.53,5.01,-7.38",
    "-0.05,0.01,-0.02,-7.38,13.29,-16.57",
    "-0.07,0.01,0.00,-11.56,16.95,-13.29",
    "-0.07,-0.02,0.02,-14.72,14.72,0.00",
    "-0.04,-0.05,0.02,-16.57,7.38,13.29",
    "-0.01,-0.06,0.01,-16.95,-2.53,16.57",
    "0.02,-0.07,0.00,-15.82,-11.56,7.38",
    "0.03,-0.07,-0.01,-13.29,-16.57,-7.38",
    "0.03,-0.06,-0.01,-9.58,-15.82,-16.57",
)


def _pose(raw):
    values = tuple(float(value) for value in raw.split(","))
    return np.asarray(values[:3]), R.from_euler("xyz", values[3:], degrees=True)


class _FakeNode:
    def __init__(self, overrides=None):
        self.values = dict(overrides or {})

    def declare_parameter(self, name, default):
        self.values.setdefault(name, default)

    def get_parameter(self, name):
        return SimpleNamespace(value=self.values[name])


def test_python_and_yaml_use_the_same_fixed_sequence_default():
    config_path = Path(__file__).parents[1] / "config" / "auto_calibration_collector.yaml"
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))["auto_calibration_collector"]["ros__parameters"]
    assert tuple(ROOT_RELATIVE_TOOL_DELTAS) == EXPECTED_TOOL_DELTAS
    assert tuple(parameters["tool_delta_specs"]) == EXPECTED_TOOL_DELTAS
    assert parameters["minimum_samples"] == 15
    assert parameters["minimum_solution_samples"] == 14
    assert parameters["step_between_actions"] is True
    for raw in EXPECTED_TOOL_DELTAS:
        assert all(re.fullmatch(r"-?\d+\.\d{2}", value) for value in raw.split(","))


def test_explicit_parameter_overrides_yaml_default():
    override = list(EXPECTED_TOOL_DELTAS)
    override[0] = "0.02,0.03,0.02,5.02,9.58,13.29"
    specs = _tool_deltas(_FakeNode({"tool_delta_specs": override}), "tool_delta_specs", EXPECTED_TOOL_DELTAS)
    assert specs[0].rx_deg == pytest.approx(5.02)

    yaml_specs = _tool_deltas(_FakeNode(), "tool_delta_specs", EXPECTED_TOOL_DELTAS)
    assert yaml_specs[0].rx_deg == pytest.approx(5.01)


def test_root_plus_nineteen_actions_are_unique_and_observable():
    assert len(ROOT_RELATIVE_TOOL_DELTAS) == 19
    poses = [(np.zeros(3), R.identity())] + [_pose(raw) for raw in ROOT_RELATIVE_TOOL_DELTAS]
    for (left_t, left_r), (right_t, right_r) in combinations(poses, 2):
        assert np.linalg.norm(left_t - right_t) >= 0.006 or np.degrees((left_r.inv() * right_r).magnitude()) >= 3.0
    translations = [np.linalg.norm(left_t - right_t) for (left_t, _), (right_t, _) in combinations(poses, 2)]
    rotations = [np.degrees((left_r.inv() * right_r).magnitude()) for (_, left_r), (_, right_r) in combinations(poses, 2)]
    assert max(translations) >= 0.040
    assert max(rotations) >= 20.0


def test_rotation_pairs_have_two_axis_observability():
    rotations = [R.identity()] + [_pose(raw)[1] for raw in ROOT_RELATIVE_TOOL_DELTAS]
    informative_axes = []
    for left, right in combinations(rotations, 2):
        rotvec = (left.inv() * right).as_rotvec()
        angle = np.linalg.norm(rotvec)
        if 17.0 <= np.degrees(angle) <= 120.0:
            informative_axes.append(rotvec / angle)

    scatter = sum((np.outer(axis, axis) for axis in informative_axes), start=np.zeros((3, 3)))
    eigenvalues = np.linalg.eigvalsh(scatter)[::-1]
    config_path = Path(__file__).parents[1] / "config" / "auto_calibration_collector.yaml"
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))["auto_calibration_collector"]["ros__parameters"]
    axis_ratio = eigenvalues[1] / eigenvalues[0]
    assert len(informative_axes) == 164
    assert axis_ratio == pytest.approx(0.9543589373, abs=1.0e-6)
    assert len(informative_axes) >= parameters["min_informative_rotation_pairs"]
    assert axis_ratio >= parameters["min_rotation_axis_ratio"]


def test_adjacent_actions_remain_conservative():
    poses = [(np.zeros(3), R.identity())] + [_pose(raw) for raw in ROOT_RELATIVE_TOOL_DELTAS]
    root_norm = max(np.linalg.norm(translation) for translation, _ in poses)
    adjacent_translation = max(np.linalg.norm(left_t - right_t) for (left_t, _), (right_t, _) in zip(poses, poses[1:]))
    adjacent_rotation = max(np.degrees((left_r.inv() * right_r).magnitude()) for (_, left_r), (_, right_r) in zip(poses, poses[1:]))
    assert root_norm == pytest.approx(0.0768114575, abs=1.0e-9)
    assert adjacent_translation == pytest.approx(0.0458257569, abs=1.0e-9)
    assert adjacent_rotation == pytest.approx(18.52292129, abs=1.0e-8)


def test_trajectory_gate_keys_are_present_in_yaml():
    config_path = Path(__file__).parents[1] / "config" / "auto_calibration_collector.yaml"
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))["auto_calibration_collector"]["ros__parameters"]
    for key in ("candidate_max_joint_excursion_rad", "candidate_max_adjacent_joint_jump_rad", "candidate_max_wrist_travel_rad"):
        assert key in parameters
    assert "look_at_specs" not in parameters
