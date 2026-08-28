#!/usr/bin/env python3
"""Small regression checks for node-owned benchmark artifacts."""

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from motion_planning_node_sim import MotionPlanningNodeSim  # noqa: E402


def _benchmark_node(directory, run_mode):
    node = object.__new__(MotionPlanningNodeSim)
    run_dir = Path(directory) / "run"
    run_dir.mkdir()
    node.run_mode = run_mode
    node.benchmark_executes_trajectory = run_mode == "benchmark_execution"
    node.benchmark_output_dir = directory
    node.benchmark_repetitions = 1
    node.benchmark_startup_joint_state_timeout_s = 1.0
    node.default_planner_id = "birrt*"
    node.planner_random_seed = 7
    node.scene_benchmark = {"start_pose": "0,0,0"}
    node.setup_scene = Mock()
    node._wait_for_complete_joint_state = Mock(return_value=True)
    node._ensure_home = Mock(return_value=(True, ""))
    node.go_home = Mock(return_value=True)
    node._execute_joint_trajectory = Mock(return_value=(True, ""))
    node._prepare_benchmark_artifacts = Mock(return_value=(directory, str(run_dir), {}))
    node.get_parameter = lambda name: SimpleNamespace(
        value=False if name == "auto_add_obstacle" else "0,-180,0"
    )
    node._as_bool = bool
    node._parse_float_list = lambda _value: (0.0, -180.0, 0.0)
    node._parse_pose_values = lambda *_args: ((0.0, 0.0, 0.0), (0.0, -180.0, 0.0))
    node._generate_benchmark_goals = Mock(return_value=[((0.2, 0.2, 0.2), (0.0, -180.0, 0.0))])
    node._write_generated_goals_csv = Mock()
    node.make_pose_from_xyzrpy = lambda *args: args
    node._plan_pose_from_home = Mock(return_value={
        "success": True, "error_code": "", "core_planning_time_s": 0.1, "trajectory": object(),
    })
    node._joint_trajectory_path_length = Mock(return_value=1.0)
    node._publish_display_trajectory = Mock()
    return node, run_dir


def test_summary_and_results_keep_only_core_benchmark_metrics():
    rows = [
        {
            "run_index": 1, "run_mode": "benchmark_execution", "planner_id": "birrt*", "planner_random_seed": 7,
            "plan_success": "true", "success": "true", "failure_phase": "none",
            "error_code": "", "goal_pose": "0/0/0/0/0/0",
            "core_planning_time_s": "0.100000",
            "optimized_joint_path_length_rad": "1.000000",
            "execution_success": "true", "return_home_success": "true",
        },
        {
            "run_index": 2, "run_mode": "benchmark_execution", "planner_id": "birrt*", "planner_random_seed": 7,
            "plan_success": "false", "success": "false", "failure_phase": "goal_plan",
            "error_code": "-1", "goal_pose": "0/0/0/0/0/0",
            "core_planning_time_s": "0.000000",
            "optimized_joint_path_length_rad": "0.000000",
            "execution_success": "false", "return_home_success": "false",
        },
    ]
    with tempfile.TemporaryDirectory() as directory:
        results, summary = Path(directory) / "results.csv", Path(directory) / "summary.md"
        MotionPlanningNodeSim._write_results(str(results), rows)
        MotionPlanningNodeSim._write_benchmark_summary(
            str(summary), rows, expected=3, run_mode="benchmark_execution"
        )
        written = list(csv.DictReader(results.open(encoding="utf-8")))
        assert written[1]["failure_phase"] == "goal_plan"
        assert list(written[0]) == [
            "run_index", "run_mode", "planner_id", "planner_random_seed", "plan_success",
            "success", "failure_phase", "error_code", "goal_pose",
            "core_planning_time_s", "optimized_joint_path_length_rad",
            "execution_success", "return_home_success",
        ]
        text = summary.read_text(encoding="utf-8")
        assert "expected_runs: 3" in text
        assert "actual_runs: 2" in text
        assert "goal_plan" in text
        assert "run_mode: benchmark_execution" in text
        assert "closed_loop_success_rate" in text
        assert "final_path" not in text


def test_algorithm_summary_never_claims_closed_loop_success():
    rows = [{
        "run_index": 1, "run_mode": "benchmark_algorithm", "planner_id": "birrt*",
        "planner_random_seed": 7, "plan_success": "true", "success": "true",
        "failure_phase": "none", "error_code": "", "goal_pose": "0/0/0/0/0/0",
        "core_planning_time_s": "0.100000", "optimized_joint_path_length_rad": "1.000000",
        "execution_success": "not_run", "return_home_success": "not_run",
    }]
    with tempfile.TemporaryDirectory() as directory:
        summary = Path(directory) / "summary.md"
        MotionPlanningNodeSim._write_benchmark_summary(
            str(summary), rows, expected=1, run_mode="benchmark_algorithm"
        )
        text = summary.read_text(encoding="utf-8")
        assert "planning_success_rate: 1/1" in text
        assert "closed_loop_success_rate" not in text


def test_benchmark_algorithm_homes_once_without_goal_execution():
    with tempfile.TemporaryDirectory() as directory:
        node, run_dir = _benchmark_node(directory, "benchmark_algorithm")
        node.run_benchmark()
        row = next(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
        assert node._ensure_home.call_count == 1
        node._execute_joint_trajectory.assert_not_called()
        node.go_home.assert_not_called()
        assert row["success"] == "true"
        assert row["execution_success"] == "not_run"


def test_benchmark_execution_keeps_goal_and_return_home_closure():
    with tempfile.TemporaryDirectory() as directory:
        node, _run_dir = _benchmark_node(directory, "benchmark_execution")
        node.run_benchmark()
        assert node._ensure_home.call_count == 1
        node._execute_joint_trajectory.assert_called_once()
        node.go_home.assert_called_once()


def test_sphere_extents_do_not_read_none_height():
    node = object.__new__(MotionPlanningNodeSim)
    assert node._obstacle_half_extents({"shape": "sphere", "radius": 0.06, "height": None}) == (0.06, 0.06, 0.06)


def test_benchmark_output_dir_expands_home_and_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_ARCHIVE_ROOT", str(tmp_path))
    assert MotionPlanningNodeSim._resolve_benchmark_output_dir(
        "$BENCHMARK_ARCHIVE_ROOT/case"
    ) == str(tmp_path / "case")
    assert MotionPlanningNodeSim._resolve_benchmark_output_dir(
        "~/sample_data/trajectory_plan_benchmark_sample"
    ) == "/home/robot/sample_data/trajectory_plan_benchmark_sample"


def test_case_lock_rejects_changed_benchmark_conditions():
    node = object.__new__(MotionPlanningNodeSim)
    node.default_planner_id = "birrt*"
    node.planner_random_seed = 7
    node._benchmark_config = lambda: {"scene": "layout_a", "seed": 17}
    with tempfile.TemporaryDirectory() as directory:
        node.benchmark_output_dir = directory
        with patch("motion_planning_node_sim.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.strftime.return_value = "20260827_205938"
            _case, run_dir, _config = node._prepare_benchmark_artifacts()
            _case, next_run_dir, _config = node._prepare_benchmark_artifacts()
        assert Path(run_dir).parent == Path(directory)
        assert Path(run_dir).name == "birrt__seed7_20260827_205938"
        assert Path(next_run_dir).name == "birrt__seed7_20260827_205938_1"
        assert (Path(run_dir) / "benchmark_config.yaml").exists()
        assert (Path(next_run_dir) / "benchmark_config.yaml").exists()
        assert not (Path(directory) / "benchmark_config.yaml").exists()
        assert not (Path(directory) / "generated_goals.csv").exists()
        assert not (Path(run_dir) / "run_config.yaml").exists()
        node._benchmark_config = lambda: {"scene": "layout_b", "seed": 17}
        try:
            node._prepare_benchmark_artifacts()
        except RuntimeError as exc:
            assert "case lock" in str(exc)
        else:
            raise AssertionError("changed case conditions must be rejected")


def test_legacy_root_snapshots_are_migrated_into_the_only_run():
    node = object.__new__(MotionPlanningNodeSim)
    node.default_planner_id = "birrt*"
    node.planner_random_seed = 7
    node._benchmark_config = lambda: {"scene": "layout_a", "seed": 17}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "birrt__seed7_20260827_205938"
        run.mkdir()
        (run / "results.csv").write_text("run_index\n1\n", encoding="utf-8")
        (root / "benchmark_config.yaml").write_text("scene: layout_a\nseed: 17\n", encoding="utf-8")
        (root / "generated_goals.csv").write_text("goal_index\n1\n", encoding="utf-8")
        node.benchmark_output_dir = directory
        with patch("motion_planning_node_sim.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.strftime.return_value = "20260827_205939"
            _case, new_run, _config = node._prepare_benchmark_artifacts()
        assert (run / "benchmark_config.yaml").exists()
        assert (run / "generated_goals.csv").exists()
        assert not (root / "benchmark_config.yaml").exists()
        assert not (root / "generated_goals.csv").exists()
        assert (Path(new_run) / "benchmark_config.yaml").exists()


def test_existing_goal_snapshot_is_selected_for_new_run():
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "birrt__seed7_20260827_205938"
        second = Path(directory) / "aapf_birrt__seed7_20260827_205939"
        first.mkdir()
        second.mkdir()
        goals = first / "generated_goals.csv"
        goals.write_text("goal_index\n1\n", encoding="utf-8")
        assert MotionPlanningNodeSim._find_existing_goals(directory, str(second)) == str(goals)
