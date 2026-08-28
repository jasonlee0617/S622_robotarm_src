#!/usr/bin/env python3
"""Static contract checks for the merged planning/IK simulation entrypoint."""

import ast
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
NODE_FILE = PACKAGE_DIR / "scripts" / "motion_planning_node_sim.py"
LAUNCH_FILE = PACKAGE_DIR / "launch" / "motion_planning_demo_sim.launch.py"


def _method_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        item.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MotionPlanningNodeSim"
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    }


def test_merged_node_keeps_planning_and_raw_ik_modes():
    methods = _method_names(NODE_FILE)
    assert {
        "setup_scene",
        "select_mode",
        "run_planning_mode",
        "run_ik_comparison_mode",
        "compare_ik",
        "report_tf_position_error",
    } <= methods
    source = NODE_FILE.read_text(encoding="utf-8")
    assert "request.ik_request.avoid_collisions = False" in source


def test_launch_uses_only_the_merged_executable():
    source = LAUNCH_FILE.read_text(encoding="utf-8")
    assert 'executable="motion_planning_node_sim.py"' in source
    assert "trajectory_plan" + "_node.py" not in source
    assert "run_mode" in source
    assert "benchmark_output_dir" in source
    assert "OnProcessExit" in source
    assert "benchmark_execution" in source
    assert "benchmark_algorithm" in source
    assert "'benchmark'" not in source


def test_planning_launches_keep_cases_in_yaml_and_map_public_moveit_names():
    motion = LAUNCH_FILE.read_text(encoding="utf-8")
    assert "motion_planning_demo_params.yaml" in motion
    assert "planning_client" in motion
    assert '"scene_name"' not in motion.split("_PUBLIC_ARGUMENTS", 1)[1].split("_DEFAULTS", 1)[0]
    assert "benchmark_goal_seed" not in motion.split("_PUBLIC_ARGUMENTS", 1)[1].split("_RAW_DEFAULTS", 1)[0]
    assert '"moveit_clients"' in motion


def test_legacy_benchmark_entrypoints_are_removed():
    assert not (PACKAGE_DIR / "launch" / ("trajectory_plan" + "_test_sim.launch.py")).exists()
    assert not (PACKAGE_DIR / "scripts" / ("trajectory_plan" + "_test_node_sim.py")).exists()
    assert not (PACKAGE_DIR / "scripts" / ("collect_planning" + "_diagnostics.sh")).exists()


def test_benchmark_uses_selected_moveit_client():
    motion = LAUNCH_FILE.read_text(encoding="utf-8")
    assert '"moveit_clients"' in motion
    assert 'else (node_params["planning_client"],)' in motion
