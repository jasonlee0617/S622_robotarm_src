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
