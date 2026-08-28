"""Static regression checks for the MPC demo entry point."""

import os
from pathlib import Path


def test_mpc_demo_uses_shared_moveit_path_without_manual_plan_request():
    script = Path(__file__).parents[1] / "scripts" / "mpc_avoidance_node_sim.py"
    source = script.read_text()

    for symbol in ("param", "PoseTools", "MoveIt2", "MoveItMotion", "DemoStage"):
        assert symbol in source
    removed_symbols = (
        "MotionPlanRequest",
        "PositionConstraint",
        "OrientationConstraint",
        "GetMotionPlan",
    )
    for removed in removed_symbols:
        assert removed not in source
    assert "self.motion.plan_to_pose(" in source


def test_mpc_demo_entry_point_is_executable():
    script = Path(__file__).parents[1] / "scripts" / "mpc_avoidance_node_sim.py"
    assert os.access(script, os.X_OK)


def test_mpc_launch_keeps_solver_selection_in_yaml():
    launch = (
        Path(__file__).parents[1] / "launch" / "mpc_avoidance_demo_sim.launch.py"
    ).read_text()
    assert '("solver_type"' not in launch
    assert '"ik_plugin"' in launch
    assert '"planning_pipeline_id"' in launch
