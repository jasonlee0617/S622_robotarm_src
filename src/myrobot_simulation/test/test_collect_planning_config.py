#!/usr/bin/env python3
import re
import unittest
from pathlib import Path


class CollectPlanningConfigTest(unittest.TestCase):
    def setUp(self):
        self.script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "collect_planning_diagnostics.sh"
        ).read_text(encoding="utf-8")

    def test_top_level_comparison_values_are_direct_assignments(self):
        expected = {
            "PLANNER": '"aapf_birrt*"',
            "SCENE_NAME": '"multi_obstacle_3d_avoidance"',
            "RUNS": '"20"',
            "GOAL_MODE": '"adaptive_obstacle_challenge_region"',
            "SEED": '"17"',
            "PLANNER_RANDOM_SEED": '"7"',
            "EXECUTE": '"false"',
            "GO_HOME_BEFORE_BENCHMARK": '"true"',
            "ENABLE_RVIZ": '"false"',
            "BENCHMARK_CASE_ID": '"multi_obstacle_layout04_seed17"',
            "BENCHMARK_CASE_ROOT": '"${HOME}/tmp/trajectory_plan_benchmark_cases"',
            "STATIC_NODE_CSV": '"/tmp/trajectory_plan_test_node_results.csv"',
            "OUTPUT_DIR": '""',
        }
        for name, value in expected.items():
            self.assertRegex(self.script, rf"(?m)^{name}={re.escape(value)}$")
            if name not in ("OUTPUT_DIR", "BENCHMARK_CASE_ROOT"):
                self.assertNotIn(f'{name}="${{', self.script)

    def test_source_uses_only_new_scene_names(self):
        self.assertNotIn("".join(chr(value) for value in (112, 97, 112, 101, 114)), self.script.lower())
        self.assertIn("multi_obstacle_3d_avoidance", self.script)
        self.assertIn("multi_obstacle_layout04_seed17", self.script)


if __name__ == "__main__":
    unittest.main()
