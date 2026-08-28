#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from motion_planning_node_sim import MotionPlanningNodeSim  # noqa: E402
from planning_benchmark import select_farthest_goals  # noqa: E402


class AdaptiveGoalGeometryTest(unittest.TestCase):
    def setUp(self):
        self.node = object.__new__(MotionPlanningNodeSim)
        self.node.active_obstacles = [
            {
                "name": "left",
                "shape": "cylinder",
                "position": (0.25, 0.32, 0.25),
                "radius": 0.055,
                "height": 0.34,
            },
            {
                "name": "goal",
                "shape": "sphere",
                "position": (0.42, -0.10, 0.36),
                "radius": 0.060,
            },
            {
                "name": "lower",
                "shape": "sphere",
                "position": (0.25, 0.10, 0.17),
                "radius": 0.040,
            },
            {
                "name": "right",
                "shape": "box",
                "position": (0.46, -0.12, 0.20),
                "size": (0.08, 0.08, 0.16),
            },
        ]
        self.node.planning_scene_obstacle_padding_m = 0.03
        self.node.benchmark_goal_corridor_clearance_max_m = 0.10
        self.node.scene_name = "multi_obstacle_3d_avoidance"
        self.node.benchmark_goal_mode = "adaptive_obstacle_challenge_region"
        self.node.benchmark_goal_seed = 17

    def test_central_goal_is_adaptive_challenge(self):
        goal = (0.25, 0.11, 0.30)
        metrics = self.node._adaptive_challenge_metrics(
            goal, start_xyz=(0.40, 0.20, 0.20)
        )
        clearance = self.node._distance_to_obstacle_surface(
            goal, self.node.active_obstacles
        )

        self.assertTrue(metrics["accepted"])
        self.assertGreaterEqual(metrics["angular_coverage_deg"], 180.0)
        self.assertLessEqual(metrics["corridor_min_clearance_m"], 0.10)
        self.assertGreaterEqual(clearance, 0.06)
        self.assertLessEqual(clearance, 0.14)

    def test_only_adaptive_goal_mode_is_accepted(self):
        self.assertEqual(
            self.node._normalize_benchmark_goal_mode("adaptive"),
            "adaptive_obstacle_challenge_region",
        )
        self.assertEqual(
            self.node._normalize_benchmark_goal_mode(
                "adaptive_obstacle_challenge_region"
            ),
            "adaptive_obstacle_challenge_region",
        )
        for removed_mode in (
            "fixed",
            "random_obstacle_envelope",
            "random_pose_goal_region",
        ):
            with self.assertRaisesRegex(ValueError, "仅支持"):
                self.node._normalize_benchmark_goal_mode(removed_mode)

    def test_scene_config_keeps_start_pose_without_goal_pose(self):
        scene_file = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "scenes"
            / "pathplanning_scenes_params.yaml"
        )
        scenes = yaml.safe_load(scene_file.read_text(encoding="utf-8"))["scenes"]
        for scene in scenes.values():
            benchmark = scene.get("benchmark", {})
            self.assertIn("start_pose", benchmark)
            self.assertNotIn("goal_pose", benchmark)

    def test_layout_signature_changes_with_obstacle_position(self):
        original = self.node._obstacle_signature()
        self.node.active_obstacles[0]["position"] = (0.26, 0.32, 0.25)
        self.assertNotEqual(original, self.node._obstacle_signature())

    def test_shared_goals_reject_changed_layout(self):
        goal = ((0.25, 0.11, 0.30), (0.0, -180.0, 0.0))
        self.node.benchmark_repetitions = 1
        self.node._goal_is_valid_for_benchmark = lambda *_args: True
        with tempfile.TemporaryDirectory() as tmp_dir:
            goal_file = str(Path(tmp_dir) / "goals.csv")
            self.node._write_generated_goals_csv(
                [goal], goal_file, start_xyz=(0.40, 0.20, 0.20)
            )
            loaded = self.node._read_generated_goals_csv(
                goal_file,
                start_xyz=(0.40, 0.20, 0.20),
                expected_goal_rpy=(0.0, -180.0, 0.0),
            )
            self.assertEqual(loaded, [goal])

            self.node.active_obstacles[0]["position"] = (0.26, 0.32, 0.25)
            with self.assertRaisesRegex(ValueError, "障碍物布局签名"):
                self.node._read_generated_goals_csv(
                    goal_file,
                    start_xyz=(0.40, 0.20, 0.20),
                    expected_goal_rpy=(0.0, -180.0, 0.0),
                )

    def test_stratified_goals_are_deterministic_and_separated(self):
        validator = lambda _point: (True, "")
        first, diagnostics = select_farthest_goals(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 20, 512, 17, 0.04, validator
        )
        second, _ = select_farthest_goals(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 20, 512, 17, 0.04, validator
        )
        self.assertEqual(first, second)
        self.assertEqual(diagnostics["reachable"], 512)
        for index, point in enumerate(first):
            for other in first[index + 1:]:
                self.assertGreaterEqual(sum((a - b) ** 2 for a, b in zip(point, other)) ** 0.5, 0.04)

    def test_stratified_goal_failure_reports_rejections(self):
        with self.assertRaisesRegex(ValueError, "geometry_rejected=64"):
            select_farthest_goals(
                (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 20, 64, 17, 0.04,
                lambda _point: (False, "geometry"),
            )


if __name__ == "__main__":
    unittest.main()
