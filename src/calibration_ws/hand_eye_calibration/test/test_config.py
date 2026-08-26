import unittest
from types import SimpleNamespace

from hand_eye_calibration import config


class ConfigTests(unittest.TestCase):
    def test_waypoints_keep_twenty_slots_and_root(self):
        slots = [",".join(str(value) for value in config.INITIAL_JOINT_DEG)] + ["TODO"] * 19
        parsed = config._parse_joint_waypoints(slots)
        self.assertEqual(len(parsed), 20)
        self.assertEqual(config._waypoint_specs(parsed)[0].joints_deg, config.INITIAL_JOINT_DEG)

    def test_invalid_waypoint_and_limits_are_rejected(self):
        root = ",".join(str(value) for value in config.INITIAL_JOINT_DEG)
        slots = [root] + ["TODO"] * 19
        slots[1] = root
        with self.assertRaises(ValueError):
            config._parse_joint_waypoints(slots)
        for limits in (["-1,1"] * 5, ["1,-1"] * 6, ["nan,1"] * 6):
            with self.assertRaises(ValueError):
                config._parse_joint_limits(limits, 6)

    def test_projection_source_is_fixed_to_p(self):
        self.assertEqual(config._default_joint_limits_deg()[0], "-175,175")
        self.assertEqual(config.JOINT_WAYPOINT_SLOTS, 20)

    def test_calibration_type_is_canonical(self):
        self.assertIs(
            config.normalize_calibration_type("Eye-On-Base"),
            config.CalibrationType.EYE_ON_BASE,
        )
        with self.assertRaises(ValueError):
            config.normalize_calibration_type("unknown")

    def test_source_yaml_loads_twenty_real_slots(self):
        class Node:
            values = {}

            def declare_parameter(self, name, default):
                self.values.setdefault(name, default)

            def get_parameter(self, name):
                return SimpleNamespace(value=self.values[name])

        frames, _motion, sampling = config.load_collector_config(Node())
        self.assertEqual(frames.camera_intrinsics_source, "p")
        self.assertIs(frames.calibration_type, config.CalibrationType.EYE_IN_HAND)
        self.assertEqual(len(sampling.waypoint_specs), 20)
        self.assertEqual((sampling.minimum_samples, sampling.minimum_solution_samples), (15, 14))
        self.assertEqual(sampling.algorithm_names, ("OpenCV/Park", "OpenCV/Horaud"))
        self.assertIsInstance(sampling.ground_truth_check_enabled, bool)
        self.assertEqual(sampling.maximum_eye_on_base_camera_translation_norm_m, 2.0)

    def test_yaml_defines_the_direct_collector_time_source(self):
        self.assertIsInstance(config.yaml_use_sim_time(), bool)

    def test_grouped_yaml_defaults_are_flattened_with_tool0(self):
        automatic = config._load_yaml_defaults()
        manual = config._load_yaml_defaults(
            "manual_calibration_assistant_params.yaml", "manual_calibration_assistant"
        )
        self.assertEqual(automatic["ee_frame"], "tool0")
        self.assertEqual(manual["ee_frame"], "tool0")
        self.assertIn("joint_waypoints_deg", automatic)
        self.assertNotIn("joint_waypoints_deg", manual)
        self.assertFalse(any(isinstance(value, dict) for value in automatic.values()))
        self.assertFalse(any(isinstance(value, dict) for value in manual.values()))

    def test_grouped_yaml_duplicate_leaf_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate grouped parameter: ee_frame"):
            config.flatten_ros_parameters({"frames": {"ee_frame": "tool0"}, "other": {"ee_frame": "x"}})

    def test_manual_config_has_no_waypoints(self):
        class Node:
            values = {}

            def declare_parameter(self, name, default):
                self.values.setdefault(name, default)

            def get_parameter(self, name):
                return SimpleNamespace(value=self.values[name])

        _frames, _motion, sampling = config.load_manual_config(Node())
        self.assertEqual(sampling.waypoint_specs, ())
        self.assertEqual(sampling.joint_limits_deg, ())


if __name__ == "__main__":
    unittest.main()
