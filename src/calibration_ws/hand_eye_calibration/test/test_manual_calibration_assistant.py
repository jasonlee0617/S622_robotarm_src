import threading
import unittest
import runpy
import math
from pathlib import Path
from types import SimpleNamespace

from scipy.spatial.transform import Rotation as R

from hand_eye_calibration.solver import CalibrationSample, TransformMatrix
from hand_eye_calibration.config import CalibrationType
from hand_eye_calibration.manual_calibration_assistant import (
    GUIDANCE,
    MINIMUM_SAMPLES,
    TARGET_SAMPLES,
    ManualCalibrationAssistant,
    ManualSessionState,
    coordinate_frame_markers,
    guidance_for,
    guidance_pose_metrics,
    guidance_readiness,
)


def _record(index):
    pose = TransformMatrix(R.identity(), (float(index) * 0.01, 0.0, 0.0))
    return CalibrationSample(index, (float(index),) * 6, pose, pose)


class ManualSessionStateTests(unittest.TestCase):
    def test_assistant_has_no_collector_node_or_auto_yaml_dependency(self):
        source = (Path(__file__).resolve().parents[1] / "hand_eye_calibration" / "manual_calibration_assistant.py").read_text()
        for forbidden in ("AutoCalibrationCollector", "collector.auto_calibration_collector", "auto_calibration_collector_params.yaml"):
            self.assertNotIn(forbidden, source)

    def test_assistant_source_is_importable_as_an_installed_script(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "hand_eye_calibration"
            / "manual_calibration_assistant.py"
        )
        symbols = runpy.run_path(str(script), run_name="manual_assistant_script_test")

        self.assertIn("ManualCalibrationAssistant", symbols)

    def test_invalid_latest_blocks_until_easy_removes_it(self):
        state = ManualSessionState()
        self.assertTrue(state.begin_validation(1)[0])
        state.reject("marker is too close to the image edge")

        self.assertFalse(state.status()["can_take"])
        self.assertFalse(state.status()["can_save"])
        self.assertFalse(state.begin_validation(2)[0])
        self.assertTrue(state.remove_after_easy(0)[0])
        self.assertTrue(state.status()["can_take"])

    def test_remove_is_latest_only_and_removing_root_resets(self):
        state = ManualSessionState()
        for index in (1, 2):
            self.assertTrue(state.begin_validation(index)[0])
            state.accept(_record(index))

        self.assertTrue(state.remove_after_easy(1)[0])
        self.assertEqual([record.waypoint_index for record in state.records], [1])
        ok, message = state.remove_after_easy(0)
        self.assertTrue(ok)
        self.assertIn("root", message)
        self.assertEqual(state.records, [])

    def test_preexisting_easy_samples_must_be_removed_to_empty(self):
        state = ManualSessionState()
        self.assertFalse(state.begin_validation(3)[0])
        self.assertTrue(state.status()["blocked"])
        self.assertTrue(state.remove_after_easy(2)[0])
        self.assertTrue(state.status()["blocked"])
        self.assertTrue(state.remove_after_easy(1)[0])
        self.assertTrue(state.remove_after_easy(0)[0])
        self.assertFalse(state.status()["blocked"])

    def test_preexisting_easy_samples_can_be_cleared_before_first_take(self):
        state = ManualSessionState()
        self.assertTrue(state.remove_after_easy(2)[0])
        self.assertTrue(state.remove_after_easy(1)[0])
        ok, message = state.remove_after_easy(0)
        self.assertTrue(ok)
        self.assertIn("root", message)

    def test_save_allows_fifteen_to_twenty_synchronized_samples(self):
        state = ManualSessionState()
        for index in range(1, MINIMUM_SAMPLES + 1):
            self.assertTrue(state.begin_validation(index)[0])
            state.accept(_record(index))
        self.assertTrue(state.status()["can_save"])
        self.assertFalse(state.begin_save(MINIMUM_SAMPLES - 1)[0])
        self.assertTrue(state.begin_save(MINIMUM_SAMPLES)[0])
        state.finish_save(True, "saved")
        self.assertTrue(state.status()["saved"])
        self.assertFalse(state.status()["can_take"])

    def test_stop_never_enables_save(self):
        state = ManualSessionState()
        state.records = [_record(index) for index in range(1, TARGET_SAMPLES + 1)]
        state.stop("Ctrl-C")
        self.assertFalse(state.begin_save(TARGET_SAMPLES)[0])
        self.assertFalse(state.status()["can_save"])
        self.assertIn("未保存", state.status()["message"])

    def test_guidance_is_one_fixed_root_relative_table(self):
        self.assertEqual(len(GUIDANCE), TARGET_SAMPLES)
        self.assertEqual([guide.index for guide in GUIDANCE], list(range(1, TARGET_SAMPLES + 1)))
        self.assertEqual(GUIDANCE[0].category, "ROOT")
        self.assertEqual(sum("Z 滚转" in guide.category for guide in GUIDANCE), 2)
        self.assertEqual(sum("X 倾斜" in guide.category for guide in GUIDANCE), 4)
        self.assertEqual(sum("Y 倾斜" in guide.category for guide in GUIDANCE), 4)
        self.assertEqual(sum("XY 复合" in guide.category for guide in GUIDANCE), 4)
        self.assertEqual(sum("横向" in guide.category for guide in GUIDANCE), 4)
        self.assertIn("相机 depth", GUIDANCE[-1].instruction)
        self.assertIn("固定 ±Z", GUIDANCE[-1].translation_hint)
        self.assertEqual([guide.angle_deg for guide in GUIDANCE[1:3]], [20.0, 20.0])
        self.assertEqual([guide.angle_deg for guide in GUIDANCE[3:7]], [15.0] * 4)
        self.assertEqual([guide.angle_deg for guide in GUIDANCE[7:11]], [28.0] * 4)

    def test_guidance_uses_type_specific_local_frame_words(self):
        eye_in_hand = guidance_for(CalibrationType.EYE_IN_HAND)
        eye_on_base = guidance_for(CalibrationType.EYE_ON_BASE)
        self.assertIn("末端", eye_in_hand[1].instruction)
        self.assertIn("标定板/腕部", eye_on_base[1].instruction)

    def test_guidance_metrics_and_coordinate_frames(self):
        root = R.identity()
        current = R.from_rotvec((math.radians(20.0), 0.0, 0.0))
        target, actual, error = guidance_pose_metrics(GUIDANCE[7], root, current)
        self.assertEqual(target, "+X tilt 28°")
        self.assertEqual(actual, "+X tilt +20.0°")
        self.assertAlmostEqual(error, 8.0, places=6)

        markers = coordinate_frame_markers(
            frame_id="base_link", stamp=None, namespace="test", start_id=1,
            pose=TransformMatrix(R.identity(), (0.0, 0.0, 0.0)), alpha=0.35,
        )
        self.assertEqual(len(markers), 3)
        self.assertAlmostEqual(markers[0].color.r, 1.0)
        self.assertAlmostEqual(markers[1].color.g, 1.0)
        self.assertAlmostEqual(markers[2].color.b, 1.0)
        self.assertAlmostEqual(markers[0].color.a, 0.35)

        self.assertEqual(guidance_readiness(True, 5.0, "ok"), "READY ✓")
        self.assertEqual(guidance_readiness(True, 5.1, "ok"), "VISION OK / POSE ADJUST")
        self.assertEqual(guidance_readiness(False, 0.0, "no marker"), "VISION ADJUST: no marker")

    def test_validate_keeps_type_adjusted_robot_pose_from_collector(self):
        assistant = object.__new__(ManualCalibrationAssistant)
        assistant.state = ManualSessionState()
        assistant._state_lock = threading.Lock()
        robot = TransformMatrix(R.identity(), (0.3, -0.2, 0.1))
        tracking = TransformMatrix(R.identity(), (0.0, 0.0, 0.4))
        assistant._easy_count = lambda: 1
        assistant._wait_for_joint_stationary = lambda: (True, "stationary")
        assistant._stable_sample = lambda: (robot, tracking, "stable")
        assistant._is_diverse = lambda pose: (pose is robot, "diverse")
        assistant._joint_snapshot_deg = lambda: (1.0,) * 6
        assistant._publish_guidance = lambda: None
        assistant.get_logger = lambda: SimpleNamespace(info=lambda *_: None, warn=lambda *_: None)
        response = SimpleNamespace(success=False, message="")

        ManualCalibrationAssistant._validate_service(assistant, None, response)

        self.assertTrue(response.success, response.message)
        self.assertIs(assistant.state.records[0].robot_pose, robot)
        self.assertIn("h+Enter 返回 root", response.message)

    def test_return_root_uses_one_motion_executor_call(self):
        assistant = object.__new__(ManualCalibrationAssistant)
        assistant.state = ManualSessionState()
        assistant.state.records = [_record(1)]
        assistant._state_lock = threading.Lock()
        calls = []
        assistant._motion = SimpleNamespace(move_to_joints=lambda *args, **kwargs: calls.append((args, kwargs)))
        assistant._motion_thread = None
        assistant.motion_config = SimpleNamespace(
            max_velocity=0.3,
            max_acceleration=0.3,
            allowed_planning_time=5.0,
            allowed_start_tolerance=0.1,
        )
        assistant.get_logger = lambda: SimpleNamespace(warn=lambda *_: None, error=lambda *_: None)

        assistant._return_root()
        assistant._motion_thread.join(timeout=1.0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0][0][0]), 6)


if __name__ == "__main__":
    unittest.main()
