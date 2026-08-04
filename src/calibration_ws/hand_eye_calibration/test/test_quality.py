from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

_PACKAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

from hand_eye_calibration.collector.auto_calibration_collector import (
    AutoCalibrationCollector,
)
from hand_eye_calibration.collector.quality import CameraInfoState, camera_model_metrics


class _TransformBroadcaster:
    def __init__(self):
        self.transforms = []

    def sendTransform(self, transform):
        self.transforms.append(transform)


class _VisionGate:
    def __init__(self, camera_info):
        self._camera_info = camera_info

    def observation_quality(self, *_args, **_kwargs):
        return True, "observation quality ok"

    def camera_info_snapshot(self):
        return self._camera_info


class PublishMarkerObservationTest(unittest.TestCase):
    def test_publishes_observation_as_camera_marker_transform(self):
        collector = object.__new__(AutoCalibrationCollector)
        collector.frames_config = SimpleNamespace(
            tracking_base_frame="camera_color_optical_frame",
            tracking_marker_frame="calibration_aruco",
        )
        collector._marker_broadcaster = broadcaster = _TransformBroadcaster()
        observation = SimpleNamespace(
            image_stamp_ns=12_345_678_901,
            tvec=(0.1, -0.2, 0.3),
            rvec=(0.0, 0.0, math.pi / 2.0),
        )

        collector._publish_marker_observation(observation)

        self.assertEqual(len(broadcaster.transforms), 1)
        transform = broadcaster.transforms[0]
        self.assertEqual(transform.header.frame_id, "camera_color_optical_frame")
        self.assertEqual(transform.child_frame_id, "calibration_aruco")
        self.assertEqual(transform.header.stamp.sec, 12)
        self.assertEqual(transform.header.stamp.nanosec, 345_678_901)
        self.assertAlmostEqual(transform.transform.translation.x, 0.1)
        self.assertAlmostEqual(transform.transform.translation.y, -0.2)
        self.assertAlmostEqual(transform.transform.translation.z, 0.3)
        self.assertAlmostEqual(transform.transform.rotation.x, 0.0)
        self.assertAlmostEqual(transform.transform.rotation.y, 0.0)
        self.assertAlmostEqual(transform.transform.rotation.z, math.sqrt(0.5))
        self.assertAlmostEqual(transform.transform.rotation.w, math.sqrt(0.5))

    def test_truly_face_on_ippe_remains_rejected_without_unambiguous_window_frames(self):
        collector = object.__new__(AutoCalibrationCollector)
        collector.sampling_config = SimpleNamespace(marker_size_m=0.07, ippe_ambiguity_abs_gap_px=0.05, ippe_ambiguity_max_ratio=1.10)
        info = CameraInfoState(
            width=1280, height=720,
            fx=907.7698364257812, fy=907.7734985351562,
            cx=648.0337524414062, cy=360.25384521484375,
            k=(907.7698364257812, 0.0, 648.0337524414062,
               0.0, 907.7734985351562, 360.25384521484375,
               0.0, 0.0, 1.0),
        )
        half = collector.sampling_config.marker_size_m * 0.5
        object_points = np.asarray(
            ((-half, half, 0.0), (half, half, 0.0),
             (half, -half, 0.0), (-half, -half, 0.0)), dtype=np.float32,
        )
        corners, _ = cv2.projectPoints(
            object_points, R.identity().as_rotvec(), (0.0, 0.0, 0.3975),
            np.asarray(info.k, dtype=float).reshape(3, 3), np.zeros(5),
        )
        corners = corners.reshape(4, 2)
        rvec, tvec, rms, alternative_rms, ambiguous = collector._estimate_marker_pose(corners, info)
        self.assertTrue(ambiguous)
        self.assertLess(rms, 0.01)
        self.assertLess(alternative_rms, 0.20)
        observation = collector._build_aruco_observation(
            corners, info, rvec, tvec, 1, 0.0,
            pnp_ambiguous=ambiguous,
            ippe_absolute_gap_px=alternative_rms - rms,
            ippe_error_ratio=alternative_rms / max(rms, 1.0e-4),
        )
        session = SimpleNamespace(
            vision_gate=_VisionGate(info),
            sampling_cfg=SimpleNamespace(
                marker_size_m=0.07,
                pnp_reprojection_rms_max_px=2.0,
                pnp_reprojection_max_corner_px=3.0,
            ),
        )
        initial_ok, initial_note, _ = camera_model_metrics(
            session, observation, reject_pnp_ambiguity=False,
        )
        strict_ok, strict_note, _ = camera_model_metrics(session, observation)
        self.assertTrue(initial_ok, initial_note)
        self.assertFalse(strict_ok)
        self.assertIn("non-ambiguous frames", strict_note)
        window_ok, _, _ = camera_model_metrics(
            session, observation,
            stable_metrics=SimpleNamespace(non_ambiguous_frame_count=3),
        )
        self.assertTrue(window_ok)
        session.sampling_cfg.ippe_min_non_ambiguous_frames = 0
        zero_requirement_ok, zero_requirement_note, _ = camera_model_metrics(
            session, observation,
            stable_metrics=SimpleNamespace(non_ambiguous_frame_count=0),
        )
        self.assertTrue(zero_requirement_ok, zero_requirement_note)

    def test_wait_for_start_uses_event_not_a_second_stdin_poll(self):
        collector = object.__new__(AutoCalibrationCollector)
        collector._start_requested = __import__("threading").Event()
        collector._start_requested.set()
        collector._stop_collection_requested = __import__("threading").Event()
        collector.motion_config = SimpleNamespace(start_wait_poll_period=0.1)
        collector._should_exit = lambda: False
        collector._clear_collection_stop = lambda: None
        collector.get_logger = lambda: SimpleNamespace(info=lambda *_: None)
        collector.poll_keyboard_once = lambda: self.fail("wait loop must not poll stdin")
        with patch("hand_eye_calibration.collector.auto_calibration_collector.rclpy.ok", return_value=True):
            self.assertTrue(collector._wait_for_start_request())

    def test_keyboard_polling_cannot_crash_collector_when_select_is_unavailable(self):
        collector = object.__new__(AutoCalibrationCollector)
        warnings = []
        collector.get_logger = lambda: SimpleNamespace(warn=warnings.append)
        with patch("hand_eye_calibration.collector.auto_calibration_collector.sys.stdin", SimpleNamespace(isatty=lambda: True)), \
             patch("hand_eye_calibration.collector.auto_calibration_collector.select.select", None):
            collector.poll_keyboard_once()
        self.assertEqual(len(warnings), 1)
        self.assertIn("Keyboard polling is unavailable", warnings[0])


if __name__ == "__main__":
    unittest.main()
