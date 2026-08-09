import unittest

from hand_eye_calibration.vision import (
    ArucoObservation,
    CameraInfoState,
    VisionQualityGate,
    median_marker_corners,
)


def _info():
    return CameraInfoState(640, 480, (500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0), (), "camera_color_optical_frame")


def _observation(index=0, *, margin=60.0, side=90.0, depth=0.4):
    return ArucoObservation(
        receipt_time=float(index), center_px=(320.0 + index * 0.1, 240.0),
        corners_px=((275.0, 195.0), (365.0, 195.0), (365.0, 285.0), (275.0, 285.0)),
        side_px=side, margin_px=margin, tvec=(0.0, 0.0, depth), rvec=(0.0, 0.0, 0.1),
    )


class VisionGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = VisionQualityGate(
            marker_distance_min_m=0.20, marker_distance_max_m=0.80,
            minimum_corner_margin_px=60.0, minimum_marker_side_px=90.0,
            stable_frames=10, maximum_center_std_px=4.0,
            maximum_marker_depth_std_m=0.003, maximum_marker_angle_std_deg=0.8,
            logger_warn=lambda _message: None,
        )
        self.gate.update_camera_info(_info())

    def test_ten_consecutive_frames_pass_and_failure_clears(self):
        for index in range(10):
            self.assertTrue(self.gate.record_success(_observation(index)))
        frames, note = self.gate.stable_window()
        self.assertEqual(len(frames), 10, note)
        self.gate.record_failure("no marker")
        frames, _ = self.gate.stable_window()
        self.assertIsNone(frames)

    def test_all_inclusive_image_boundaries_pass(self):
        self.assertTrue(self.gate.record_success(_observation(margin=60.0, side=90.0, depth=0.20)))
        self.assertTrue(self.gate.record_success(_observation(margin=60.0, side=90.0, depth=0.80)))
        self.assertFalse(self.gate.record_success(_observation(margin=59.9)))
        self.assertFalse(self.gate.record_success(_observation(side=89.9)))

    def test_latest_observation_reports_live_quality_without_stable_window(self):
        observation = _observation(margin=80.0, side=100.0, depth=0.40)
        self.assertTrue(self.gate.record_success(observation))
        latest, accepted, note = self.gate.latest_observation()
        self.assertEqual(latest, observation)
        self.assertTrue(accepted, note)

        self.gate.record_failure("no marker")
        latest, accepted, note = self.gate.latest_observation()
        self.assertIsNone(latest)
        self.assertFalse(accepted)
        self.assertEqual(note, "no marker")

    def test_median_corners(self):
        frames = tuple(_observation(index) for index in range(10))
        self.assertEqual(median_marker_corners(frames)[0], (275.0, 195.0))
        self.assertEqual(_info().camera_matrix().shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
