from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest


_PACKAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

from hand_eye_calibration.collector.auto_calibration_collector import (
    AutoCalibrationCollector,
)


class _TransformBroadcaster:
    def __init__(self):
        self.transforms = []

    def sendTransform(self, transform):
        self.transforms.append(transform)


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


if __name__ == "__main__":
    unittest.main()
