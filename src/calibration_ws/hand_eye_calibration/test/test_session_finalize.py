from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest

from scipy.spatial.transform import Rotation as R


_COLLECTOR_DIR = Path(__file__).parents[1] / "hand_eye_calibration" / "collector"
_collector_package = ModuleType("hand_eye_calibration.collector")
_collector_package.__path__ = [str(_COLLECTOR_DIR)]
sys.modules.setdefault("hand_eye_calibration.collector", _collector_package)

from hand_eye_calibration.collector.geometry import CollectorGeometry
from hand_eye_calibration.collector.session_finalize import log_saved_calibration


def _transform(xyz, rpy_deg=(0.0, 0.0, 0.0)):
    q = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
    return SimpleNamespace(
        translation=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
        rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
    )


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []
        self.warn_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)

    def warn(self, message):
        self.warn_messages.append(message)


class _Session:
    geometry = CollectorGeometry

    def __init__(self, tracking_T_camera_link, mount_T_tracking=None, mount_T_camera_link=None):
        self.logger = _Logger()
        self.tracking_T_camera_link = tracking_T_camera_link
        self.mount_T_tracking = mount_T_tracking
        self.mount_T_camera_link = mount_T_camera_link
        self.lookup_calls = []

    def _logger(self):
        return self.logger

    def _current_transform(self, target_frame, source_frame):
        self.lookup_calls.append((target_frame, source_frame))
        if (target_frame, source_frame) == ("grasp_frame", "camera_color_optical_frame"):
            return self.mount_T_tracking
        if (target_frame, source_frame) == ("grasp_frame", "camera_link"):
            return self.mount_T_camera_link
        return self.tracking_T_camera_link


def _calibration():
    return SimpleNamespace(
        parameters=SimpleNamespace(
            robot_effector_frame="grasp_frame",
            tracking_base_frame="camera_color_optical_frame",
        ),
        transform=_transform((1.0, 2.0, 3.0)),
    )


class LogSavedCalibrationTest(unittest.TestCase):
    def test_logs_file_and_composed_camera_link_pose(self):
        session = _Session(
            CollectorGeometry.transform_to_matrix(_transform((0.1, -0.2, 0.3), (0.0, 0.0, 90.0))),
            CollectorGeometry.transform_to_matrix(_transform((1.003, 2.0, 3.0), (0.0, 0.0, 2.0))),
            CollectorGeometry.transform_to_matrix(_transform((1.103, 1.8, 3.3), (0.0, 0.0, 92.0))),
        )
        content = "parameters:\n  name: robot_calibration\ntransform:\n  translation: {}\n"
        with TemporaryDirectory() as directory:
            filepath = Path(directory) / "robot_calibration.calib"
            filepath.write_text(content, encoding="utf-8")
            log_saved_calibration(session, _calibration(), str(filepath))

        output = "\n".join(session.logger.info_messages)
        self.assertIn(content, output)
        self.assertIn(
            "grasp_frame -> camera_color_optical_frame: xyz,rx,ry,rz="
            "(1.000000, 2.000000, 3.000000, 0.000, 0.000, 0.000) [m, deg]",
            output,
        )
        self.assertIn(
            "grasp_frame -> camera_link: xyz,rx,ry,rz="
            "(1.100000, 1.800000, 3.300000, 0.000, 0.000, 90.000) [m, deg]",
            output,
        )
        self.assertIn(
            "Ground-truth comparison (grasp_frame -> camera_color_optical_frame): "
            "translation_error=3.00mm, rotation_error=2.00deg",
            output,
        )
        self.assertIn(
            "Ground-truth comparison (grasp_frame -> camera_link): "
            "translation_error=3.00mm, rotation_error=2.00deg",
            output,
        )
        self.assertEqual(
            session.lookup_calls,
            [
                ("grasp_frame", "camera_color_optical_frame"),
                ("camera_color_optical_frame", "camera_link"),
                ("grasp_frame", "camera_link"),
            ],
        )
        self.assertEqual(session.logger.error_messages, [])

    def test_reports_missing_file_or_tf_without_raising(self):
        session = _Session(None)
        log_saved_calibration(session, _calibration(), "/missing/robot_calibration.calib")

        errors = "\n".join(session.logger.error_messages)
        self.assertIn("Cannot read saved calibration file", errors)
        self.assertIn("required TF camera_color_optical_frame -> camera_link is unavailable", errors)
        self.assertIn("grasp_frame -> camera_color_optical_frame", "\n".join(session.logger.info_messages))


if __name__ == "__main__":
    unittest.main()
