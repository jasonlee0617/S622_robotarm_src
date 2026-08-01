import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from easy_handeye2 import (
    latest_snapshot_filepath,
    load_filepath,
    next_snapshot_id,
    resolve_storage_directory,
    snapshot_filepath,
)
from easy_handeye2.handeye_calibration import load_calibration, save_calibration
from easy_handeye2.handeye_server import HandeyeServer
from easy_handeye2_msgs.msg import HandeyeCalibration


class StoragePathsTest(unittest.TestCase):
    def test_storage_directory_expands_home(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"HOME": directory}):
            self.assertEqual(resolve_storage_directory("$HOME/calib"), Path(directory) / "calib")

    def test_latest_timestamped_file_and_explicit_name(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            older = directory / "robot_calibration_20260622_232125.calib"
            newer = directory / "robot_calibration_20260623_120000.calib"
            older.touch()
            newer.touch()

            self.assertEqual(latest_snapshot_filepath(directory, "robot_calibration", ".calib"), newer)
            self.assertEqual(load_filepath(directory, "robot_calibration", ".calib", timestamped=True), newer)
            self.assertEqual(
                load_filepath(
                    directory, "robot_calibration_20260622_232125", ".calib", timestamped=True
                ),
                older,
            )

    def test_snapshot_id_never_overwrites_existing_pair(self):
        clock = type("Clock", (), {
            "now": staticmethod(lambda: type("Now", (), {
                "strftime": staticmethod(lambda _: "20260801_153045")
            })())
        })
        with TemporaryDirectory() as directory, patch("easy_handeye2.datetime", clock):
            directory = Path(directory)
            snapshot_filepath(directory, "robot_calibration", ".calib", "20260801_153045").touch()
            self.assertEqual(next_snapshot_id(directory, "robot_calibration"), "20260801_153045_01")

    def test_calibration_round_trip_uses_timestamped_path(self):
        calibration = HandeyeCalibration()
        calibration.parameters.name = "robot_calibration"
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            saved = save_calibration(
                calibration, storage_directory=str(directory), snapshot_id="20260801_153045"
            )

            self.assertEqual(saved, directory / "robot_calibration_20260801_153045.calib")
            self.assertEqual(
                load_calibration("robot_calibration", storage_directory=str(directory)).parameters.name,
                "robot_calibration",
            )

    def test_save_calibration_also_persists_matching_samples(self):
        calibration = HandeyeCalibration()
        calibration.parameters.name = "robot_calibration"

        with TemporaryDirectory() as directory:
            directory = Path(directory)

            class Sampler:
                def save_samples(self, snapshot_id):
                    path = directory / f"robot_calibration_{snapshot_id}.samples"
                    path.write_text("samples: []\n")
                    return path

            server = SimpleNamespace(
                last_calibration=calibration,
                storage_directory=str(directory),
                sampler=Sampler(),
                _pending_snapshot=lambda: "20260801_153045",
                _clear_pending_snapshot=lambda: None,
                get_logger=lambda: SimpleNamespace(info=lambda _: None, error=lambda _: None),
            )
            response = SimpleNamespace(success=False, filepath=SimpleNamespace(data=""))

            HandeyeServer.save_calibration(server, None, response)

            self.assertTrue(response.success)
            self.assertEqual(
                Path(response.filepath.data),
                directory / "robot_calibration_20260801_153045.calib",
            )
            self.assertTrue((directory / "robot_calibration_20260801_153045.samples").is_file())


if __name__ == "__main__":
    unittest.main()
