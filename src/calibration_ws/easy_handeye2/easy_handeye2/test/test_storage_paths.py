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
    typed_snapshot_id,
)
from easy_handeye2.handeye_calibration import load_calibration, save_calibration
from easy_handeye2.handeye_sampler import HandeyeSampler
from easy_handeye2.handeye_server import HandeyeServer
from easy_handeye2_msgs.msg import HandeyeCalibration, HandeyeCalibrationParameters, SampleList


class StoragePathsTest(unittest.TestCase):
    def test_storage_directory_expands_home(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"HOME": directory}):
            self.assertEqual(resolve_storage_directory("$HOME/calib"), Path(directory) / "calib")

    def test_latest_timestamped_file_and_explicit_name(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            older = directory / "robot_calibration_20260622_232125_eye_in_hand.calib"
            newer = directory / "robot_calibration_20260623_120000_eye_in_hand.calib"
            newest = directory / "robot_calibration_20260623_120000_01_eye_in_hand.calib"
            older.touch()
            newer.touch()
            newest.touch()

            self.assertEqual(latest_snapshot_filepath(directory, "robot_calibration", ".calib"), newest)
            self.assertEqual(load_filepath(directory, "robot_calibration", ".calib", timestamped=True), newest)
            self.assertEqual(
                load_filepath(
                    directory,
                    "robot_calibration_20260622_232125_eye_in_hand",
                    ".calib",
                    timestamped=True,
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
            snapshot_filepath(
                directory,
                "robot_calibration",
                ".calib",
                "20260801_153045_eye_in_hand",
            ).touch()
            self.assertEqual(
                next_snapshot_id(directory, "robot_calibration", "eye_in_hand"),
                "20260801_153045_01_eye_in_hand",
            )

    def test_snapshot_type_is_validated_and_not_duplicated(self):
        self.assertEqual(
            typed_snapshot_id("20260801_153045", "eye_on_base"),
            "20260801_153045_eye_on_base",
        )
        self.assertEqual(
            typed_snapshot_id("20260801_153045_eye_in_hand", "eye_in_hand"),
            "20260801_153045_eye_in_hand",
        )
        with self.assertRaises(ValueError):
            typed_snapshot_id("20260801_153045", "other")
        with self.assertRaises(ValueError):
            typed_snapshot_id("20260801_153045_eye_on_base", "eye_in_hand")

    def test_calibration_round_trip_uses_timestamped_path(self):
        calibration = HandeyeCalibration()
        calibration.parameters.name = "custom_name"
        calibration.parameters.calibration_type = "eye_in_hand"
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            saved = save_calibration(
                calibration, storage_directory=str(directory), snapshot_id="20260801_153045"
            )

            self.assertEqual(
                saved,
                directory / "robot_calibration_20260801_153045_eye_in_hand.calib",
            )
            self.assertEqual(
                load_calibration("robot_calibration", storage_directory=str(directory)).parameters.name,
                "custom_name",
            )

    def test_save_calibration_also_persists_matching_samples(self):
        calibration = HandeyeCalibration()
        calibration.parameters.name = "robot_calibration"
        calibration.parameters.calibration_type = "eye_on_base"

        with TemporaryDirectory() as directory:
            directory = Path(directory)

            class Sampler:
                def save_samples(self, snapshot_id):
                    path = directory / f"robot_calibration_{snapshot_id}.samples"
                    path.write_text("samples: []\n", encoding="utf-8")
                    return path

            server = SimpleNamespace(
                last_calibration=calibration,
                storage_directory=str(directory),
                sampler=Sampler(),
                _pending_snapshot=lambda: "20260801_153045_eye_on_base",
                _clear_pending_snapshot=lambda: None,
                get_logger=lambda: SimpleNamespace(info=lambda _: None, error=lambda _: None),
            )
            response = SimpleNamespace(success=False, filepath=SimpleNamespace(data=""))

            HandeyeServer.save_calibration(server, None, response)

            self.assertTrue(response.success)
            self.assertEqual(
                Path(response.filepath.data),
                directory / "robot_calibration_20260801_153045_eye_on_base.calib",
            )
            self.assertTrue(
                (directory / "robot_calibration_20260801_153045_eye_on_base.samples").is_file()
            )

    def test_timestamped_calibration_refuses_overwrite(self):
        calibration = HandeyeCalibration()
        calibration.parameters.name = "robot_calibration"
        calibration.parameters.calibration_type = "eye_in_hand"
        with TemporaryDirectory() as directory:
            save_calibration(calibration, directory, "20260801_153045")
            with self.assertRaises(FileExistsError):
                save_calibration(calibration, directory, "20260801_153045")

    def test_timestamped_samples_use_type_suffix_and_refuse_overwrite(self):
        with TemporaryDirectory() as directory:
            sampler = HandeyeSampler.__new__(HandeyeSampler)
            sampler.storage_directory = directory
            sampler.handeye_parameters = HandeyeCalibrationParameters(
                name="robot_calibration",
                calibration_type="eye_on_base",
            )
            sampler.samples = SampleList(parameters=sampler.handeye_parameters)
            saved = sampler.save_samples("20260801_153045")
            self.assertEqual(
                saved,
                Path(directory) / "robot_calibration_20260801_153045_eye_on_base.samples",
            )
            with self.assertRaises(FileExistsError):
                sampler.save_samples("20260801_153045")


if __name__ == "__main__":
    unittest.main()
