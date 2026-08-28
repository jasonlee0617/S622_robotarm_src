from collections import deque
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

from rclpy.clock import ClockType
from scipy.spatial.transform import Rotation as R

from hand_eye_calibration.auto_calibration_collector import AutoCalibrationCollector
from hand_eye_calibration.solver import CalibrationSample, TransformMatrix


class CollectorTests(unittest.TestCase):
    def test_keyboard_timer_uses_steady_time(self):
        timer_calls = []
        fake = SimpleNamespace(
            motion_config=SimpleNamespace(keyboard_poll_period=0.1, step_between_actions=True),
            _start_requested=threading.Event(),
            _keyboard_timer=None,
            _keyboard_clock=None,
            _keyboard_enabled=False,
            _on_start_request=lambda *_args: None,
            declare_parameter=lambda *_args: SimpleNamespace(value=False),
            create_service=lambda *_args, **_kwargs: None,
            create_timer=lambda *args, **kwargs: timer_calls.append((args, kwargs)) or object(),
            poll_keyboard_once=lambda: None,
        )
        stdin = SimpleNamespace(isatty=lambda: True)
        with patch("hand_eye_calibration.auto_calibration_collector.sys.stdin", stdin):
            AutoCalibrationCollector._setup_manual_control(fake)

        self.assertTrue(fake._keyboard_enabled)
        self.assertEqual(timer_calls[0][1]["clock"].clock_type, ClockType.STEADY_TIME)

    def test_keyboard_timer_accepts_launch_non_tty_stdin(self):
        timer_calls = []
        fake = SimpleNamespace(
            motion_config=SimpleNamespace(keyboard_poll_period=0.1, step_between_actions=False),
            _start_requested=threading.Event(),
            _keyboard_timer=None,
            _keyboard_clock=None,
            _keyboard_enabled=False,
            _on_start_request=lambda *_args: None,
            declare_parameter=lambda *_args: SimpleNamespace(value=False),
            create_service=lambda *_args, **_kwargs: None,
            create_timer=lambda *args, **kwargs: timer_calls.append((args, kwargs)) or object(),
            poll_keyboard_once=lambda: None,
        )
        stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 0)
        with patch("hand_eye_calibration.auto_calibration_collector.sys.stdin", stdin):
            AutoCalibrationCollector._setup_manual_control(fake)

        self.assertTrue(fake._keyboard_enabled)
        self.assertEqual(len(timer_calls), 1)

    def test_keyboard_timer_prefers_controlling_terminal(self):
        timer_calls = []
        fake = SimpleNamespace(
            motion_config=SimpleNamespace(keyboard_poll_period=0.1, step_between_actions=False),
            _start_requested=threading.Event(),
            _keyboard_timer=None,
            _keyboard_clock=None,
            _keyboard_enabled=False,
            _on_start_request=lambda *_args: None,
            declare_parameter=lambda *_args: SimpleNamespace(value=False),
            create_service=lambda *_args, **_kwargs: None,
            create_timer=lambda *args, **kwargs: timer_calls.append((args, kwargs)) or object(),
            poll_keyboard_once=lambda: None,
        )
        stdin = SimpleNamespace(isatty=lambda: False, fileno=lambda: 0)
        terminal = SimpleNamespace(isatty=lambda: True, fileno=lambda: 1)
        with patch("hand_eye_calibration.auto_calibration_collector.sys.stdin", stdin):
            with patch("hand_eye_calibration.auto_calibration_collector.open", return_value=terminal):
                AutoCalibrationCollector._setup_manual_control(fake)

        self.assertTrue(fake._keyboard_enabled)
        self.assertIs(fake._keyboard_stream, terminal)
        self.assertEqual(len(timer_calls), 1)

    def test_keyboard_poll_starts_collection_from_non_tty_input(self):
        read_fd, write_fd = os.pipe()
        try:
            fake = SimpleNamespace(
                _keyboard_enabled=True,
                _keyboard_stream=os.fdopen(read_fd, "r"),
                _start_requested=threading.Event(),
                _step_continue=threading.Event(),
                _collection_active=threading.Event(),
                session_state="STANDBY",
            )
            os.write(write_fd, b"\n")
            AutoCalibrationCollector.poll_keyboard_once(fake)
            self.assertTrue(fake._start_requested.is_set())
        finally:
            os.close(write_fd)
            fake._keyboard_stream.close()

    def test_time_base_rejects_mismatched_clock_configuration(self):
        messages = []
        logger = SimpleNamespace(error=messages.append)
        for use_sim_time, topics, expected in (
            (False, [("/clock", ["rosgraph_msgs/msg/Clock"])], False),
            (True, [], False),
            (False, [], True),
            (True, [("/clock", ["rosgraph_msgs/msg/Clock"])], True),
        ):
            fake = SimpleNamespace(
                _use_sim_time=use_sim_time,
                get_topic_names_and_types=lambda topics=topics: topics,
                get_logger=lambda: logger,
            )
            self.assertIs(AutoCalibrationCollector._validate_time_base(fake), expected)

    def test_time_base_errors_explain_the_yaml_fix_and_parameter_syntax(self):
        messages = []
        fake = SimpleNamespace(
            _use_sim_time=False,
            get_topic_names_and_types=lambda: [("/clock", ["rosgraph_msgs/msg/Clock"])],
            get_logger=lambda: SimpleNamespace(error=messages.append),
        )
        self.assertFalse(AutoCalibrationCollector._validate_time_base(fake))
        self.assertIn("auto_calibration_collector_params.yaml", messages[0])
        self.assertIn("--ros-args -p use_sim_time:=true", messages[0])

    def test_moveit_readiness_reports_the_expected_namespace(self):
        messages = []
        fake = SimpleNamespace(
            sampling_config=SimpleNamespace(moveit_ready_timeout=0.0, moveit_ready_poll_interval=0.01),
            motion_config=SimpleNamespace(move_group_ns_fairino="/move_group_fairino"),
            _should_stop=lambda: False,
            _arm=SimpleNamespace(),
            get_logger=lambda: SimpleNamespace(error=messages.append),
        )
        self.assertFalse(AutoCalibrationCollector._wait_for_moveit(fake))
        self.assertIn("/move_group_fairino/plan_kinematic_path", messages[0])

    def test_execution_prechecks_require_root_controller_and_fresh_joint_states(self):
        messages = []
        fake = SimpleNamespace(
            sampling_config=SimpleNamespace(moveit_ready_timeout=0.0, moveit_ready_poll_interval=0.01),
            _should_stop=lambda: False,
            _controller_action_client=SimpleNamespace(server_is_ready=lambda: False),
            get_logger=lambda: SimpleNamespace(error=messages.append),
        )
        self.assertFalse(AutoCalibrationCollector._wait_for_execution_controller(fake))
        self.assertIn("/robot_arm_controller/follow_joint_trajectory", messages[0])

        now = time.monotonic()
        fake._joint_lock = threading.Lock()
        fake._joint_history = deque(((now - 0.10, (0.0,) * 6), (now - 0.02, (0.0,) * 6)))
        self.assertTrue(AutoCalibrationCollector._joint_state_stream_ready(fake))
        fake._joint_history = deque(((now - 1.1, (0.0,) * 6),))
        self.assertFalse(AutoCalibrationCollector._joint_state_stream_ready(fake))

    def test_camera_info_wait_handles_delayed_and_missing_messages(self):
        class Gate:
            def __init__(self, ready_after):
                self.calls, self.ready_after = 0, ready_after

            def camera_info_snapshot(self):
                self.calls += 1
                return SimpleNamespace(ready=self.calls >= self.ready_after)

        config = SimpleNamespace(stable_marker_timeout_sec=0.02)
        motion = SimpleNamespace(start_wait_poll_period=0.001)
        delayed = SimpleNamespace(sampling_config=config, motion_config=motion, vision_gate=Gate(2))
        missing = SimpleNamespace(sampling_config=config, motion_config=motion, vision_gate=Gate(1000))
        self.assertTrue(AutoCalibrationCollector._wait_for_camera_info(delayed))
        self.assertFalse(AutoCalibrationCollector._wait_for_camera_info(missing))

    def test_stationary_window_and_diversity(self):
        now = time.monotonic()
        config = SimpleNamespace(joint_stationary_timeout_sec=0.1, joint_stationary_window_sec=0.30, joint_stationary_max_position_delta_rad=0.0001, minimum_translation_delta_m=0.006, minimum_rotation_delta_deg=3.0)
        fake = SimpleNamespace(
            sampling_config=config, _joint_lock=threading.Lock(),
            _joint_history=deque(((now - 0.29, (0.0,) * 6), (now, (0.00005,) * 6))),
            _should_stop=lambda: False, _accepted=[],
        )
        self.assertTrue(AutoCalibrationCollector._wait_for_joint_stationary(fake)[0])
        pose = TransformMatrix(R.identity(), (0.0, 0.0, 0.0))
        fake._accepted = [CalibrationSample(1, (0.0,) * 6, pose, pose)]
        self.assertFalse(AutoCalibrationCollector._is_diverse(fake, pose)[0])

    def test_sampling_modules_are_flat_at_package_root(self):
        package = Path(__file__).parents[1] / "hand_eye_calibration"
        self.assertFalse((package / "collector").exists())
        self.assertFalse((package / "core").exists())
        for filename in (
            "auto_calibration_collector.py", "manual_calibration_assistant.py",
            "sampling_runtime.py", "config.py", "vision.py", "solver.py",
        ):
            self.assertTrue((package / filename).is_file())

    def test_removed_gate_interfaces_are_absent(self):
        root = Path(__file__).parents[1]
        text = "\n".join(path.read_text() for path in (
            root / "hand_eye_calibration" / "config.py",
            root / "hand_eye_calibration" / "vision.py",
            root / "hand_eye_calibration" / "solver.py",
            root / "hand_eye_calibration" / "auto_calibration_collector.py",
            root / "config" / "auto_calibration_collector_params.yaml",
            root / "CMakeLists.txt",
        ))
        for removed in ("checkpoint", "leave_one_out", "compute_fk", "pnp_reprojection", "ippe_ambiguity", "image_stamp_ns"):
            self.assertNotIn(removed, text)


if __name__ == "__main__":
    unittest.main()
