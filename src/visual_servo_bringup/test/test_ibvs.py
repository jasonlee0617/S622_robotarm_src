import numpy as np
from pathlib import Path

from visual_servo_bringup.ibvs import (
    clip_twist,
    ibvs_camera_twist,
    interaction_matrix,
    normalize_corners,
)
from visual_servo_bringup.image_servo_timing import feature_timestamp_ns
from visual_servo_bringup.image_servo_config import image_servo_parameters
from visual_servo_bringup.servo.servo_io import ServoIO


PACKAGE = Path(__file__).resolve().parents[1]


class _Stamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


def test_normalized_corner_error_has_zero_command_at_reference():
    camera = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    features = normalize_corners(
        np.array([[280.0, 200.0], [360.0, 200.0], [360.0, 280.0], [280.0, 280.0]]), camera
    )
    twist, error = ibvs_camera_twist(features, features, np.full(4, 0.5), 0.5, 0.03)

    assert np.allclose(error, 0.0)
    assert np.allclose(twist, 0.0)


def test_interaction_matrix_and_twist_limits_cover_all_six_dof():
    features = np.array([-0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1])
    matrix = interaction_matrix(features, np.full(4, 0.5))
    twist, error = ibvs_camera_twist(features, features + 0.03, np.full(4, 0.5), 0.5, 0.03)
    clipped = clip_twist(twist, linear_max=0.01, angular_max=0.02)

    assert matrix.shape == (8, 6)
    assert np.linalg.norm(error) > 0.0
    assert np.linalg.norm(clipped[:3]) <= 0.0100001
    assert np.linalg.norm(clipped[3:]) <= 0.0200001


def test_feature_timestamp_prefers_image_stamp_and_falls_back_when_unset():
    assert feature_timestamp_ns(_Stamp(12, 345), 999) == 12_000_000_345
    assert feature_timestamp_ns(_Stamp(0, 0), 999) == 999


def test_image_servo_uses_source_timestamps_and_humble_reset_service():
    node_source = (PACKAGE / "visual_servo_bringup" / "nodes" / "visual_image_servo_node.py").read_text(
        encoding="utf-8"
    )
    config = (PACKAGE / "config" / "visual_image_servo.yaml").read_text(encoding="utf-8")
    servo_io = (PACKAGE / "visual_servo_bringup" / "servo" / "servo_io.py").read_text(encoding="utf-8")

    assert "feature_timestamp_ns(message.header.stamp, arrival_ns)" in node_source
    assert "stable_frame_count" not in node_source
    assert "stable_frame_count" not in config
    assert "create_client(Empty" in servo_io
    assert "Empty.Request()" in servo_io
    assert "def start_servo_async" in servo_io
    assert "self._tf_ready = self._transforms_ready()" in node_source


def test_image_servo_yaml_is_complete_and_has_no_profiles():
    config_path = PACKAGE / "config" / "visual_image_servo.yaml"
    config = config_path.read_text(encoding="utf-8")
    parameters = image_servo_parameters(config_path)

    assert "profiles:" not in config
    assert set(parameters) == {
        "image_topic", "camera_info_topic", "debug_image_topic", "error_topic",
        "marker_dictionary", "marker_id", "marker_size_m",
        "base_frame", "camera_frame", "ee_frame", "servo_ns",
        "control_rate_hz", "lambda_gain", "damping", "max_linear_speed", "max_angular_speed",
        "feature_timeout_sec", "image_error_tolerance", "servo_status_halt_codes",
        "reference_path", "auto_start",
    }
    assert parameters["auto_start"] is True
    assert parameters["reference_path"] == ""


class _FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)

    def info(self, _message):
        pass


class _FakeNode:
    def __init__(self):
        self.logger = _FakeLogger()

    def get_logger(self):
        return self.logger


class _FakeFuture:
    def __init__(self, response=None, done=False):
        self.response = response
        self.done_value = done

    def done(self):
        return self.done_value

    def result(self):
        return self.response


class _FakeClient:
    def __init__(self, future):
        self.future = future
        self.calls = 0

    def service_is_ready(self):
        return True

    def call_async(self, _request):
        self.calls += 1
        return self.future


def _async_servo(future):
    servo = ServoIO.__new__(ServoIO)
    servo.node = _FakeNode()
    servo.servo_started = False
    servo._start_servo_future = None
    servo._start_servo_deadline = 0.0
    servo.start_servo_cli = _FakeClient(future)
    servo.reset_servo_status_cli = _FakeClient(_FakeFuture())
    servo.unpause_servo_cli = _FakeClient(_FakeFuture())
    servo.publish_zero_twist = lambda **_kwargs: None
    return servo


def test_async_servo_start_waits_without_duplicate_calls_then_starts():
    response = type("Response", (), {"success": True, "message": ""})()
    future = _FakeFuture(response)
    servo = _async_servo(future)

    assert servo.start_servo_async() is None
    assert servo.start_servo_async() is None
    assert servo.start_servo_cli.calls == 1

    future.done_value = True
    assert servo.start_servo_async() is True
    assert servo.servo_started


def test_async_servo_start_times_out_without_blocking():
    servo = _async_servo(_FakeFuture())

    assert servo.start_servo_async() is None
    servo._start_servo_deadline = 0.0
    assert servo.start_servo_async() is False
    assert servo.node.logger.errors == ["start_servo timeout"]
