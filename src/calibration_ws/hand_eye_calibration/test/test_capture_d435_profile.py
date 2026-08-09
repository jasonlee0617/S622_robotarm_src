"""Unit tests for the hardware-independent D435 profile capture helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_d435_profile.py"
_SPEC = importlib.util.spec_from_file_location("capture_d435_profile", _SCRIPT)
capture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(capture)


def _frames(rate_hz, count=60, dimensions=(640, 480)):
    period_ns = round(1_000_000_000 / rate_hz)
    return [(dimensions, index * period_ns + 1) for index in range(count)]


def test_profile_spec_requires_positive_width_height_and_fps():
    assert capture._profile_spec("640x480x60", "color_profile") == (640, 480, 60)
    with pytest.raises(ValueError, match="WIDTHxHEIGHTxFPS"):
        capture._profile_spec("640x480", "color_profile")
    with pytest.raises(ValueError, match="positive"):
        capture._profile_spec("640x0x60", "color_profile")


def test_image_stream_accepts_requested_60_hz_and_rejects_mismatches():
    ok, measured = capture._validate_image_stream(
        "Color", _frames(60), (640, 480, 60), 0.10
    )
    assert ok
    assert measured == pytest.approx(60.0, rel=1.0e-6)

    ok, reason = capture._validate_image_stream(
        "Color", _frames(30), (640, 480, 60), 0.10
    )
    assert not ok
    assert "does not match" in reason

    frames = _frames(60)
    frames[10] = ((640, 480), frames[9][1])
    ok, reason = capture._validate_image_stream("Color", frames, (640, 480, 60), 0.10)
    assert not ok
    assert "duplicate or out of order" in reason

    ok, reason = capture._validate_image_stream(
        "Color", _frames(60, dimensions=(848, 480)), (640, 480, 60), 0.10
    )
    assert not ok
    assert "dimensions" in reason


def test_temporal_depth_noise_uses_per_pixel_variation():
    patches = [
        np.full((16, 16), 1.000, dtype=float),
        np.full((16, 16), 1.001, dtype=float),
        np.full((16, 16), 0.999, dtype=float),
    ]
    assert capture._temporal_depth_noise_stddev(patches) == pytest.approx(0.001)
    assert capture._temporal_depth_noise_stddev([patches[0]]) is None
