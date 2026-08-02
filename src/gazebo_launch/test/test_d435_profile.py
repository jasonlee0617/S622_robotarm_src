"""Focused tests for named D435 simulator profiles."""

import math
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml


GAZEBO_LAUNCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GAZEBO_LAUNCH_ROOT))

from launch_utils import d435_profile  # noqa: E402


PROFILE_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "camera_ws"
    / "realsense2_gz_description"
)
PROFILE_640 = "d435_color_640x480x30_depth_640x480x30"
PROFILE_1280 = "d435_color_1280x720x30_depth_848x480x30"


def _mappings(profile, profile_file="", noise_mode="off", **kwargs):
    with patch.object(
        d435_profile,
        "get_package_share_directory",
        return_value=str(PROFILE_PACKAGE_ROOT),
    ):
        return d435_profile.d435_mappings(profile, profile_file, noise_mode, **kwargs)


@pytest.mark.parametrize(
    ("profile", "width", "height", "fx", "fy"),
    [
        (PROFILE_640, "640", "480", "605.179931640625", "605.182373046875"),
        (PROFILE_1280, "1280", "720", "907.7698364257812", "907.7734985351562"),
    ],
)
def test_named_profile_selects_color_geometry_without_fps_override(profile, width, height, fx, fy):
    mappings = _mappings(profile)

    assert mappings["camera_image_width"] == width
    assert mappings["camera_image_height"] == height
    assert mappings["camera_fx"] == fx
    assert mappings["camera_fy"] == fy
    assert [
        mappings[name]
        for name in ("camera_k1", "camera_k2", "camera_k3", "camera_p1", "camera_p2")
    ] == ["0.0"] * 5
    assert "camera_fps" not in mappings


@pytest.mark.parametrize(
    (
        "profile",
        "color_dimensions",
        "color_focal_lengths",
        "depth_dimensions",
        "depth_intrinsics",
        "depth_near_m",
    ),
    [
        (
            PROFILE_640,
            (640, 480),
            (605.179931640625, 605.182373046875),
            (640, 480),
            (381.0703125, 381.0703125, 326.4877624511719, 236.5358123779297),
            0.175,
        ),
        (
            PROFILE_1280,
            (1280, 720),
            (907.7698364257812, 907.7734985351562),
            (848, 480),
            (420.76513671875, 420.76513671875, 431.16357421875, 236.17495727539062),
            0.195,
        ),
    ],
)
def test_named_profile_maps_color_and_native_depth_geometry(
    profile,
    color_dimensions,
    color_focal_lengths,
    depth_dimensions,
    depth_intrinsics,
    depth_near_m,
):
    mappings = _mappings(profile)

    assert (
        mappings["camera_depth_image_width"],
        mappings["camera_depth_image_height"],
    ) == tuple(str(value) for value in depth_dimensions)
    assert tuple(float(mappings[name]) for name in (
        "camera_depth_fx",
        "camera_depth_fy",
        "camera_depth_cx",
        "camera_depth_cy",
    )) == pytest.approx(depth_intrinsics)
    assert [
        mappings[name]
        for name in (
            "camera_depth_k1",
            "camera_depth_k2",
            "camera_depth_k3",
            "camera_depth_p1",
            "camera_depth_p2",
        )
    ] == ["0.0"] * 5
    assert float(mappings["camera_h_fov"]) == pytest.approx(
        2.0 * math.atan(color_dimensions[0] / (2.0 * color_focal_lengths[0]))
    )
    assert float(mappings["camera_v_fov"]) == pytest.approx(
        2.0 * math.atan(color_dimensions[1] / (2.0 * color_focal_lengths[1]))
    )
    assert float(mappings["camera_depth_h_fov"]) == pytest.approx(
        2.0 * math.atan(depth_dimensions[0] / (2.0 * depth_intrinsics[0]))
    )
    assert float(mappings["camera_depth_v_fov"]) == pytest.approx(
        2.0 * math.atan(depth_dimensions[1] / (2.0 * depth_intrinsics[1]))
    )
    assert float(mappings["camera_depth_near_m"]) == pytest.approx(depth_near_m)
    assert float(mappings["camera_depth_far_m"]) == pytest.approx(3.0)


def test_named_profile_and_external_file_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either camera_profile or camera_profile_file"):
        _mappings(PROFILE_640, "/tmp/external.yaml")


def test_profile_geometry_keeps_visual_servo_fps_at_60():
    visual_servo_mappings = {
        "camera_fps": "60",
        "camera_image_width": "640",
        "camera_image_height": "480",
    }
    visual_servo_mappings.update(_mappings(PROFILE_640, camera_fps="60"))

    assert visual_servo_mappings["camera_fps"] == "60"
    assert visual_servo_mappings["camera_image_width"] == "640"
    assert visual_servo_mappings["camera_image_height"] == "480"


def test_1280_profile_rejects_unsupported_shared_60_fps():
    with pytest.raises(ValueError, match=r"D435 color mode 1280x720@60 is unsupported"):
        _mappings(PROFILE_1280, camera_fps=60)


def test_depth_far_range_defaults_to_three_metres_and_allows_ten():
    assert _mappings(PROFILE_640)["camera_depth_far_m"] == "3.0"
    assert _mappings(PROFILE_640, camera_depth_far_m="10")["camera_depth_far_m"] == "10.0"
    with pytest.raises(ValueError, match="camera_depth_far_m"):
        _mappings(PROFILE_640, camera_depth_far_m="10.01")


def test_canonical_profiles_do_not_store_device_serials():
    profiles_dir = PROFILE_PACKAGE_ROOT / "config" / "d435_profiles"
    profile_paths = sorted(profiles_dir.glob("d435_*.yaml"))

    assert profile_paths
    for profile_path in profile_paths:
        assert "unknown" not in profile_path.name
        assert "serial" not in yaml.safe_load(profile_path.read_text(encoding="utf-8"))


def test_profile_name_must_be_a_stem():
    with pytest.raises(ValueError, match="stem without a path or .yaml suffix"):
        _mappings(f"{PROFILE_640}.yaml")
