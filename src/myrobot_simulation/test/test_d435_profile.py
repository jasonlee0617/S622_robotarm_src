"""Focused tests for named D435 simulator profiles."""

import math
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest
import xacro
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
PROFILE_640_60 = "d435_color_640x480x60_depth_640x480x60"
PROFILE_1280 = "d435_color_1280x720x30_depth_848x480x30"
FAIRINO_XACRO_ROOT = GAZEBO_LAUNCH_ROOT / "config" / "robots" / "fairino_arm"
EYE_ON_BASE_XACRO = FAIRINO_XACRO_ROOT / "fairino_arm_eye_on_base_gazebo.urdf.xacro"


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


def test_profile_geometry_keeps_visual_servo_fps_at_30():
    visual_servo_mappings = {
        "camera_fps": "30",
        "camera_image_width": "640",
        "camera_image_height": "480",
    }
    visual_servo_mappings.update(_mappings(PROFILE_640, camera_fps="30"))

    assert visual_servo_mappings["camera_fps"] == "30"
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


@pytest.mark.parametrize(
    ("xacro_name", "color_type", "color_topic", "has_custom_lens", "has_aligned_depth"),
    [
        ("fairino_arm_handeye_gazebo.urdf.xacro", "camera", "camera/image", False, True),
        ("fairino_arm_gazebo.urdf.xacro", "camera", "camera/image", False, True),
    ],
)
def test_fairino_camera_sensor_mode(
    xacro_name, color_type, color_topic, has_custom_lens, has_aligned_depth
):
    mappings = _mappings(PROFILE_1280) if has_aligned_depth else {}
    root = ET.fromstring(
        xacro.process_file(str(FAIRINO_XACRO_ROOT / xacro_name), mappings=mappings).toxml()
    )
    sensors = {sensor.get("name"): sensor for sensor in root.findall(".//sensor")}

    color = sensors["camera"]
    assert color.get("type") == color_type
    assert color.findtext("topic") == color_topic
    assert (color.find("camera/lens") is not None) is has_custom_lens
    assert sensors["camera_native_depth"].get("type") == "depth_camera"
    assert not [sensor for sensor in sensors.values() if sensor.get("type") == "rgbd_camera"]
    if has_aligned_depth:
        aligned = sensors["camera_aligned_depth"]
        assert aligned.get("type") == "depth_camera"
        assert color.findtext("camera/camera_info_topic") == "camera/camera_info"
        assert aligned.findtext("topic") == "camera/aligned_depth/image"
        assert aligned.findtext("camera/camera_info_topic") == "camera/aligned_depth/camera_info"
        assert aligned.findtext("camera/optical_frame_id") == "camera_color_optical_frame"
        assert aligned.findtext("ignition_frame_id") == "camera_color_frame"
        assert aligned.findtext("camera/image/width") == color.findtext("camera/image/width")
        assert aligned.findtext("camera/image/height") == color.findtext("camera/image/height")
        assert aligned.findtext("camera/horizontal_fov") == color.findtext("camera/horizontal_fov")
        assert aligned.findtext("camera/lens/intrinsics/fx") == "907.7698364257812"
        assert aligned.findtext("camera/lens/intrinsics/fy") == "907.7734985351562"
        assert aligned.findtext("camera/lens/intrinsics/cx") == "648.0337524414062"
        assert aligned.findtext("camera/lens/intrinsics/cy") == "360.25384521484375"
        assert aligned.findtext("camera/lens/projection/p_fx") == "907.7698364257812"
        assert aligned.findtext("camera/lens/projection/p_fy") == "907.7734985351562"
        assert aligned.findtext("camera/lens/projection/p_cx") == "648.0337524414062"
        assert aligned.findtext("camera/lens/projection/p_cy") == "360.25384521484375"
    else:
        assert "camera_aligned_depth" not in sensors


def test_d435_rgb_mode_creates_only_the_color_sensor(tmp_path):
    wrapper = tmp_path / "d435_rgb_mode.xacro"
    wrapper.write_text(
        """<?xml version=\"1.0\"?>
<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\">
  <xacro:include filename=\"%s\"/>
  <xacro:gazebo_d435 type=\"rgb\"/>
</robot>
""" % (PROFILE_PACKAGE_ROOT / "urdf" / "_d435.gazebo.xacro"),
        encoding="utf-8",
    )
    root = ET.fromstring(xacro.process_file(str(wrapper)).toxml())
    sensors = root.findall(".//sensor")

    assert [(sensor.get("name"), sensor.get("type")) for sensor in sensors] == [
        ("camera", "camera")
    ]


def test_camera_info_bridges_are_isolated_and_gazebo_to_ros_only(monkeypatch):
    from launch_utils import perception_stack

    monkeypatch.setattr(perception_stack, "Node", lambda **kwargs: kwargs)
    nodes = perception_stack.camera_bridge_nodes(use_sim_time=True)
    camera_info_arguments = [
        argument
        for node in nodes
        for argument in node["arguments"]
        if "CameraInfo" in argument
    ]
    assert camera_info_arguments == [
        "/camera/native_depth/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
        "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
        "/camera/aligned_depth/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
    ]
    color_info_node = next(
        node
        for node in nodes
        if "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo"
        in node["arguments"]
    )
    aligned_depth_info_node = next(
        node
        for node in nodes
        if "/camera/aligned_depth/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo"
        in node["arguments"]
    )
    assert color_info_node["remappings"] == [
        ("/camera/camera_info", "/camera/camera/color/camera_info")
    ]
    assert aligned_depth_info_node["remappings"] == [
        (
            "/camera/aligned_depth/camera_info",
            "/camera/camera/aligned_depth_to_color/camera_info",
        )
    ]
    image_bridge = nodes[1]
    assert "/camera/aligned_depth/image@sensor_msgs/msg/Image@ignition.msgs.Image" in image_bridge[
        "arguments"
    ]
    assert "/camera/depth_image@sensor_msgs/msg/Image@ignition.msgs.Image" not in image_bridge[
        "arguments"
    ]



def test_eye_on_base_profile_mounts_board_on_wrist_with_base_camera():
    root = ET.fromstring(xacro.process_file(str(EYE_ON_BASE_XACRO)).toxml())
    board_links = root.findall("./link[@name='calibration_board_link']")
    board_joint = root.find("./joint[@name='wrist3_to_calibration_board']")
    camera_joint = root.find("./joint[@name='camera_joint']")

    assert len(board_links) == 1
    assert board_joint.find("parent").get("link") == "grasp_frame"
    assert board_joint.find("child").get("link") == "calibration_board_link"
    assert board_joint.find("origin").get("xyz") == "0.00 -0.03 -0.03"
    assert board_joint.find("origin").get("rpy") == "1.5708 0 0"
    assert camera_joint.find("parent").get("link") == "base_link"

    visuals = board_links[0].findall("visual")
    assert len([v for v in visuals if v.get("name", "").startswith("front_black_")]) == 39
    assert not [v for v in visuals if v.get("name", "").startswith("back_black_")]
    assert board_links[0].find("collision/geometry/box").get("size") == "0.17 0.17 0.01"


def test_eye_on_base_board_mount_accepts_all_six_overrides():
    mappings = {
        "calibration_board_x": "0.1",
        "calibration_board_y": "0.2",
        "calibration_board_z": "0.3",
        "calibration_board_roll": "0.4",
        "calibration_board_pitch": "0.5",
        "calibration_board_yaw": "0.6",
    }
    root = ET.fromstring(
        xacro.process_file(str(EYE_ON_BASE_XACRO), mappings=mappings).toxml()
    )
    origin = root.find("./joint[@name='wrist3_to_calibration_board']/origin")

    assert origin.get("xyz") == "0.1 0.2 0.3"
    assert origin.get("rpy") == "0.4 0.5 0.6"


def test_calibration_launch_defaults_select_the_correct_board_source():
    eye_in_hand = (
        GAZEBO_LAUNCH_ROOT / "launch" / "calibration_gazebo.launch.py"
    ).read_text(encoding="utf-8")
    eye_on_base = (
        GAZEBO_LAUNCH_ROOT / "launch" / "calibration_on_base_gazebo.launch.py"
    ).read_text(encoding="utf-8")

    assert "auto_calibration_collector.py" not in eye_in_hand
    assert "manual_calibration_assistant.py" not in eye_in_hand
    assert "easy_handeye2" not in eye_in_hand
    assert '"spawn_fixed_board": "false"' in eye_on_base
    assert "auto_calibration_collector.py" not in eye_on_base
    assert "manual_calibration_assistant.py" not in eye_on_base
    assert "model.sdf" not in eye_on_base
    assert PROFILE_640 in eye_on_base
    assert '"camera_fps",' in eye_on_base
    assert '("camera_fps", "30", "仿真相机帧率。")' in eye_on_base
    assert PROFILE_1280 in eye_in_hand


def test_visual_servo_uses_the_configured_real_640_profile_and_frame_rate():
    params_path = (
        GAZEBO_LAUNCH_ROOT.parent
        / "visual_servo"
        / "config"
        / "visual_servo_params.yaml"
    )
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    launch = (GAZEBO_LAUNCH_ROOT / "launch" / "visual_servo_gazebo.launch.py").read_text(
        encoding="utf-8"
    )

    assert params["camera_profile"] in {PROFILE_640, PROFILE_640_60}
    expected_fps = "60" if params["camera_profile"] == PROFILE_640_60 else "30"
    assert str(params["camera_fps"]) == expected_fps
    assert f'_SERVO_RUNTIME_DEFAULTS.get("camera_fps", "{expected_fps}")' in launch
    assert "D435 camera profile:" in launch


def test_only_calibrate_is_the_real_environment_entrypoint():
    launch_root = (
        GAZEBO_LAUNCH_ROOT.parent
        / "calibration_ws"
        / "hand_eye_calibration"
        / "launch"
    )

    assert (launch_root / "calibrate.launch.py").exists()
    assert not (launch_root / "auto_calibration.launch.py").exists()
