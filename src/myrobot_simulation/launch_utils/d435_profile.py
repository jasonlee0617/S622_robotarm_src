"""Load a D435 profile into xacro-safe simulator mappings."""

import math
from pathlib import Path
import warnings

from ament_index_python.packages import get_package_share_directory
import yaml


# The simulator only accepts the modes represented by the selected profile.
# These tables deliberately describe the common D435 modes, rather than using
# the profile filename's capture FPS as a run-time FPS limit: the same
# calibration geometry can be used at another supported sensor rate.
_D435_COLOR_MODE_FPS = {
    (1920, 1080): (6, 15, 30),
    (1280, 720): (6, 15, 30),
    (960, 540): (6, 15, 30, 60),
    (848, 480): (6, 15, 30, 60),
    (640, 480): (6, 15, 30, 60),
    (640, 360): (6, 15, 30, 60),
    (424, 240): (6, 15, 30, 60),
}
_D435_DEPTH_MODE_FPS = {
    (1280, 720): (6, 15, 30),
    (848, 480): (6, 15, 30, 60, 90),
    (640, 480): (6, 15, 30, 60, 90),
    (640, 360): (6, 15, 30, 60, 90),
    (480, 270): (6, 15, 30, 60, 90),
    (424, 240): (6, 15, 30, 60, 90),
}
_D435_DEPTH_NEAR_M = {
    (1280, 720): 0.280,
    (848, 480): 0.195,
    (640, 480): 0.175,
    (640, 360): 0.150,
    (480, 270): 0.120,
    (424, 240): 0.105,
}
_D435_DEPTH_FAR_M_MAX = 10.0
_D435_DEPTH_FAR_M_DEFAULT = 3.0


def _profile_dimensions(profile):
    values = str(profile).lower().split("x")
    if len(values) != 3 or not all(value.isdigit() for value in values):
        raise ValueError(f"profile must be WIDTHxHEIGHTxFPS, got {profile!r}")
    return tuple(int(value) for value in values)


def _named_profile_path(camera_profile):
    profile = str(camera_profile).strip()
    if not profile or profile.lower() == "none":
        return None
    if Path(profile).name != profile or profile.endswith(".yaml"):
        raise ValueError("camera_profile must be a profile stem without a path or .yaml suffix")
    directory = (
        Path(get_package_share_directory("realsense2_gz_description"))
        / "config"
        / "d435_profiles"
    )
    path = directory / f"{profile}.yaml"
    if not path.is_file():
        available = ", ".join(item.stem for item in sorted(directory.glob("*.yaml"))) or "none"
        raise ValueError(f"Unknown camera_profile {profile!r}; available profiles: {available}")
    return path


def _camera_info_values(camera_info, label, path):
    if not isinstance(camera_info, dict) or len(camera_info.get("k", ())) < 6:
        raise ValueError(f"{path} has no valid {label}.k")

    try:
        width = int(camera_info["width"])
        height = int(camera_info["height"])
        fx = float(camera_info["k"][0])
        fy = float(camera_info["k"][4])
        cx = float(camera_info["k"][2])
        cy = float(camera_info["k"][5])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{path} has invalid {label} dimensions or intrinsics") from exc
    if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"{path} has non-positive {label} dimensions or focal length")

    distortion = list(camera_info.get("d", ())) + [0.0] * 5
    try:
        distortion = tuple(float(value) for value in distortion[:5])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid {label}.d") from exc
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": distortion[0],
        "k2": distortion[1],
        "p1": distortion[2],
        "p2": distortion[3],
        "k3": distortion[4],
        "h_fov": 2.0 * math.atan(width / (2.0 * fx)),
        "v_fov": 2.0 * math.atan(height / (2.0 * fy)),
    }


def _validate_mode(width, height, fps, mode_fps, sensor_name):
    allowed = mode_fps.get((width, height))
    if allowed is None or fps not in allowed:
        available = ", ".join(str(value) for value in allowed or ()) or "none"
        raise ValueError(
            f"D435 {sensor_name} mode {width}x{height}@{fps} is unsupported "
            f"(allowed FPS: {available})"
        )


def _requested_fps(value):
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera_fps must be an integer D435 frame rate, got {value!r}") from exc
    if not math.isfinite(fps) or not fps.is_integer() or fps <= 0:
        raise ValueError(f"camera_fps must be an integer D435 frame rate, got {value!r}")
    return int(fps)


def _validate_shared_mode(color, depth, camera_fps):
    fps = _requested_fps(camera_fps)
    _validate_mode(
        color["width"], color["height"], fps, _D435_COLOR_MODE_FPS, "color"
    )
    _validate_mode(
        depth["width"], depth["height"], fps, _D435_DEPTH_MODE_FPS, "depth"
    )


def _depth_range(depth, camera_depth_far_m):
    dimensions = (depth["width"], depth["height"])
    near = _D435_DEPTH_NEAR_M.get(dimensions)
    if near is None:
        raise ValueError(
            f"No D435 native-depth minimum range is defined for "
            f"{dimensions[0]}x{dimensions[1]}"
        )
    try:
        far = float(camera_depth_far_m)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"camera_depth_far_m must be in ({near}, {_D435_DEPTH_FAR_M_MAX}], "
            f"got {camera_depth_far_m!r}"
        ) from exc
    if not math.isfinite(far) or far <= near or far > _D435_DEPTH_FAR_M_MAX:
        raise ValueError(
            f"camera_depth_far_m must be in ({near}, {_D435_DEPTH_FAR_M_MAX}], "
            f"got {camera_depth_far_m!r}"
        )
    return near, far


def d435_mappings(
    camera_profile,
    profile_file,
    noise_mode,
    camera_fps=None,
    camera_depth_far_m=_D435_DEPTH_FAR_M_DEFAULT,
):
    """
    Return profile-controlled color/native-depth xacro mappings.

    ``camera_fps`` intentionally remains owned by the calling launch file: a
    profile captures a sensor mode's geometry, not the launch timing policy.
    When supplied, it is validated against both selected D435 stream modes.
    """
    named_path = _named_profile_path(camera_profile)
    external_file = str(profile_file).strip()
    if named_path is not None and external_file:
        raise ValueError("Specify either camera_profile or camera_profile_file, not both")
    if named_path is None and not external_file:
        warnings.warn(
            "No D435 profile supplied: calibration simulation is using nominal "
            "color intrinsics and zero noise.",
            RuntimeWarning,
        )
        return {
            "camera_fx": "0", "camera_fy": "0", "camera_cx": "0", "camera_cy": "0",
            "camera_k1": "0", "camera_k2": "0", "camera_k3": "0",
            "camera_p1": "0", "camera_p2": "0",
            "camera_noise_stddev": "0",
        }
    path = named_path if named_path is not None else Path(external_file).expanduser()
    if not path.is_file():
        raise ValueError(f"D435 profile file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    color = _camera_info_values(data.get("color_camera_info"), "color_camera_info", path)
    depth = _camera_info_values(data.get("depth_camera_info"), "depth_camera_info", path)
    width, height, recorded_color_fps = _profile_dimensions(data.get("color_profile", ""))
    depth_width, depth_height, recorded_depth_fps = _profile_dimensions(data.get("depth_profile", ""))
    if (color["width"], color["height"]) != (width, height):
        raise ValueError(f"{path} color_profile does not match color CameraInfo dimensions")
    if (depth["width"], depth["height"]) != (depth_width, depth_height):
        raise ValueError(f"{path} depth_profile does not match depth CameraInfo dimensions")
    _validate_mode(width, height, recorded_color_fps, _D435_COLOR_MODE_FPS, "color")
    _validate_mode(depth_width, depth_height, recorded_depth_fps, _D435_DEPTH_MODE_FPS, "depth")
    if camera_fps is not None:
        _validate_shared_mode(color, depth, camera_fps)
    near, far = _depth_range(depth, camera_depth_far_m)
    if noise_mode not in ("off", "d435_empirical"):
        raise ValueError("camera_noise_mode must be off or d435_empirical")
    empirical = data.get("empirical_depth_noise_stddev_m")
    if noise_mode == "d435_empirical" and empirical is None:
        raise ValueError(f"{path} has no empirical_depth_noise_stddev_m for d435_empirical mode")
    return {
        "camera_image_width": str(width), "camera_image_height": str(height),
        "camera_fx": str(color["fx"]), "camera_fy": str(color["fy"]),
        "camera_cx": str(color["cx"]), "camera_cy": str(color["cy"]),
        "camera_k1": str(color["k1"]), "camera_k2": str(color["k2"]),
        "camera_p1": str(color["p1"]), "camera_p2": str(color["p2"]),
        "camera_k3": str(color["k3"]),
        "camera_h_fov": str(color["h_fov"]),
        "camera_v_fov": str(color["v_fov"]),
        "camera_depth_image_width": str(depth["width"]),
        "camera_depth_image_height": str(depth["height"]),
        "camera_depth_fx": str(depth["fx"]), "camera_depth_fy": str(depth["fy"]),
        "camera_depth_cx": str(depth["cx"]), "camera_depth_cy": str(depth["cy"]),
        "camera_depth_k1": str(depth["k1"]), "camera_depth_k2": str(depth["k2"]),
        "camera_depth_p1": str(depth["p1"]), "camera_depth_p2": str(depth["p2"]),
        "camera_depth_k3": str(depth["k3"]),
        "camera_depth_h_fov": str(depth["h_fov"]),
        "camera_depth_v_fov": str(depth["v_fov"]),
        "camera_depth_near_m": str(near),
        "camera_depth_far_m": str(far),
        "camera_noise_stddev": str(float(empirical) if noise_mode == "d435_empirical" else 0.0),
    }
