"""Typed configuration for the fixed root-relative hand-eye collection run."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Tuple

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from manipulation_common.planning.motion_executor import PlannerSwitch
from scipy.spatial.transform import Rotation as R

from .model import ToolDeltaSpec


_DEFAULT_JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")

# Root is sample 0. YAML normally supplies these 19 actions; this identical
# tuple is the fallback when the YAML key is absent.
ROOT_RELATIVE_TOOL_DELTAS = (
    "-0.03,0.02,0.01,5.01,9.58,13.29",
    "-0.04,0.03,0.02,9.58,15.82,16.57",
    "-0.03,0.05,0.03,13.29,16.57,7.38",
    "0.00,0.07,0.02,15.82,11.56,-7.38",
    "0.04,0.06,0.00,16.95,2.53,-16.57",
    "0.06,0.04,-0.01,16.57,-7.38,-13.29",
    "0.07,0.02,0.00,14.72,-14.72,0.00",
    "0.06,0.02,0.02,11.56,-16.95,13.29",
    "0.04,0.01,0.03,7.38,-13.29,16.57",
    "0.02,0.00,0.01,2.53,-5.01,7.38",
    "-0.02,0.00,-0.01,-2.53,5.01,-7.38",
    "-0.05,0.01,-0.02,-7.38,13.29,-16.57",
    "-0.07,0.01,0.00,-11.56,16.95,-13.29",
    "-0.07,-0.02,0.02,-14.72,14.72,0.00",
    "-0.04,-0.05,0.02,-16.57,7.38,13.29",
    "-0.01,-0.06,0.01,-16.95,-2.53,16.57",
    "0.02,-0.07,0.00,-15.82,-11.56,7.38",
    "0.03,-0.07,-0.01,-13.29,-16.57,-7.38",
    "0.03,-0.06,-0.01,-9.58,-15.82,-16.57",
)


@dataclass(frozen=True)
class CollectorFramesConfig:
    base_frame: str
    ee_frame: str
    tracking_base_frame: str
    tracking_marker_frame: str
    marker_id: int
    image_topic: str
    aruco_dictionary_id: str
    camera_info_topic: str


@dataclass(frozen=True)
class CollectorMotionConfig:
    move_group_name: str
    move_group_ns_fairino: str
    move_group_ns_kdl: str
    ik_plugin: str
    planning_pipeline_id: str
    planner_id: str
    joint_names: Tuple[str, ...]
    original_place_xyz: Tuple[float, float, float]
    original_place_rpy_deg: Tuple[float, float, float]
    workspace_min_xyz: Tuple[float, float, float]
    workspace_max_xyz: Tuple[float, float, float]
    max_velocity: float
    max_acceleration: float
    allowed_planning_time: float
    position_tolerance: float
    orientation_tolerance: float
    allowed_start_tolerance: float
    action_delay: float
    num_candidate_plans: int
    wrist_weight: float
    settle_time: float
    original_place_attempts: int
    original_place_motion_timeout: float
    original_place_retry_wait: float
    candidate_motion_timeout: float
    candidate_max_joint_excursion_rad: float
    candidate_max_adjacent_joint_jump_rad: float
    candidate_max_wrist_travel_rad: float
    keyboard_poll_period: float
    start_wait_poll_period: float
    step_between_actions: bool


@dataclass(frozen=True)
class CollectorSamplingConfig:
    marker_recent_timeout: float
    min_marker_distance: float
    max_marker_distance: float
    marker_size_m: float
    min_visible_border_px: float
    min_marker_side_px: float
    stable_frame_count: int
    stable_min_valid_frames: int
    stable_observation_timeout: float
    calibration_output_directory: str
    calibration_file_prefix: str
    max_calibration_translation_norm_m: float
    moveit_ready_timeout: float
    moveit_ready_poll_interval: float
    minimum_samples: int
    minimum_solution_samples: int
    tool_delta_specs: Tuple[ToolDeltaSpec, ...]
    root_position_tolerance_m: float
    root_orientation_tolerance_deg: float
    pnp_reprojection_rms_max_px: float
    pnp_reprojection_max_corner_px: float
    ippe_ambiguity_abs_gap_px: float
    ippe_ambiguity_max_ratio: float
    ippe_min_non_ambiguous_frames: int
    max_pnp_translation_mad_m: float
    max_pnp_rotation_mad_deg: float
    max_joint_velocity_rad_s: float
    max_ee_translation_drift_m: float
    max_ee_rotation_drift_deg: float
    sample_min_translation_delta_m: float
    sample_min_rotation_delta_deg: float
    solver_translation_sigma_m: float
    solver_rotation_sigma_deg: float
    max_algorithm_translation_delta_m: float
    max_algorithm_rotation_delta_deg: float
    max_marker_position_rms_m: float
    max_marker_rotation_rms_deg: float
    min_translation_span_m: float
    min_rotation_span_deg: float
    min_informative_rotation_pairs: int
    min_rotation_axis_ratio: float
    simulation_truth_translation_m: float
    simulation_truth_rotation_deg: float


def _load_yaml_defaults() -> dict:
    paths = []
    try:
        paths.append(os.path.join(get_package_share_directory("hand_eye_calibration"), "config", "auto_calibration_collector.yaml"))
    except Exception:
        pass
    paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "auto_calibration_collector.yaml")))
    for path in paths:
        try:
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
            parameters = data.get("auto_calibration_collector", {}).get("ros__parameters", {})
            if isinstance(parameters, dict):
                return parameters
        except OSError:
            pass
    return {}


def _str(node, name, default):
    node.declare_parameter(name, default)
    return str(node.get_parameter(name).value)


def _float(node, name, default):
    node.declare_parameter(name, default)
    return float(node.get_parameter(name).value)


def _int(node, name, default):
    node.declare_parameter(name, default)
    return int(node.get_parameter(name).value)


def _bool(node, name, default):
    node.declare_parameter(name, default)
    return bool(node.get_parameter(name).value)


def _list(node, name, default):
    node.declare_parameter(name, list(default))
    value = node.get_parameter(name).value
    return list(default if value is None else value)


def _triple(values, name):
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(float(value) for value in values)


def _tool_deltas(node, name, default):
    specs = []
    for index, raw in enumerate(_list(node, name, default), start=1):
        try:
            values = tuple(float(value.strip()) for value in str(raw).split(","))
        except ValueError as exc:
            raise ValueError(f"{name}[{index}] must contain exactly six comma-separated numbers") from exc
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name}[{index}] must contain exactly six finite values")
        if not any(abs(value) > 1.0e-12 for value in values):
            raise ValueError(f"{name}[{index}] must not duplicate the root pose")
        specs.append(ToolDeltaSpec(*values))
    return tuple(specs)


def _frames(node, d):
    return CollectorFramesConfig(
        base_frame=_str(node, "base_frame", d("base_frame", "base_link")),
        ee_frame=_str(node, "ee_frame", d("ee_frame", "grasp_frame")),
        tracking_base_frame=_str(node, "tracking_base_frame", d("tracking_base_frame", "camera_color_optical_frame")),
        tracking_marker_frame=_str(node, "tracking_marker_frame", d("tracking_marker_frame", "calibration_aruco")),
        marker_id=_int(node, "marker_id", d("marker_id", 1)),
        image_topic=_str(node, "image_topic", d("image_topic", "/camera/camera/color/image_raw")),
        aruco_dictionary_id=_str(node, "aruco_dictionary_id", d("aruco_dictionary_id", "DICT_5X5_250")),
        camera_info_topic=_str(node, "camera_info_topic", d("camera_info_topic", "/camera/camera/color/camera_info")),
    )


def _motion(node, d):
    pipeline = PlannerSwitch.normalize_pipeline(_str(node, "planning_pipeline_id", d("planning_pipeline_id", "fairino")))
    planner = PlannerSwitch.normalize_planner(pipeline, _str(node, "planner_id", d("planner_id", "tube_birrt*" if pipeline == "fairino" else "RRTConnectFast")))
    if not PlannerSwitch.is_valid(pipeline, planner):
        raise ValueError(f"Unsupported planner config: pipeline={pipeline}, planner={planner}")
    config = CollectorMotionConfig(
        move_group_name=_str(node, "move_group_name", d("move_group_name", "robot_arm")),
        move_group_ns_fairino=_str(node, "move_group_ns_fairino", d("move_group_ns_fairino", "/move_group_fairino")),
        move_group_ns_kdl=_str(node, "move_group_ns_kdl", d("move_group_ns_kdl", "/move_group_kdl")),
        ik_plugin=PlannerSwitch.normalize_ik(_str(node, "ik_plugin", d("ik_plugin", "fairino"))),
        planning_pipeline_id=pipeline, planner_id=planner,
        joint_names=tuple(str(value) for value in _list(node, "joint_names", d("joint_names", _DEFAULT_JOINT_NAMES))),
        original_place_xyz=_triple(_list(node, "original_place_xyz", d("original_place_xyz", [0.3150, 0.0325, 0.3506])), "original_place_xyz"),
        original_place_rpy_deg=_triple(_list(node, "original_place_rpy_deg", d("original_place_rpy_deg", [0.0, 180.0, 90.0])), "original_place_rpy_deg"),
        workspace_min_xyz=_triple(_list(node, "workspace_min_xyz", d("workspace_min_xyz", [0.05, -0.35, 0.02])), "workspace_min_xyz"),
        workspace_max_xyz=_triple(_list(node, "workspace_max_xyz", d("workspace_max_xyz", [0.55, 0.35, 0.45])), "workspace_max_xyz"),
        max_velocity=_float(node, "max_velocity", d("max_velocity", 0.15)),
        max_acceleration=_float(node, "max_acceleration", d("max_acceleration", 0.15)),
        allowed_planning_time=_float(node, "allowed_planning_time", d("allowed_planning_time", 8.0)),
        position_tolerance=_float(node, "position_tolerance", d("position_tolerance", 0.008)),
        orientation_tolerance=_float(node, "orientation_tolerance", d("orientation_tolerance", 0.0175)),
        allowed_start_tolerance=_float(node, "allowed_start_tolerance", d("allowed_start_tolerance", 0.05)),
        action_delay=_float(node, "action_delay", d("action_delay", 0.2)),
        num_candidate_plans=_int(node, "num_candidate_plans", d("num_candidate_plans", 8)),
        wrist_weight=_float(node, "wrist_weight", d("wrist_weight", 80.0)),
        settle_time=_float(node, "settle_time", d("settle_time", 1.2)),
        original_place_attempts=max(1, _int(node, "original_place_attempts", d("original_place_attempts", 3))),
        original_place_motion_timeout=_float(node, "original_place_motion_timeout", d("original_place_motion_timeout", 30.0)),
        original_place_retry_wait=_float(node, "original_place_retry_wait", d("original_place_retry_wait", 2.0)),
        candidate_motion_timeout=_float(node, "candidate_motion_timeout", d("candidate_motion_timeout", 30.0)),
        candidate_max_joint_excursion_rad=_float(node, "candidate_max_joint_excursion_rad", d("candidate_max_joint_excursion_rad", 0.45)),
        candidate_max_adjacent_joint_jump_rad=_float(node, "candidate_max_adjacent_joint_jump_rad", d("candidate_max_adjacent_joint_jump_rad", 0.10)),
        candidate_max_wrist_travel_rad=_float(node, "candidate_max_wrist_travel_rad", d("candidate_max_wrist_travel_rad", 0.55)),
        keyboard_poll_period=_float(node, "keyboard_poll_period", d("keyboard_poll_period", 0.1)),
        start_wait_poll_period=_float(node, "start_wait_poll_period", d("start_wait_poll_period", 0.1)),
        step_between_actions=_bool(node, "step_between_actions", d("step_between_actions", True)),
    )
    if any(low >= high for low, high in zip(config.workspace_min_xyz, config.workspace_max_xyz)):
        raise ValueError("workspace_min_xyz must be strictly smaller than workspace_max_xyz")
    return config


def _candidate_separation_ok(specs, translation_min, rotation_min_deg):
    transforms = [(0, np.zeros(3), R.identity())]
    for index, spec in enumerate(specs, start=1):
        transforms.append((index, np.asarray((spec.dx_m, spec.dy_m, spec.dz_m)), R.from_euler("xyz", (spec.rx_deg, spec.ry_deg, spec.rz_deg), degrees=True)))
    for (left_index, left_t, left_r), (right_index, right_t, right_r) in product(transforms, transforms):
        if left_index < right_index and np.linalg.norm(left_t - right_t) < translation_min and math.degrees((left_r.inv() * right_r).magnitude()) < rotation_min_deg:
            return False
    return True


def _sampling(node, d):
    config = CollectorSamplingConfig(
        marker_recent_timeout=_float(node, "marker_recent_timeout", d("marker_recent_timeout", 1.5)),
        min_marker_distance=_float(node, "min_marker_distance", d("min_marker_distance", 0.25)),
        max_marker_distance=_float(node, "max_marker_distance", d("max_marker_distance", 0.80)),
        marker_size_m=_float(node, "marker_size_m", d("marker_size_m", 0.07)),
        min_visible_border_px=_float(node, "min_visible_border_px", d("min_visible_border_px", 60.0)),
        min_marker_side_px=_float(node, "min_marker_side_px", d("min_marker_side_px", 90.0)),
        stable_frame_count=max(1, _int(node, "stable_frame_count", d("stable_frame_count", 10))),
        stable_min_valid_frames=max(1, _int(node, "stable_min_valid_frames", d("stable_min_valid_frames", 10))),
        stable_observation_timeout=_float(node, "stable_observation_timeout", d("stable_observation_timeout", 4.0)),
        calibration_output_directory=os.path.expanduser(os.path.expandvars(_str(node, "calibration_output_directory", d("calibration_output_directory", "$HOME/fairino_robotarm/src/calibration_ws/hand_eye_calibration/calib/sim")))),
        calibration_file_prefix=_str(node, "calibration_file_prefix", d("calibration_file_prefix", "robot_calibration")),
        max_calibration_translation_norm_m=_float(node, "max_calibration_translation_norm_m", d("max_calibration_translation_norm_m", 0.30)),
        moveit_ready_timeout=_float(node, "moveit_ready_timeout", d("moveit_ready_timeout", 30.0)),
        moveit_ready_poll_interval=_float(node, "moveit_ready_poll_interval", d("moveit_ready_poll_interval", 0.2)),
        minimum_samples=max(3, _int(node, "minimum_samples", d("minimum_samples", 15))),
        minimum_solution_samples=max(3, _int(node, "minimum_solution_samples", d("minimum_solution_samples", 14))),
        tool_delta_specs=_tool_deltas(node, "tool_delta_specs", d("tool_delta_specs", ROOT_RELATIVE_TOOL_DELTAS)),
        root_position_tolerance_m=_float(node, "root_position_tolerance_m", d("root_position_tolerance_m", 0.003)),
        root_orientation_tolerance_deg=_float(node, "root_orientation_tolerance_deg", d("root_orientation_tolerance_deg", 0.5)),
        pnp_reprojection_rms_max_px=_float(node, "pnp_reprojection_rms_max_px", d("pnp_reprojection_rms_max_px", 1.0)),
        pnp_reprojection_max_corner_px=_float(node, "pnp_reprojection_max_corner_px", d("pnp_reprojection_max_corner_px", 2.0)),
        ippe_ambiguity_abs_gap_px=_float(node, "ippe_ambiguity_abs_gap_px", d("ippe_ambiguity_abs_gap_px", 0.05)),
        ippe_ambiguity_max_ratio=_float(node, "ippe_ambiguity_max_ratio", d("ippe_ambiguity_max_ratio", 1.10)),
        ippe_min_non_ambiguous_frames=max(0, _int(node, "ippe_min_non_ambiguous_frames", d("ippe_min_non_ambiguous_frames", 3))),
        max_pnp_translation_mad_m=_float(node, "max_pnp_translation_mad_m", d("max_pnp_translation_mad_m", 0.0005)),
        max_pnp_rotation_mad_deg=_float(node, "max_pnp_rotation_mad_deg", d("max_pnp_rotation_mad_deg", 0.15)),
        max_joint_velocity_rad_s=_float(node, "max_joint_velocity_rad_s", d("max_joint_velocity_rad_s", 0.005)),
        max_ee_translation_drift_m=_float(node, "max_ee_translation_drift_m", d("max_ee_translation_drift_m", 0.0003)),
        max_ee_rotation_drift_deg=_float(node, "max_ee_rotation_drift_deg", d("max_ee_rotation_drift_deg", 0.05)),
        sample_min_translation_delta_m=_float(node, "sample_min_translation_delta_m", d("sample_min_translation_delta_m", 0.006)),
        sample_min_rotation_delta_deg=_float(node, "sample_min_rotation_delta_deg", d("sample_min_rotation_delta_deg", 3.0)),
        solver_translation_sigma_m=_float(node, "solver_translation_sigma_m", d("solver_translation_sigma_m", 0.0005)),
        solver_rotation_sigma_deg=_float(node, "solver_rotation_sigma_deg", d("solver_rotation_sigma_deg", 0.30)),
        max_algorithm_translation_delta_m=_float(node, "max_algorithm_translation_delta_m", d("max_algorithm_translation_delta_m", 0.003)),
        max_algorithm_rotation_delta_deg=_float(node, "max_algorithm_rotation_delta_deg", d("max_algorithm_rotation_delta_deg", 1.0)),
        max_marker_position_rms_m=_float(node, "max_marker_position_rms_m", d("max_marker_position_rms_m", 0.002)),
        max_marker_rotation_rms_deg=_float(node, "max_marker_rotation_rms_deg", d("max_marker_rotation_rms_deg", 0.7)),
        min_translation_span_m=_float(node, "min_translation_span_m", d("min_translation_span_m", 0.040)),
        min_rotation_span_deg=_float(node, "min_rotation_span_deg", d("min_rotation_span_deg", 20.0)),
        min_informative_rotation_pairs=max(1, _int(node, "min_informative_rotation_pairs", d("min_informative_rotation_pairs", 20))),
        min_rotation_axis_ratio=_float(node, "min_rotation_axis_ratio", d("min_rotation_axis_ratio", 0.20)),
        simulation_truth_translation_m=_float(node, "simulation_truth_translation_m", d("simulation_truth_translation_m", 0.003)),
        simulation_truth_rotation_deg=_float(node, "simulation_truth_rotation_deg", d("simulation_truth_rotation_deg", 1.0)),
    )
    if not 0.0 < config.min_marker_distance < config.max_marker_distance:
        raise ValueError("marker distance bounds are invalid")
    if not 1 <= config.stable_min_valid_frames <= config.stable_frame_count:
        raise ValueError("stable_min_valid_frames must be within stable_frame_count")
    if len(config.tool_delta_specs) != 19:
        raise ValueError("tool_delta_specs must contain exactly 19 actions; root is sample 0")
    if config.minimum_solution_samples > config.minimum_samples:
        raise ValueError("minimum_solution_samples must not exceed minimum_samples")
    if not 0.0 < config.min_rotation_axis_ratio <= 1.0:
        raise ValueError("min_rotation_axis_ratio must be within (0, 1]")
    if not _candidate_separation_ok(config.tool_delta_specs, config.sample_min_translation_delta_m, config.sample_min_rotation_delta_deg):
        raise ValueError("tool_delta_specs contain indistinguishable poses")
    return config


def load_collector_config(node):
    defaults = _load_yaml_defaults()
    return _frames(node, defaults.get), _motion(node, defaults.get), _sampling(node, defaults.get)
