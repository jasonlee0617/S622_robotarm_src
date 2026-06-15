from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from yolov8_grasping.planning.motion_executor import PlannerSwitch


_DEFAULT_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


@dataclass(frozen=True)
class CollectorFramesConfig:
    base_frame: str
    ee_frame: str
    tracking_base_frame: str
    tracking_marker_frame: str
    marker_id: int
    aruco_topic: str
    image_topic: str
    aruco_dictionary_id: str
    camera_info_topic: str
    take_sample_service: str
    get_sample_list_service: str
    remove_sample_service: str
    compute_calibration_service: str
    save_calibration_service: str
    save_samples_service: str


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
    seed_camera_xyz_m: Tuple[float, float, float]
    seed_camera_rpy_deg: Tuple[float, float, float]
    seed_usage_mode: str
    workspace_min_xyz: Tuple[float, float, float]
    workspace_max_xyz: Tuple[float, float, float]
    preplan_original_place: bool
    max_velocity: float
    max_acceleration: float
    allowed_planning_time: float
    max_step_size: float
    position_tolerance: float
    orientation_tolerance: float
    allowed_start_tolerance: float
    action_delay: float
    num_candidate_plans: int
    wrist_weight: float
    wrist_joint_indices: Tuple[int, ...]
    require_marker_tf: bool
    settle_time: float
    recenter_gain: float
    max_recenter_iters: int
    recenter_max_step_m: float
    recenter_min_step_m: float
    recenter_max_total_translation_m: float
    recenter_improvement_ratio: float
    recenter_axis_frame: str
    recenter_right_sign: float
    recenter_up_sign: float
    recenter_depth_scale_gain: float
    recover_last_good_on_marker_loss: bool
    original_place_attempts: int
    original_place_motion_timeout: float
    original_place_retry_wait: float
    recovery_motion_timeout: float
    recenter_max_velocity: float
    recenter_max_acceleration: float
    recenter_motion_timeout: float
    standby_retry_wait: float
    keyboard_poll_period: float
    start_wait_poll_period: float


@dataclass(frozen=True)
class CollectorSamplingConfig:
    marker_timeout: float
    marker_recent_timeout: float
    min_marker_distance: float
    max_marker_distance: float
    marker_size_m: float
    min_image_margin_px: float
    min_projected_marker_px: float
    startup_min_corner_margin_px: float
    min_corner_margin_px: float
    min_marker_side_px: float
    max_center_error_px: float
    visibility_stable_frames: int
    stable_frame_count: int
    visibility_stable_timeout: float
    max_center_std_px: float
    max_depth_std_m: float
    max_angle_std_deg: float
    camera_model_max_pixel_error: float
    min_successful_samples: int
    max_candidate_attempts: int
    auto_compute: bool
    auto_save_calibration: bool
    auto_save_samples: bool
    enable_calibration_sanity_check: bool
    validate_calibration_against_tf_mount: bool
    calibration_tf_mount_check_hard_gate: bool
    max_calibration_translation_norm_m: float
    max_calibration_tf_translation_error_m: float
    max_calibration_tf_rotation_error_deg: float
    max_calibration_marker_span_m: float
    min_coverage_xy_span_m: float
    min_coverage_z_span_m: float
    min_coverage_rotation_span_deg: float
    coverage_stop_extra_samples: int
    coverage_stop_margin_xy_m: float
    coverage_stop_margin_z_m: float
    coverage_stop_margin_rot_deg: float
    sample_min_translation_delta: float
    sample_min_rotation_delta_deg: float
    nominal_translation_delta_scale: float
    nominal_rotation_delta_scale: float
    sampling_base_x_offsets_m: Tuple[float, ...]
    sampling_base_y_offsets_m: Tuple[float, ...]
    sampling_base_z_offsets_m: Tuple[float, ...]
    sampling_tilt_x_offsets_deg: Tuple[float, ...]
    sampling_tilt_y_offsets_deg: Tuple[float, ...]
    sampling_roll_offsets_deg: Tuple[float, ...]
    get_samples_service_wait_timeout: float
    get_samples_call_timeout: float
    remove_samples_service_wait_timeout: float
    remove_samples_call_timeout: float
    take_sample_service_wait_timeout: float
    take_sample_call_timeout: float
    empty_service_wait_timeout: float
    save_samples_timeout: float
    compute_calibration_timeout: float
    save_calibration_timeout: float
    moveit_ready_timeout: float
    moveit_ready_poll_interval: float
    candidate_preplan_enabled: bool
    recenter_sign_error_growth_ratio: float
    recenter_error_stall_max_iters: int
    auto_prune_outlier_samples: bool
    max_outlier_prune_rounds: int
    outlier_residual_min_m: float
    outlier_residual_keep_ratio: float


def _load_yaml_defaults() -> dict:
    candidate_paths = []
    try:
        candidate_paths.append(
            os.path.join(
                get_package_share_directory("hand_eye_calibration"),
                "config",
                "auto_calibration_collector.yaml",
            )
        )
    except Exception:
        pass
    candidate_paths.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "auto_calibration_collector.yaml")
        )
    )

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
        except Exception:
            continue
        params = data.get("auto_calibration_collector", {}).get("ros__parameters", {})
        if isinstance(params, dict):
            return params
    return {}


def _yaml_default(defaults: dict, name: str, fallback):
    return defaults.get(name, fallback)


def _param_str(node, name: str, default: str) -> str:
    node.declare_parameter(name, default)
    return str(node.get_parameter(name).value)


def _param_float(node, name: str, default: float) -> float:
    node.declare_parameter(name, default)
    return float(node.get_parameter(name).value)


def _param_int(node, name: str, default: int) -> int:
    node.declare_parameter(name, default)
    return int(node.get_parameter(name).value)


def _param_bool(node, name: str, default: bool) -> bool:
    node.declare_parameter(name, default)
    value = node.get_parameter(name).value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _param_list(node, name: str, default: List) -> List:
    node.declare_parameter(name, default)
    value = node.get_parameter(name).value
    if value is None:
        return list(default)
    return list(value)


def load_collector_config(node):
    defaults = _load_yaml_defaults()
    def d(name: str, fallback):
        return _yaml_default(defaults, name, fallback)

    base_frame = _param_str(node, "base_frame", d("base_frame", "base_link"))
    ee_frame = _param_str(node, "ee_frame", d("ee_frame", "grasp_frame"))
    tracking_base_frame = _param_str(
        node,
        "tracking_base_frame",
        d("tracking_base_frame", "camera_color_optical_frame"),
    )
    tracking_marker_frame = _param_str(
        node,
        "tracking_marker_frame",
        d("tracking_marker_frame", "calibration_aruco"),
    )

    move_group_name = _param_str(node, "move_group_name", d("move_group_name", "robot_arm"))
    legacy_move_group_namespace = _param_str(node, "move_group_namespace", d("move_group_namespace", ""))
    move_group_ns_fairino = _param_str(
        node,
        "move_group_ns_fairino",
        _yaml_default(defaults, "move_group_ns_fairino", legacy_move_group_namespace or "/move_group_fairino"),
    )
    move_group_ns_kdl = _param_str(
        node,
        "move_group_ns_kdl",
        _yaml_default(defaults, "move_group_ns_kdl", legacy_move_group_namespace or "/move_group_kdl"),
    )
    ik_plugin = PlannerSwitch.normalize_ik(
        _param_str(node, "ik_plugin", _yaml_default(defaults, "ik_plugin", "fairino"))
    )
    planning_pipeline_id = PlannerSwitch.normalize_pipeline(
        _param_str(node, "planning_pipeline_id", _yaml_default(defaults, "planning_pipeline_id", "fairino"))
    )
    planner_default = "birrt*" if planning_pipeline_id == "fairino" else "RRTConnectFast"
    planner_id = PlannerSwitch.normalize_planner(
        planning_pipeline_id,
        _param_str(node, "planner_id", _yaml_default(defaults, "planner_id", "")) or planner_default,
    )

    frames_config = CollectorFramesConfig(
        base_frame=base_frame,
        ee_frame=ee_frame,
        tracking_base_frame=tracking_base_frame,
        tracking_marker_frame=tracking_marker_frame,
        marker_id=int(_param_int(node, "marker_id", _yaml_default(defaults, "marker_id", 1))),
        aruco_topic=_param_str(node, "aruco_topic", _yaml_default(defaults, "aruco_topic", "/aruco_markers")),
        image_topic=_param_str(
            node, "image_topic", _yaml_default(defaults, "image_topic", "/camera/camera/color/image_raw")
        ),
        aruco_dictionary_id=_param_str(
            node,
            "aruco_dictionary_id",
            _yaml_default(defaults, "aruco_dictionary_id", "DICT_5X5_250"),
        ),
        camera_info_topic=_param_str(
            node,
            "camera_info_topic",
            _yaml_default(defaults, "camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"),
        ),
        take_sample_service=_param_str(
            node,
            "take_sample_service",
            _yaml_default(defaults, "take_sample_service", "/easy_handeye2/calibration/take_sample"),
        ),
        get_sample_list_service=_param_str(
            node,
            "get_sample_list_service",
            _yaml_default(defaults, "get_sample_list_service", "/easy_handeye2/calibration/get_sample_list"),
        ),
        remove_sample_service=_param_str(
            node,
            "remove_sample_service",
            _yaml_default(defaults, "remove_sample_service", "/easy_handeye2/calibration/remove_sample"),
        ),
        compute_calibration_service=_param_str(
            node,
            "compute_calibration_service",
            _yaml_default(
                defaults,
                "compute_calibration_service",
                "/easy_handeye2/calibration/compute_calibration",
            ),
        ),
        save_calibration_service=_param_str(
            node,
            "save_calibration_service",
            _yaml_default(defaults, "save_calibration_service", "/easy_handeye2/calibration/save_calibration"),
        ),
        save_samples_service=_param_str(
            node,
            "save_samples_service",
            _yaml_default(defaults, "save_samples_service", "/easy_handeye2/calibration/save_samples"),
        ),
    )

    motion_config = CollectorMotionConfig(
        move_group_name=move_group_name,
        move_group_ns_fairino=move_group_ns_fairino,
        move_group_ns_kdl=move_group_ns_kdl,
        ik_plugin=ik_plugin,
        planning_pipeline_id=planning_pipeline_id,
        planner_id=planner_id,
        joint_names=tuple(
            _param_list(node, "joint_names", _yaml_default(defaults, "joint_names", _DEFAULT_JOINT_NAMES))
        ),
        original_place_xyz=tuple(
            float(v)
            for v in _param_list(
                node,
                "original_place_xyz",
                _yaml_default(defaults, "original_place_xyz", [0.25, 0.0, 0.23]),
            )
        ),
        original_place_rpy_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "original_place_rpy_deg",
                _yaml_default(defaults, "original_place_rpy_deg", [0.0, 180.0, 0.0]),
            )
        ),
        seed_camera_xyz_m=tuple(
            float(v)
            for v in _param_list(
                node,
                "seed_camera_xyz_m",
                _yaml_default(defaults, "seed_camera_xyz_m", [0.012, -0.030, -0.078]),
            )
        ),
        seed_camera_rpy_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "seed_camera_rpy_deg",
                _yaml_default(defaults, "seed_camera_rpy_deg", [6.0, -86.0, -96.0]),
            )
        ),
        seed_usage_mode=_param_str(
            node,
            "seed_usage_mode",
            _yaml_default(defaults, "seed_usage_mode", "approximate_mount"),
        ),
        workspace_min_xyz=tuple(
            float(v)
            for v in _param_list(
                node,
                "workspace_min_xyz",
                _yaml_default(defaults, "workspace_min_xyz", [0.05, -0.35, 0.02]),
            )
        ),
        workspace_max_xyz=tuple(
            float(v)
            for v in _param_list(
                node,
                "workspace_max_xyz",
                _yaml_default(defaults, "workspace_max_xyz", [0.55, 0.35, 0.45]),
            )
        ),
        preplan_original_place=_param_bool(
            node, "preplan_original_place", _yaml_default(defaults, "preplan_original_place", True)
        ),
        max_velocity=_param_float(node, "max_velocity", _yaml_default(defaults, "max_velocity", 0.1)),
        max_acceleration=_param_float(
            node, "max_acceleration", _yaml_default(defaults, "max_acceleration", 0.10)
        ),
        allowed_planning_time=_param_float(
            node, "allowed_planning_time", _yaml_default(defaults, "allowed_planning_time", 5.0)
        ),
        max_step_size=_param_float(node, "max_step_size", _yaml_default(defaults, "max_step_size", 0.05)),
        position_tolerance=_param_float(
            node, "position_tolerance", _yaml_default(defaults, "position_tolerance", 0.005)
        ),
        orientation_tolerance=_param_float(
            node, "orientation_tolerance", _yaml_default(defaults, "orientation_tolerance", 0.005)
        ),
        allowed_start_tolerance=_param_float(
            node, "allowed_start_tolerance", _yaml_default(defaults, "allowed_start_tolerance", 0.1)
        ),
        action_delay=_param_float(node, "action_delay", _yaml_default(defaults, "action_delay", 0.2)),
        num_candidate_plans=int(
            _param_int(node, "num_candidate_plans", _yaml_default(defaults, "num_candidate_plans", 5))
        ),
        wrist_weight=_param_float(node, "wrist_weight", _yaml_default(defaults, "wrist_weight", 50.0)),
        wrist_joint_indices=tuple(
            int(v)
            for v in _param_list(
                node,
                "wrist_joint_indices",
                _yaml_default(defaults, "wrist_joint_indices", [2, 3, 4]),
            )
        ),
        require_marker_tf=_param_bool(
            node, "require_marker_tf", _yaml_default(defaults, "require_marker_tf", False)
        ),
        settle_time=_param_float(node, "settle_time", _yaml_default(defaults, "settle_time", 1.0)),
        recenter_gain=_param_float(node, "recenter_gain", _yaml_default(defaults, "recenter_gain", 0.55)),
        max_recenter_iters=max(
            0,
            int(_param_int(node, "max_recenter_iters", _yaml_default(defaults, "max_recenter_iters", 4))),
        ),
        recenter_max_step_m=_param_float(
            node, "recenter_max_step_m", _yaml_default(defaults, "recenter_max_step_m", 0.005)
        ),
        recenter_min_step_m=_param_float(
            node, "recenter_min_step_m", _yaml_default(defaults, "recenter_min_step_m", 0.0015)
        ),
        recenter_max_total_translation_m=_param_float(
            node,
            "recenter_max_total_translation_m",
            _yaml_default(defaults, "recenter_max_total_translation_m", 0.015),
        ),
        recenter_improvement_ratio=_param_float(
            node, "recenter_improvement_ratio", _yaml_default(defaults, "recenter_improvement_ratio", 0.90)
        ),
        recenter_axis_frame=_param_str(
            node,
            "recenter_axis_frame",
            _yaml_default(defaults, "recenter_axis_frame", "ee"),
        ),
        recenter_right_sign=_param_float(
            node,
            "recenter_right_sign",
            _yaml_default(defaults, "recenter_right_sign", 1.0),
        ),
        recenter_up_sign=_param_float(
            node,
            "recenter_up_sign",
            _yaml_default(defaults, "recenter_up_sign", 1.0),
        ),
        recenter_depth_scale_gain=_param_float(
            node,
            "recenter_depth_scale_gain",
            _yaml_default(defaults, "recenter_depth_scale_gain", 1.0),
        ),
        recover_last_good_on_marker_loss=_param_bool(
            node,
            "recover_last_good_on_marker_loss",
            _yaml_default(defaults, "recover_last_good_on_marker_loss", True),
        ),
        original_place_attempts=max(
            1,
            int(_param_int(node, "original_place_attempts", _yaml_default(defaults, "original_place_attempts", 3))),
        ),
        original_place_motion_timeout=_param_float(
            node,
            "original_place_motion_timeout",
            _yaml_default(defaults, "original_place_motion_timeout", 30.0),
        ),
        original_place_retry_wait=_param_float(
            node, "original_place_retry_wait", _yaml_default(defaults, "original_place_retry_wait", 2.0)
        ),
        recovery_motion_timeout=_param_float(
            node, "recovery_motion_timeout", _yaml_default(defaults, "recovery_motion_timeout", 30.0)
        ),
        recenter_max_velocity=_param_float(
            node, "recenter_max_velocity", _yaml_default(defaults, "recenter_max_velocity", 0.08)
        ),
        recenter_max_acceleration=_param_float(
            node, "recenter_max_acceleration", _yaml_default(defaults, "recenter_max_acceleration", 0.08)
        ),
        recenter_motion_timeout=_param_float(
            node, "recenter_motion_timeout", _yaml_default(defaults, "recenter_motion_timeout", 20.0)
        ),
        standby_retry_wait=_param_float(
            node, "standby_retry_wait", _yaml_default(defaults, "standby_retry_wait", 1.0)
        ),
        keyboard_poll_period=_param_float(
            node, "keyboard_poll_period", _yaml_default(defaults, "keyboard_poll_period", 0.1)
        ),
        start_wait_poll_period=_param_float(
            node, "start_wait_poll_period", _yaml_default(defaults, "start_wait_poll_period", 0.1)
        ),
    )

    sampling_config = CollectorSamplingConfig(
        marker_timeout=_param_float(node, "marker_timeout", d("marker_timeout", 3.0)),
        marker_recent_timeout=_param_float(node, "marker_recent_timeout", d("marker_recent_timeout", 1.8)),
        min_marker_distance=_param_float(node, "min_marker_distance", d("min_marker_distance", 0.05)),
        max_marker_distance=_param_float(node, "max_marker_distance", d("max_marker_distance", 1.20)),
        marker_size_m=_param_float(node, "marker_size_m", d("marker_size_m", 0.07)),
        min_image_margin_px=_param_float(node, "min_image_margin_px", d("min_image_margin_px", 60.0)),
        min_projected_marker_px=_param_float(node, "min_projected_marker_px", d("min_projected_marker_px", 28.0)),
        startup_min_corner_margin_px=_param_float(node, "startup_min_corner_margin_px", d("startup_min_corner_margin_px", 40.0)),
        min_corner_margin_px=_param_float(node, "min_corner_margin_px", d("min_corner_margin_px", 70.0)),
        min_marker_side_px=_param_float(node, "min_marker_side_px", d("min_marker_side_px", 40.0)),
        max_center_error_px=_param_float(node, "max_center_error_px", d("max_center_error_px", 80.0)),
        visibility_stable_frames=max(1, int(_param_int(node, "visibility_stable_frames", d("visibility_stable_frames", 5)))),
        stable_frame_count=max(1, int(_param_int(node, "stable_frame_count", d("stable_frame_count", 5)))),
        visibility_stable_timeout=_param_float(node, "visibility_stable_timeout", d("visibility_stable_timeout", 7.0)),
        max_center_std_px=_param_float(node, "max_center_std_px", d("max_center_std_px", 12.0)),
        max_depth_std_m=_param_float(node, "max_depth_std_m", d("max_depth_std_m", 0.006)),
        max_angle_std_deg=_param_float(node, "max_angle_std_deg", d("max_angle_std_deg", 2.0)),
        camera_model_max_pixel_error=_param_float(node, "camera_model_max_pixel_error", d("camera_model_max_pixel_error", 50.0)),
        min_successful_samples=max(3, int(_param_int(node, "min_successful_samples", d("min_successful_samples", 20)))),
        max_candidate_attempts=max(1, int(_param_int(node, "max_candidate_attempts", d("max_candidate_attempts", 40)))),
        auto_compute=_param_bool(node, "auto_compute", d("auto_compute", True)),
        auto_save_calibration=_param_bool(node, "auto_save_calibration", d("auto_save_calibration", True)),
        auto_save_samples=_param_bool(node, "auto_save_samples", d("auto_save_samples", True)),
        enable_calibration_sanity_check=_param_bool(node, "enable_calibration_sanity_check", d("enable_calibration_sanity_check", True)),
        validate_calibration_against_tf_mount=_param_bool(node, "validate_calibration_against_tf_mount", d("validate_calibration_against_tf_mount", False)),
        calibration_tf_mount_check_hard_gate=_param_bool(node, "calibration_tf_mount_check_hard_gate", d("calibration_tf_mount_check_hard_gate", False)),
        max_calibration_translation_norm_m=_param_float(node, "max_calibration_translation_norm_m", d("max_calibration_translation_norm_m", 0.30)),
        max_calibration_tf_translation_error_m=_param_float(node, "max_calibration_tf_translation_error_m", d("max_calibration_tf_translation_error_m", 0.02)),
        max_calibration_tf_rotation_error_deg=_param_float(node, "max_calibration_tf_rotation_error_deg", d("max_calibration_tf_rotation_error_deg", 5.0)),
        max_calibration_marker_span_m=_param_float(node, "max_calibration_marker_span_m", d("max_calibration_marker_span_m", 0.02)),
        min_coverage_xy_span_m=_param_float(node, "min_coverage_xy_span_m", d("min_coverage_xy_span_m", 0.04)),
        min_coverage_z_span_m=_param_float(node, "min_coverage_z_span_m", d("min_coverage_z_span_m", 0.06)),
        min_coverage_rotation_span_deg=_param_float(node, "min_coverage_rotation_span_deg", d("min_coverage_rotation_span_deg", 25.0)),
        coverage_stop_extra_samples=max(
            0, int(_param_int(node, "coverage_stop_extra_samples", d("coverage_stop_extra_samples", 6)))
        ),
        coverage_stop_margin_xy_m=_param_float(
            node, "coverage_stop_margin_xy_m", d("coverage_stop_margin_xy_m", 0.012)
        ),
        coverage_stop_margin_z_m=_param_float(
            node, "coverage_stop_margin_z_m", d("coverage_stop_margin_z_m", 0.015)
        ),
        coverage_stop_margin_rot_deg=_param_float(
            node, "coverage_stop_margin_rot_deg", d("coverage_stop_margin_rot_deg", 5.0)
        ),
        sample_min_translation_delta=_param_float(node, "sample_min_translation_delta_m", d("sample_min_translation_delta_m", 0.006)),
        sample_min_rotation_delta_deg=_param_float(node, "sample_min_rotation_delta_deg", d("sample_min_rotation_delta_deg", 3.0)),
        nominal_translation_delta_scale=_param_float(
            node, "nominal_translation_delta_scale", d("nominal_translation_delta_scale", 0.8)
        ),
        nominal_rotation_delta_scale=_param_float(
            node, "nominal_rotation_delta_scale", d("nominal_rotation_delta_scale", 0.6)
        ),
        sampling_base_x_offsets_m=tuple(
            float(v)
            for v in _param_list(
                node,
                "sampling_base_x_offsets_m",
                d("sampling_base_x_offsets_m", [0.0, 0.010, -0.010, 0.020, -0.015, 0.030, -0.020, 0.035, -0.025]),
            )
        ),
        sampling_base_y_offsets_m=tuple(
            float(v)
            for v in _param_list(
                node,
                "sampling_base_y_offsets_m",
                d("sampling_base_y_offsets_m", [0.0, 0.010, -0.010, 0.020, -0.015, 0.030, -0.020, 0.035, -0.025]),
            )
        ),
        sampling_base_z_offsets_m=tuple(
            float(v)
            for v in _param_list(
                node,
                "sampling_base_z_offsets_m",
                d("sampling_base_z_offsets_m", [0.0, 0.015, -0.015, 0.030, -0.030, 0.040, -0.040, 0.050]),
            )
        ),
        sampling_tilt_x_offsets_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "sampling_tilt_x_offsets_deg",
                d("sampling_tilt_x_offsets_deg", [0.0]),
            )
        ),
        sampling_tilt_y_offsets_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "sampling_tilt_y_offsets_deg",
                d("sampling_tilt_y_offsets_deg", [0.0]),
            )
        ),
        sampling_roll_offsets_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "sampling_roll_offsets_deg",
                d("sampling_roll_offsets_deg", [0.0, 6.0, -6.0, 12.0, -12.0, 18.0, -18.0]),
            )
        ),
        get_samples_service_wait_timeout=_param_float(node, "get_samples_service_wait_timeout", d("get_samples_service_wait_timeout", 1.0)),
        get_samples_call_timeout=_param_float(node, "get_samples_call_timeout", d("get_samples_call_timeout", 3.0)),
        remove_samples_service_wait_timeout=_param_float(node, "remove_samples_service_wait_timeout", d("remove_samples_service_wait_timeout", 2.0)),
        remove_samples_call_timeout=_param_float(node, "remove_samples_call_timeout", d("remove_samples_call_timeout", 5.0)),
        take_sample_service_wait_timeout=_param_float(node, "take_sample_service_wait_timeout", d("take_sample_service_wait_timeout", 2.0)),
        take_sample_call_timeout=_param_float(node, "take_sample_call_timeout", d("take_sample_call_timeout", 5.0)),
        empty_service_wait_timeout=_param_float(node, "empty_service_wait_timeout", d("empty_service_wait_timeout", 2.0)),
        save_samples_timeout=_param_float(node, "save_samples_timeout", d("save_samples_timeout", 8.0)),
        compute_calibration_timeout=_param_float(node, "compute_calibration_timeout", d("compute_calibration_timeout", 15.0)),
        save_calibration_timeout=_param_float(node, "save_calibration_timeout", d("save_calibration_timeout", 8.0)),
        moveit_ready_timeout=_param_float(node, "moveit_ready_timeout", d("moveit_ready_timeout", 30.0)),
        moveit_ready_poll_interval=_param_float(node, "moveit_ready_poll_interval", d("moveit_ready_poll_interval", 0.2)),
        candidate_preplan_enabled=_param_bool(node, "candidate_preplan_enabled", d("candidate_preplan_enabled", True)),
        recenter_sign_error_growth_ratio=_param_float(node, "recenter_sign_error_growth_ratio", d("recenter_sign_error_growth_ratio", 1.05)),
        recenter_error_stall_max_iters=max(
            1, int(_param_int(node, "recenter_error_stall_max_iters", d("recenter_error_stall_max_iters", 1)))
        ),
        auto_prune_outlier_samples=_param_bool(
            node, "auto_prune_outlier_samples", d("auto_prune_outlier_samples", True)
        ),
        max_outlier_prune_rounds=max(
            0, int(_param_int(node, "max_outlier_prune_rounds", d("max_outlier_prune_rounds", 3)))
        ),
        outlier_residual_min_m=_param_float(
            node, "outlier_residual_min_m", d("outlier_residual_min_m", 0.015)
        ),
        outlier_residual_keep_ratio=_param_float(
            node, "outlier_residual_keep_ratio", d("outlier_residual_keep_ratio", 1.5)
        ),
    )
    return frames_config, motion_config, sampling_config
