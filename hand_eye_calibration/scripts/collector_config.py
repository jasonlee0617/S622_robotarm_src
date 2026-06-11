from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
    segment_settle_time: float
    segment_step_m: float
    segment_step_deg: float
    recenter_gain: float
    max_recenter_iters: int
    recenter_max_step_m: float
    recenter_min_step_m: float
    recenter_max_total_translation_m: float
    recenter_improvement_ratio: float
    recover_last_good_on_marker_loss: bool
    original_place_target_margin_px: float
    original_place_target_side_px: float
    original_place_target_center_error_px: float
    original_place_search_radius_right_m: float
    original_place_search_radius_up_m: float
    original_place_search_radius_dist_m: float
    original_place_search_step_m: float
    original_place_search_timeout: float
    local_search_radius_right_m: float
    local_search_radius_up_m: float
    local_search_radius_dist_m: float
    local_search_step_m: float
    local_search_timeout: float
    original_place_attempts: int
    original_place_motion_timeout: float
    original_place_retry_wait: float
    recovery_motion_timeout: float
    tune_search_max_velocity: float
    tune_search_max_acceleration: float
    tune_search_motion_timeout: float
    local_search_max_velocity: float
    local_search_max_acceleration: float
    local_search_motion_timeout: float
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
    sample_min_translation_delta: float
    sample_min_rotation_delta_deg: float
    tangent_right_offsets_m: Tuple[float, ...]
    tangent_up_offsets_m: Tuple[float, ...]
    distance_offsets_m: Tuple[float, ...]
    roll_offsets_deg: Tuple[float, ...]
    tilt_x_offsets_deg: Tuple[float, ...]
    tilt_y_offsets_deg: Tuple[float, ...]
    adaptive_right_levels_m: Tuple[float, ...]
    adaptive_up_levels_m: Tuple[float, ...]
    adaptive_dist_levels_m: Tuple[float, ...]
    adaptive_roll_levels_deg: Tuple[float, ...]
    adaptive_tilt_levels_deg: Tuple[float, ...]
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
    rank_visibility_margin_cap_px: float
    rank_visibility_margin_scale_px: float
    rank_visibility_side_cap_px: float
    rank_visibility_side_scale_px: float
    rank_center_penalty_weight: float
    rank_right_coverage_deficit_weight: float
    rank_right_coverage_base_weight: float
    rank_up_coverage_deficit_weight: float
    rank_up_coverage_base_weight: float
    rank_dist_coverage_deficit_weight: float
    rank_dist_coverage_base_weight: float
    rank_rot_coverage_deficit_weight: float
    rank_rot_coverage_base_weight: float
    rank_path_segment_penalty_weight: float
    rank_recenter_cost_penalty_weight: float
    rank_first_candidate_failure_stop: bool
    rank_first_candidate_required_idx: int
    candidate_axis_expand_success_streak: int
    candidate_pair_enable_success_count: int
    candidate_corner_enable_success_count: int
    candidate_corner_cooldown_steps: int
    candidate_axis_failure_penalty_increment: float
    candidate_axis_failure_penalty_decay: float
    candidate_axis_failure_penalty_max: float
    candidate_pair_risk_penalty: float
    candidate_corner_risk_penalty: float
    recenter_sign_error_growth_ratio: float
    recenter_error_stall_max_iters: int


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
    base_frame = _param_str(node, "base_frame", "base_link")
    ee_frame = _param_str(node, "ee_frame", "grasp_frame")
    tracking_base_frame = _param_str(node, "tracking_base_frame", "camera_color_optical_frame")
    tracking_marker_frame = _param_str(node, "tracking_marker_frame", "calibration_aruco")

    move_group_name = _param_str(node, "move_group_name", "robot_arm")
    legacy_move_group_namespace = _param_str(node, "move_group_namespace", "")
    move_group_ns_fairino = _param_str(
        node, "move_group_ns_fairino", legacy_move_group_namespace or "/move_group_fairino"
    )
    move_group_ns_kdl = _param_str(
        node, "move_group_ns_kdl", legacy_move_group_namespace or "/move_group_kdl"
    )
    ik_plugin = PlannerSwitch.normalize_ik(_param_str(node, "ik_plugin", "fairino"))
    planning_pipeline_id = PlannerSwitch.normalize_pipeline(
        _param_str(node, "planning_pipeline_id", "fairino")
    )
    planner_default = "birrt*" if planning_pipeline_id == "fairino" else "RRTConnectFast"
    planner_id = PlannerSwitch.normalize_planner(
        planning_pipeline_id,
        _param_str(node, "planner_id", "") or planner_default,
    )

    frames_config = CollectorFramesConfig(
        base_frame=base_frame,
        ee_frame=ee_frame,
        tracking_base_frame=tracking_base_frame,
        tracking_marker_frame=tracking_marker_frame,
        marker_id=int(_param_int(node, "marker_id", 1)),
        aruco_topic=_param_str(node, "aruco_topic", "/aruco_markers"),
        image_topic=_param_str(node, "image_topic", "/camera/camera/color/image_raw"),
        aruco_dictionary_id=_param_str(node, "aruco_dictionary_id", "DICT_5X5_250"),
        camera_info_topic=_param_str(
            node, "camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"
        ),
        take_sample_service=_param_str(
            node, "take_sample_service", "/easy_handeye2/calibration/take_sample"
        ),
        get_sample_list_service=_param_str(
            node, "get_sample_list_service", "/easy_handeye2/calibration/get_sample_list"
        ),
        remove_sample_service=_param_str(
            node, "remove_sample_service", "/easy_handeye2/calibration/remove_sample"
        ),
        compute_calibration_service=_param_str(
            node, "compute_calibration_service", "/easy_handeye2/calibration/compute_calibration"
        ),
        save_calibration_service=_param_str(
            node, "save_calibration_service", "/easy_handeye2/calibration/save_calibration"
        ),
        save_samples_service=_param_str(
            node, "save_samples_service", "/easy_handeye2/calibration/save_samples"
        ),
    )

    motion_config = CollectorMotionConfig(
        move_group_name=move_group_name,
        move_group_ns_fairino=move_group_ns_fairino,
        move_group_ns_kdl=move_group_ns_kdl,
        ik_plugin=ik_plugin,
        planning_pipeline_id=planning_pipeline_id,
        planner_id=planner_id,
        joint_names=tuple(_param_list(node, "joint_names", _DEFAULT_JOINT_NAMES)),
        original_place_xyz=tuple(
            float(v) for v in _param_list(node, "original_place_xyz", [0.25, 0.0, 0.23])
        ),
        original_place_rpy_deg=tuple(
            float(v) for v in _param_list(node, "original_place_rpy_deg", [0.0, 180.0, 0.0])
        ),
        workspace_min_xyz=tuple(
            float(v) for v in _param_list(node, "workspace_min_xyz", [0.05, -0.35, 0.02])
        ),
        workspace_max_xyz=tuple(
            float(v) for v in _param_list(node, "workspace_max_xyz", [0.55, 0.35, 0.45])
        ),
        preplan_original_place=_param_bool(node, "preplan_original_place", True),
        max_velocity=_param_float(node, "max_velocity", 0.1),
        max_acceleration=_param_float(node, "max_acceleration", 0.10),
        allowed_planning_time=_param_float(node, "allowed_planning_time", 5.0),
        max_step_size=_param_float(node, "max_step_size", 0.05),
        position_tolerance=_param_float(node, "position_tolerance", 0.005),
        orientation_tolerance=_param_float(node, "orientation_tolerance", 0.005),
        allowed_start_tolerance=_param_float(node, "allowed_start_tolerance", 0.1),
        action_delay=_param_float(node, "action_delay", 0.2),
        num_candidate_plans=int(_param_int(node, "num_candidate_plans", 5)),
        wrist_weight=_param_float(node, "wrist_weight", 50.0),
        wrist_joint_indices=tuple(int(v) for v in _param_list(node, "wrist_joint_indices", [2, 3, 4])),
        require_marker_tf=_param_bool(node, "require_marker_tf", False),
        settle_time=_param_float(node, "settle_time", 1.0),
        segment_settle_time=_param_float(node, "segment_settle_time", 0.30),
        segment_step_m=_param_float(node, "segment_step_m", 0.010),
        segment_step_deg=_param_float(node, "segment_step_deg", 5.0),
        recenter_gain=_param_float(node, "recenter_gain", 0.55),
        max_recenter_iters=max(0, int(_param_int(node, "max_recenter_iters", 4))),
        recenter_max_step_m=_param_float(node, "recenter_max_step_m", 0.012),
        recenter_min_step_m=_param_float(node, "recenter_min_step_m", 0.0015),
        recenter_max_total_translation_m=_param_float(
            node, "recenter_max_total_translation_m", 0.030
        ),
        recenter_improvement_ratio=_param_float(node, "recenter_improvement_ratio", 0.90),
        recover_last_good_on_marker_loss=_param_bool(node, "recover_last_good_on_marker_loss", True),
        original_place_target_margin_px=_param_float(node, "original_place_target_margin_px", 150.0),
        original_place_target_side_px=_param_float(node, "original_place_target_side_px", 70.0),
        original_place_target_center_error_px=_param_float(
            node, "original_place_target_center_error_px", 30.0
        ),
        original_place_search_radius_right_m=_param_float(
            node, "original_place_search_radius_right_m", 0.020
        ),
        original_place_search_radius_up_m=_param_float(
            node, "original_place_search_radius_up_m", 0.020
        ),
        original_place_search_radius_dist_m=_param_float(
            node, "original_place_search_radius_dist_m", 0.020
        ),
        original_place_search_step_m=_param_float(node, "original_place_search_step_m", 0.005),
        original_place_search_timeout=_param_float(node, "original_place_search_timeout", 8.0),
        local_search_radius_right_m=_param_float(node, "local_search_radius_right_m", 0.015),
        local_search_radius_up_m=_param_float(node, "local_search_radius_up_m", 0.015),
        local_search_radius_dist_m=_param_float(node, "local_search_radius_dist_m", 0.015),
        local_search_step_m=_param_float(node, "local_search_step_m", 0.005),
        local_search_timeout=_param_float(node, "local_search_timeout", 6.0),
        original_place_attempts=max(1, int(_param_int(node, "original_place_attempts", 3))),
        original_place_motion_timeout=_param_float(node, "original_place_motion_timeout", 30.0),
        original_place_retry_wait=_param_float(node, "original_place_retry_wait", 2.0),
        recovery_motion_timeout=_param_float(node, "recovery_motion_timeout", 30.0),
        tune_search_max_velocity=_param_float(node, "tune_search_max_velocity", 0.06),
        tune_search_max_acceleration=_param_float(node, "tune_search_max_acceleration", 0.06),
        tune_search_motion_timeout=_param_float(node, "tune_search_motion_timeout", 20.0),
        local_search_max_velocity=_param_float(node, "local_search_max_velocity", 0.05),
        local_search_max_acceleration=_param_float(node, "local_search_max_acceleration", 0.05),
        local_search_motion_timeout=_param_float(node, "local_search_motion_timeout", 20.0),
        recenter_max_velocity=_param_float(node, "recenter_max_velocity", 0.08),
        recenter_max_acceleration=_param_float(node, "recenter_max_acceleration", 0.08),
        recenter_motion_timeout=_param_float(node, "recenter_motion_timeout", 20.0),
        standby_retry_wait=_param_float(node, "standby_retry_wait", 1.0),
        keyboard_poll_period=_param_float(node, "keyboard_poll_period", 0.1),
        start_wait_poll_period=_param_float(node, "start_wait_poll_period", 0.1),
    )

    sampling_config = CollectorSamplingConfig(
        marker_timeout=_param_float(node, "marker_timeout", 3.0),
        marker_recent_timeout=_param_float(node, "marker_recent_timeout", 1.8),
        min_marker_distance=_param_float(node, "min_marker_distance", 0.05),
        max_marker_distance=_param_float(node, "max_marker_distance", 1.20),
        marker_size_m=_param_float(node, "marker_size_m", 0.07),
        min_image_margin_px=_param_float(node, "min_image_margin_px", 60.0),
        min_projected_marker_px=_param_float(node, "min_projected_marker_px", 28.0),
        startup_min_corner_margin_px=_param_float(node, "startup_min_corner_margin_px", 40.0),
        min_corner_margin_px=_param_float(node, "min_corner_margin_px", 70.0),
        min_marker_side_px=_param_float(node, "min_marker_side_px", 40.0),
        max_center_error_px=_param_float(node, "max_center_error_px", 80.0),
        visibility_stable_frames=max(1, int(_param_int(node, "visibility_stable_frames", 5))),
        stable_frame_count=max(1, int(_param_int(node, "stable_frame_count", 5))),
        visibility_stable_timeout=_param_float(node, "visibility_stable_timeout", 7.0),
        max_center_std_px=_param_float(node, "max_center_std_px", 12.0),
        max_depth_std_m=_param_float(node, "max_depth_std_m", 0.006),
        max_angle_std_deg=_param_float(node, "max_angle_std_deg", 2.0),
        camera_model_max_pixel_error=_param_float(node, "camera_model_max_pixel_error", 50.0),
        min_successful_samples=max(3, int(_param_int(node, "min_successful_samples", 20))),
        max_candidate_attempts=max(1, int(_param_int(node, "max_candidate_attempts", 40))),
        auto_compute=_param_bool(node, "auto_compute", True),
        auto_save_calibration=_param_bool(node, "auto_save_calibration", True),
        auto_save_samples=_param_bool(node, "auto_save_samples", True),
        enable_calibration_sanity_check=_param_bool(node, "enable_calibration_sanity_check", True),
        validate_calibration_against_tf_mount=_param_bool(
            node, "validate_calibration_against_tf_mount", False
        ),
        calibration_tf_mount_check_hard_gate=_param_bool(
            node, "calibration_tf_mount_check_hard_gate", False
        ),
        max_calibration_translation_norm_m=_param_float(
            node, "max_calibration_translation_norm_m", 0.30
        ),
        max_calibration_tf_translation_error_m=_param_float(
            node, "max_calibration_tf_translation_error_m", 0.02
        ),
        max_calibration_tf_rotation_error_deg=_param_float(
            node, "max_calibration_tf_rotation_error_deg", 5.0
        ),
        max_calibration_marker_span_m=_param_float(node, "max_calibration_marker_span_m", 0.02),
        min_coverage_xy_span_m=_param_float(node, "min_coverage_xy_span_m", 0.04),
        min_coverage_z_span_m=_param_float(node, "min_coverage_z_span_m", 0.06),
        min_coverage_rotation_span_deg=_param_float(node, "min_coverage_rotation_span_deg", 25.0),
        sample_min_translation_delta=_param_float(node, "sample_min_translation_delta_m", 0.015),
        sample_min_rotation_delta_deg=_param_float(node, "sample_min_rotation_delta_deg", 6.0),
        tangent_right_offsets_m=tuple(
            float(v)
            for v in _param_list(
                node, "tangent_right_offsets_m", [0.0, 0.015, -0.015, 0.030, -0.030, 0.045, -0.045]
            )
        ),
        tangent_up_offsets_m=tuple(
            float(v)
            for v in _param_list(
                node, "tangent_up_offsets_m", [0.0, 0.010, -0.010, 0.020, -0.020, 0.030, -0.030]
            )
        ),
        distance_offsets_m=tuple(
            float(v)
            for v in _param_list(node, "distance_offsets_m", [0.0, 0.025, -0.020, 0.050, -0.035])
        ),
        roll_offsets_deg=tuple(
            float(v)
            for v in _param_list(node, "roll_offsets_deg", [0.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0])
        ),
        tilt_x_offsets_deg=tuple(
            float(v) for v in _param_list(node, "tilt_x_offsets_deg", [0.0, 6.0, -6.0])
        ),
        tilt_y_offsets_deg=tuple(
            float(v) for v in _param_list(node, "tilt_y_offsets_deg", [0.0, 6.0, -6.0])
        ),
        adaptive_right_levels_m=tuple(
            float(v) for v in _param_list(node, "adaptive_right_levels_m", [0.005, 0.010, 0.020, 0.030])
        ),
        adaptive_up_levels_m=tuple(
            float(v) for v in _param_list(node, "adaptive_up_levels_m", [0.005, 0.010, 0.020, 0.030])
        ),
        adaptive_dist_levels_m=tuple(
            float(v) for v in _param_list(node, "adaptive_dist_levels_m", [0.010, 0.020, 0.030, 0.040])
        ),
        adaptive_roll_levels_deg=tuple(
            float(v) for v in _param_list(node, "adaptive_roll_levels_deg", [5.0, 6.0, 10.0, 16.0])
        ),
        adaptive_tilt_levels_deg=tuple(
            float(v) for v in _param_list(node, "adaptive_tilt_levels_deg", [2.0, 4.0, 6.0])
        ),
        get_samples_service_wait_timeout=_param_float(node, "get_samples_service_wait_timeout", 1.0),
        get_samples_call_timeout=_param_float(node, "get_samples_call_timeout", 3.0),
        remove_samples_service_wait_timeout=_param_float(node, "remove_samples_service_wait_timeout", 2.0),
        remove_samples_call_timeout=_param_float(node, "remove_samples_call_timeout", 5.0),
        take_sample_service_wait_timeout=_param_float(node, "take_sample_service_wait_timeout", 2.0),
        take_sample_call_timeout=_param_float(node, "take_sample_call_timeout", 5.0),
        empty_service_wait_timeout=_param_float(node, "empty_service_wait_timeout", 2.0),
        save_samples_timeout=_param_float(node, "save_samples_timeout", 8.0),
        compute_calibration_timeout=_param_float(node, "compute_calibration_timeout", 15.0),
        save_calibration_timeout=_param_float(node, "save_calibration_timeout", 8.0),
        moveit_ready_timeout=_param_float(node, "moveit_ready_timeout", 30.0),
        moveit_ready_poll_interval=_param_float(node, "moveit_ready_poll_interval", 0.2),
        candidate_preplan_enabled=_param_bool(node, "candidate_preplan_enabled", True),
        rank_visibility_margin_cap_px=_param_float(node, "rank_visibility_margin_cap_px", 200.0),
        rank_visibility_margin_scale_px=_param_float(node, "rank_visibility_margin_scale_px", 40.0),
        rank_visibility_side_cap_px=_param_float(node, "rank_visibility_side_cap_px", 120.0),
        rank_visibility_side_scale_px=_param_float(node, "rank_visibility_side_scale_px", 30.0),
        rank_center_penalty_weight=_param_float(node, "rank_center_penalty_weight", 0.02),
        rank_right_coverage_deficit_weight=_param_float(node, "rank_right_coverage_deficit_weight", 4.0),
        rank_right_coverage_base_weight=_param_float(node, "rank_right_coverage_base_weight", 1.0),
        rank_up_coverage_deficit_weight=_param_float(node, "rank_up_coverage_deficit_weight", 3.5),
        rank_up_coverage_base_weight=_param_float(node, "rank_up_coverage_base_weight", 0.8),
        rank_dist_coverage_deficit_weight=_param_float(node, "rank_dist_coverage_deficit_weight", 3.0),
        rank_dist_coverage_base_weight=_param_float(node, "rank_dist_coverage_base_weight", 1.0),
        rank_rot_coverage_deficit_weight=_param_float(node, "rank_rot_coverage_deficit_weight", 3.0),
        rank_rot_coverage_base_weight=_param_float(node, "rank_rot_coverage_base_weight", 1.0),
        rank_path_segment_penalty_weight=_param_float(node, "rank_path_segment_penalty_weight", 0.8),
        rank_recenter_cost_penalty_weight=_param_float(node, "rank_recenter_cost_penalty_weight", 0.03),
        rank_first_candidate_failure_stop=_param_bool(node, "rank_first_candidate_failure_stop", True),
        rank_first_candidate_required_idx=max(
            1, int(_param_int(node, "rank_first_candidate_required_idx", 1))
        ),
        candidate_axis_expand_success_streak=max(
            1, int(_param_int(node, "candidate_axis_expand_success_streak", 2))
        ),
        candidate_pair_enable_success_count=max(
            0, int(_param_int(node, "candidate_pair_enable_success_count", 4))
        ),
        candidate_corner_enable_success_count=max(
            0, int(_param_int(node, "candidate_corner_enable_success_count", 8))
        ),
        candidate_corner_cooldown_steps=max(
            0, int(_param_int(node, "candidate_corner_cooldown_steps", 4))
        ),
        candidate_axis_failure_penalty_increment=_param_float(
            node, "candidate_axis_failure_penalty_increment", 1.0
        ),
        candidate_axis_failure_penalty_decay=_param_float(
            node, "candidate_axis_failure_penalty_decay", 0.5
        ),
        candidate_axis_failure_penalty_max=_param_float(node, "candidate_axis_failure_penalty_max", 4.0),
        candidate_pair_risk_penalty=_param_float(node, "candidate_pair_risk_penalty", 0.8),
        candidate_corner_risk_penalty=_param_float(node, "candidate_corner_risk_penalty", 1.5),
        recenter_sign_error_growth_ratio=_param_float(node, "recenter_sign_error_growth_ratio", 1.05),
        recenter_error_stall_max_iters=max(
            1, int(_param_int(node, "recenter_error_stall_max_iters", 2))
        ),
    )
    return frames_config, motion_config, sampling_config
