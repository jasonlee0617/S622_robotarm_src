# 启用 postponed evaluation of annotations，允许使用尚未定义的类型作为注解
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml
from ament_index_python.packages import get_package_share_directory
from manipulation_common.planning.motion_executor import PlannerSwitch

# 导入样本管理中定义的家族名称、执行顺序、以及基础偏移位姿类型
from .sample_types import BaseOffsetPose, FAMILY_EXECUTION_ORDER

# 默认关节名称列表，当 YAML 中未配置时使用
_DEFAULT_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


# ---------------------------------------------------------------------------
# 配置数据类 —— 将 YAML 参数映射为不可变的、带类型的对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectorFramesConfig:
    """坐标系与话题/服务名称配置。"""
    base_frame: str                     # 机器人基座坐标系
    ee_frame: str                       # 末端执行器坐标系
    tracking_base_frame: str            # 跟踪基准坐标系（通常为相机光心）
    tracking_marker_frame: str          # 跟踪标记坐标系（ArUco 标记）
    marker_id: int                      # 目标 ArUco 标记的 ID
    aruco_topic: str                    # ros2_aruco 发布标记位姿的话题
    image_topic: str                    # 原始图像话题
    aruco_dictionary_id: str            # ArUco 字典标识符（如 DICT_5X5_250）
    camera_info_topic: str              # 相机内参话题
    take_sample_service: str            # easy_handeye2 采样服务
    get_sample_list_service: str        # 获取样本列表服务
    get_current_transforms_service: str # 获取当前变换服务
    set_algorithm_service: str          # 设置标定算法服务
    remove_sample_service: str          # 移除样本服务
    compute_calibration_service: str    # 计算标定服务
    save_calibration_service: str       # 保存标定结果服务
    save_samples_service: str           # 保存样本集服务


@dataclass(frozen=True)
class CollectorMotionConfig:
    """机器人运动与重新居中相关配置。"""
    move_group_name: str                        # MoveIt 规划组名称
    move_group_ns_fairino: str                  # Fairino 自定义规划器的 MoveGroup 命名空间
    move_group_ns_kdl: str                      # KDL 规划器的 MoveGroup 命名空间
    ik_plugin: str                              # 当前使用的 IK 插件标识（"fairino" 或 "kdl"）
    planning_pipeline_id: str                   # 规划管线 ID
    planner_id: str                             # 规划器 ID
    joint_names: Tuple[str, ...]                # 关节名称列表
    original_place_xyz: Tuple[float, float, float]          # 原位姿 XYZ 坐标（米）
    original_place_rpy_deg: Tuple[float, float, float]      # 原位姿 RPY 欧拉角（度）
    seed_camera_xyz_m: Tuple[float, float, float]           # 相机在末端坐标系下的种子位置（米）
    seed_camera_rpy_deg: Tuple[float, float, float]         # 相机在末端坐标系下的种子姿态（度）
    seed_usage_mode: str                                    # 种子使用模式（"tf_mount" 或 "approximate_mount"）
    workspace_min_xyz: Tuple[float, float, float]           # 工作空间下限（米）
    workspace_max_xyz: Tuple[float, float, float]           # 工作空间上限（米）
    preplan_original_place: bool                            # 是否对原位姿进行预规划检查
    max_velocity: float                                     # 运动最大速度
    max_acceleration: float                                 # 运动最大加速度
    allowed_planning_time: float                            # 规划超时（秒）
    max_step_size: float                                    # 最大步长
    position_tolerance: float                               # 位置容差（米）
    orientation_tolerance: float                            # 姿态容差（弧度）
    allowed_start_tolerance: float                          # 起始状态容差
    action_delay: float                                     # 每次运动后的额外延迟（秒）
    num_candidate_plans: int                                # 每次规划生成的候选路径数量
    wrist_weight: float                                     # 腕关节评分权重
    wrist_joint_indices: Tuple[int, ...]                    # 腕关节在 joint_names 中的索引
    require_marker_tf: bool                                 # 是否强制要求标记 TF 可用
    settle_time: float                                      # 运动后稳定时间（秒）
    recenter_gain: float                                    # 重新居中的增益系数
    max_recenter_iters: int                                 # 最大重新居中迭代次数
    recenter_max_step_m: float                              # 单步最大移动距离（米）
    recenter_min_step_m: float                              # 单步最小移动距离（米）
    recenter_max_total_translation_m: float                 # 重新居中总允许平移距离（米）
    recenter_max_total_translation_sphere_anchor_m: float   # 锚点家族的重新居中总预算
    recenter_max_total_translation_sphere_height_m: float   # 高度家族的重新居中总预算
    recenter_max_total_translation_sphere_shell_m: float    # 外壳家族的重新居中总预算
    recenter_improvement_ratio: float                       # 每次迭代要求误差下降的比例阈值
    recenter_axis_frame: str                                # 重新居中偏移参考坐标系（"base" 或 "ee"）
    recenter_right_sign: float                              # 右方向符号系数
    recenter_up_sign: float                                 # 上方向符号系数
    recenter_depth_scale_gain: float                        # 深度缩放增益
    precision_recenter_trigger_center_error_px: float       # 触发精度重新居中的中心误差阈值（像素）
    precision_recenter_success_center_error_px: float       # 精度重新居中成功的目标中心误差（像素）
    precision_recenter_max_total_translation_sphere_height_m: float  # 高度家族精度重新居中总平移预算
    precision_recenter_max_total_translation_sphere_shell_m: float   # 外壳家族精度重新居中总平移预算
    recover_last_good_on_marker_loss: bool                  # 标记丢失后是否回到上一个良好位姿
    original_place_attempts: int                            # 移动到原位姿的最大尝试次数
    original_place_motion_timeout: float                    # 原位姿运动超时（秒）
    original_place_retry_wait: float                        # 原位姿重试间隔（秒）
    recovery_motion_timeout: float                          # 恢复运动超时（秒）
    recenter_max_velocity: float                            # 重新居中时的最大速度
    recenter_max_acceleration: float                        # 重新居中时的最大加速度
    recenter_motion_timeout: float                          # 重新居中运动超时（秒）
    standby_retry_wait: float                               # 待机重试间隔（秒）
    keyboard_poll_period: float                             # 键盘轮询间隔（秒）
    start_wait_poll_period: float                           # 等待启动指令的轮询间隔（秒）


@dataclass(frozen=True)
class CollectorSamplingConfig:
    """采样与门控参数配置。"""
    marker_timeout: float                               # 标记可见超时（秒）
    marker_recent_timeout: float                        # 标记观测有效期（秒）
    min_marker_distance: float                          # 标记最小距离（米）
    max_marker_distance: float                          # 标记最大距离（米）
    marker_size_m: float                                # 标记边长（米）
    min_image_margin_px: float                          # 标记投影到图像的最小边缘距离（像素）
    min_projected_marker_px: float                      # 标记投影的最小边长（像素）
    startup_min_corner_margin_px: float                 # 启动/相机模型检查时的最小角点边缘距离（像素）
    min_corner_margin_px: float                         # 正常采样时的最小角点边缘距离（像素）
    min_marker_side_px: float                           # 标记在图像中的最小边长（像素）
    max_center_error_px: float                          # 标记中心偏离图像中心的最大允许像素误差
    visibility_stable_frames: int                       # 话题级稳定所需帧数
    stable_frame_count: int                             # 图像级稳定窗口所需连续成功帧数
    visibility_stable_timeout: float                    # 等待标记稳定的超时（秒）
    max_center_std_px: float                            # 最大中心抖动标准差（像素）
    max_depth_std_m: float                              # 最大深度抖动标准差（米）
    max_angle_std_deg: float                            # 最大角度抖动标准差（度）
    camera_model_max_pixel_error: float                 # 相机模型验证允许的最大重投影像素误差
    precision_gate_enabled: bool                        # 是否启用精度门控
    precision_max_center_error_px: float                # 精度门控允许的最大中心误差（像素）
    precision_coverage_center_error_px: float           # XY 覆盖候选放宽后的最大中心误差（像素）
    precision_max_camera_model_error_px: float          # 精度门控允许的最大相机模型误差（像素）
    precision_max_center_std_px: float                  # 精度门控允许的最大中心标准差（像素）
    precision_max_depth_std_m: float                    # 精度门控允许的最大深度标准差（米）
    precision_max_angle_std_deg: float                  # 精度门控允许的最大角度标准差（度）
    precision_reject_non_strict_recenter_non_anchor: bool  # 对于非锚点候选，未严格收敛是否拒绝
    min_successful_samples: int                         # 最少成功样本数
    max_candidate_attempts: int                         # 最多尝试的候选位姿数
    auto_compute: bool                                  # 采集结束后是否自动计算标定
    auto_save_calibration: bool                         # 是否自动保存标定结果
    auto_save_samples: bool                             # 是否自动保存样本集
    enable_calibration_sanity_check: bool               # 是否启用标定合理性检查
    validate_calibration_against_tf_mount: bool         # 是否与 TF 挂载真值比对
    calibration_tf_mount_check_hard_gate: bool          # TF 比对不通过时是否硬失败
    max_calibration_translation_norm_m: float           # 标定平移范数上限（米）
    max_calibration_tf_translation_error_m: float       # 与 TF 真值比对允许的最大平移误差（米）
    max_calibration_tf_rotation_error_deg: float        # 与 TF 真值比对允许的最大旋转误差（度）
    max_calibration_marker_span_m: float                # 标记残差跨度/均方根上限（米）
    min_coverage_xy_span_m: float                       # 最小 XY 平面覆盖跨度（米）
    min_coverage_z_span_m: float                        # 最小 Z 轴覆盖跨度（米）
    min_coverage_rotation_span_deg: float               # 最小旋转覆盖跨度（度）
    sample_min_translation_delta: float                 # 样本间最小平移增量（米）
    sample_min_rotation_delta_deg: float                # 样本间最小旋转增量（度）
    orientation_sample_min_rotation_delta_deg: float    # 纯方向样本间最小旋转增量（度）
    nominal_translation_delta_scale: float              # 标称平移增量缩放因子
    nominal_rotation_delta_scale: float                 # 标称旋转增量缩放因子
    base_offsets: Dict[str, List[BaseOffsetPose]]       # 以家族为键的基础偏移位姿配置
    min_pitch_span_deg: float                           # 可观测性要求的最小俯仰角跨度（度）
    min_yaw_span_deg: float                             # 可观测性要求的最小偏航角跨度（度）
    min_roll_span_deg: float                            # 可观测性要求的最小滚转角跨度（度）
    min_sphere_anchor_samples: int                      # 最少球体锚点样本数
    min_sphere_height_samples: int                      # 最少球体高度样本数
    min_sphere_shell_samples: int                       # 最少球体外壳样本数
    solver_subset_min_samples: int                      # 求解器子集最少样本数
    solver_subset_max_samples: int                      # 求解器子集最多样本数
    max_successful_samples: int                         # 成功样本数软上限
    absolute_max_successful_samples: int                # 成功样本数绝对上限
    calibration_algorithms: Tuple[str, ...]             # 本地标定求解算法列表
    sample_consistency_max_translation_m: float         # 采样一致性检查最大平移差（米）
    sample_consistency_max_rotation_deg: float          # 采样一致性检查最大旋转差（度）
    sample_consistency_timeout: float                   # 采样一致性预检超时（秒）
    recenter_weak_allowance_sphere_anchor_pitch: int    # 球体锚点俯仰方向重新居中允许的弱收敛次数
    # 各种服务等待/调用超时
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
    moveit_ready_timeout: float                         # MoveIt 就绪等待超时（秒）
    moveit_ready_poll_interval: float                   # MoveIt 就绪轮询间隔（秒）
    candidate_preplan_enabled: bool                     # 是否启用候选位姿预规划
    recenter_sign_error_growth_ratio: float             # 重新居中方向错误的误差增长比率阈值
    recenter_error_stall_max_iters: int                 # 重新居中误差停滞的最大允许迭代次数
    auto_prune_outlier_samples: bool                    # 是否自动修剪离群样本


# ---------------------------------------------------------------------------
# YAML 默认值加载
# ---------------------------------------------------------------------------

def _load_yaml_defaults() -> dict:
    """尝试从包共享目录或本地 config 目录加载默认 YAML 参数。"""
    candidate_paths = []
    try:
        # 优先从安装后的 share 目录中查找
        candidate_paths.append(
            os.path.join(
                get_package_share_directory("hand_eye_calibration"),
                "config",
                "auto_calibration_collector.yaml",
            )
        )
    except Exception:
        pass
    # 其次从源码目录的 config 子文件夹中查找
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
        # YAML 结构: auto_calibration_collector.ros__parameters
        params = data.get("auto_calibration_collector", {}).get("ros__parameters", {})
        if isinstance(params, dict):
            return params
    return {}


# ---------------------------------------------------------------------------
# 参数声明与获取辅助函数
# ---------------------------------------------------------------------------

def _param_str(node, name: str, default: str) -> str:
    """声明并获取字符串类型参数。"""
    node.declare_parameter(name, default)
    return str(node.get_parameter(name).value)


def _param_float(node, name: str, default: float) -> float:
    """声明并获取浮点类型参数。"""
    node.declare_parameter(name, default)
    return float(node.get_parameter(name).value)


def _param_int(node, name: str, default: int) -> int:
    """声明并获取整数类型参数。"""
    node.declare_parameter(name, default)
    return int(node.get_parameter(name).value)


def _param_bool(node, name: str, default: bool) -> bool:
    """声明并获取布尔类型参数，支持字符串形式的布尔值。"""
    node.declare_parameter(name, default)
    value = node.get_parameter(name).value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _param_list(node, name: str, default: List) -> List:
    """声明并获取列表类型参数。"""
    node.declare_parameter(name, default)
    value = node.get_parameter(name).value
    if value is None:
        return list(default)
    return list(value)


# ---------------------------------------------------------------------------
# 候选家族元数据
# ---------------------------------------------------------------------------

# 候选家族的采集顺序，与 sample_types 中定义一致
_FAMILY_ORDER = list(FAMILY_EXECUTION_ORDER)

# 每个家族对应的标签（即家族名本身）
_FAMILY_LABEL = {
    "sphere_anchor": "sphere_anchor",
    "sphere_height": "sphere_height",
    "sphere_shell": "sphere_shell",
    "sphere_roll_coverage": "sphere_roll_coverage",
}

# 各家族样本默认可移除性：锚点和高度样本不可移除，外壳和滚转覆盖可移除
_FAMILY_REMOVABLE = {
    "sphere_anchor": False,
    "sphere_shell": True,
    "sphere_height": False,
    "sphere_roll_coverage": True,
}

# 各家族的设计意图说明
_FAMILY_INTENT = {
    "sphere_anchor": "orientation_excitation",
    "sphere_shell": "shell_translation_observability",
    "sphere_height": "depth_baseline",
    "sphere_roll_coverage": "rotation_coverage",
}


def _parse_base_offsets(raw: dict) -> Dict[str, List[BaseOffsetPose]]:
    """将 YAML 中的 raw base_offsets 字典转换为类型化的 BaseOffsetPose 列表。

    为每个 entry 自动生成 label（如果未显式提供），并填入家族元数据。
    """
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, List[BaseOffsetPose]] = {}
    for family_name in _FAMILY_ORDER:
        entries = raw.get(family_name)
        if not isinstance(entries, list):
            continue
        family_label = _FAMILY_LABEL.get(family_name, family_name)
        default_removable = _FAMILY_REMOVABLE.get(family_name, True)
        intent = _FAMILY_INTENT.get(family_name, "")
        family_list: List[BaseOffsetPose] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            bx = float(entry.get("base_x", 0.0))
            by = float(entry.get("base_y", 0.0))
            bz = float(entry.get("base_z", 0.0))
            pitch = float(entry.get("pitch", 0.0))
            yaw = float(entry.get("yaw", 0.0))
            roll = float(entry.get("roll", 0.0))

            removable = bool(entry.get("removable", default_removable))

            # 自动生成 label（若未显式提供）
            label = entry.get("label", "")
            if not label:
                parts = []
                if abs(bx) > 1e-9:
                    parts.append(f"x{bx:+.3f}")
                if abs(by) > 1e-9:
                    parts.append(f"y{by:+.3f}")
                if abs(bz) > 1e-9:
                    parts.append(f"z{bz:+.3f}")
                if abs(pitch) > 1e-9:
                    parts.append(f"p{pitch:+.1f}")
                if abs(yaw) > 1e-9:
                    parts.append(f"w{yaw:+.1f}")
                if abs(roll) > 1e-9:
                    parts.append(f"r{roll:+.1f}")
                label = "_".join(parts) if parts else "center"

            obs_axis = str(entry.get("observability_axis", "none")).strip().lower()
            dedup_prot = bool(entry.get("dedup_protected", False))

            family_list.append(
                BaseOffsetPose(
                    label=label,
                    family=family_label,
                    base_x=bx,
                    base_y=by,
                    base_z=bz,
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    removable=removable,
                    intent=intent,
                    observability_axis=obs_axis,
                    dedup_protected=dedup_prot,
                )
            )
        if family_list:
            result[family_name] = family_list
    return result


def _load_frames_config(node, d):
    """加载坐标系与服务名称配置。"""
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

    return CollectorFramesConfig(
        base_frame=base_frame,
        ee_frame=ee_frame,
        tracking_base_frame=tracking_base_frame,
        tracking_marker_frame=tracking_marker_frame,
        marker_id=int(_param_int(node, "marker_id", d("marker_id", 1))),
        aruco_topic=_param_str(node, "aruco_topic", d("aruco_topic", "/aruco_markers")),
        image_topic=_param_str(
            node, "image_topic", d("image_topic", "/camera/camera/color/image_raw")
        ),
        aruco_dictionary_id=_param_str(
            node,
            "aruco_dictionary_id",
            d("aruco_dictionary_id", "DICT_5X5_250"),
        ),
        camera_info_topic=_param_str(
            node,
            "camera_info_topic",
            d("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info"),
        ),
        take_sample_service=_param_str(
            node,
            "take_sample_service",
            d("take_sample_service", "/easy_handeye2/calibration/take_sample"),
        ),
        get_sample_list_service=_param_str(
            node,
            "get_sample_list_service",
            d("get_sample_list_service", "/easy_handeye2/calibration/get_sample_list"),
        ),
        get_current_transforms_service=_param_str(
            node,
            "get_current_transforms_service",
            d("get_current_transforms_service", "/easy_handeye2/calibration/get_current_transforms"),
        ),
        set_algorithm_service=_param_str(
            node,
            "set_algorithm_service",
            d("set_algorithm_service", "/easy_handeye2/calibration/set_algorithm"),
        ),
        remove_sample_service=_param_str(
            node,
            "remove_sample_service",
            d("remove_sample_service", "/easy_handeye2/calibration/remove_sample"),
        ),
        compute_calibration_service=_param_str(
            node,
            "compute_calibration_service",
            d("compute_calibration_service",
                "/easy_handeye2/calibration/compute_calibration",
            ),
        ),
        save_calibration_service=_param_str(
            node,
            "save_calibration_service",
            d("save_calibration_service", "/easy_handeye2/calibration/save_calibration"),
        ),
        save_samples_service=_param_str(
            node,
            "save_samples_service",
            d("save_samples_service", "/easy_handeye2/calibration/save_samples"),
        ),
    )


def _load_motion_config(node, d):
    """加载运动控制相关配置。"""
    move_group_name = _param_str(node, "move_group_name", d("move_group_name", "robot_arm"))
    move_group_ns_fairino = _param_str(
        node,
        "move_group_ns_fairino",
        d("move_group_ns_fairino", "/move_group_fairino"),
    )
    move_group_ns_kdl = _param_str(
        node,
        "move_group_ns_kdl",
        d("move_group_ns_kdl", "/move_group_kdl"),
    )
    # 规范化 IK 插件名称
    ik_plugin = PlannerSwitch.normalize_ik(
        _param_str(node, "ik_plugin", d("ik_plugin", "fairino"))
    )
    # 规范化规划管线 ID
    planning_pipeline_id = PlannerSwitch.normalize_pipeline(
        _param_str(node, "planning_pipeline_id", d("planning_pipeline_id", "fairino"))
    )
    # 根据管线自动选择默认规划器
    planner_default = "birrt*" if planning_pipeline_id == "fairino" else "RRTConnectFast"
    planner_id = PlannerSwitch.normalize_planner(
        planning_pipeline_id,
        _param_str(node, "planner_id", d("planner_id", "")) or planner_default,
    )
    if not PlannerSwitch.is_valid(planning_pipeline_id, planner_id):
        raise ValueError(
            "Unsupported planner config: "
            f"pipeline={planning_pipeline_id}, planner={planner_id}"
        )

    return CollectorMotionConfig(
        move_group_name=move_group_name,
        move_group_ns_fairino=move_group_ns_fairino,
        move_group_ns_kdl=move_group_ns_kdl,
        ik_plugin=ik_plugin,
        planning_pipeline_id=planning_pipeline_id,
        planner_id=planner_id,
        joint_names=tuple(
            _param_list(node, "joint_names", d("joint_names", _DEFAULT_JOINT_NAMES))
        ),
        original_place_xyz=tuple(
            float(v)
            for v in _param_list(
                node,
                "original_place_xyz",
                d("original_place_xyz", [0.25, 0.0, 0.23]),
            )
        ),
        original_place_rpy_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "original_place_rpy_deg",
                d("original_place_rpy_deg", [0.0, 180.0, 0.0]),
            )
        ),
        seed_camera_xyz_m=tuple(
            float(v)
            for v in _param_list(
                node,
                "seed_camera_xyz_m",
                d("seed_camera_xyz_m", [0.012, -0.030, -0.078]),
            )
        ),
        seed_camera_rpy_deg=tuple(
            float(v)
            for v in _param_list(
                node,
                "seed_camera_rpy_deg",
                d("seed_camera_rpy_deg", [6.0, -86.0, -96.0]),
            )
        ),
        seed_usage_mode=_param_str(
            node,
            "seed_usage_mode",
            d("seed_usage_mode", "approximate_mount"),
        ),
        workspace_min_xyz=tuple(
            float(v)
            for v in _param_list(
                node,
                "workspace_min_xyz",
                d("workspace_min_xyz", [0.05, -0.35, 0.02]),
            )
        ),
        workspace_max_xyz=tuple(
            float(v)
            for v in _param_list(
                node,
                "workspace_max_xyz",
                d("workspace_max_xyz", [0.55, 0.35, 0.45]),
            )
        ),
        preplan_original_place=_param_bool(
            node, "preplan_original_place", d("preplan_original_place", True)
        ),
        max_velocity=_param_float(node, "max_velocity", d("max_velocity", 0.1)),
        max_acceleration=_param_float(
            node, "max_acceleration", d("max_acceleration", 0.10)
        ),
        allowed_planning_time=_param_float(
            node, "allowed_planning_time", d("allowed_planning_time", 5.0)
        ),
        max_step_size=_param_float(node, "max_step_size", d("max_step_size", 0.05)),
        position_tolerance=_param_float(
            node, "position_tolerance", d("position_tolerance", 0.005)
        ),
        orientation_tolerance=_param_float(
            node, "orientation_tolerance", d("orientation_tolerance", 0.005)
        ),
        allowed_start_tolerance=_param_float(
            node, "allowed_start_tolerance", d("allowed_start_tolerance", 0.1)
        ),
        action_delay=_param_float(node, "action_delay", d("action_delay", 0.2)),
        num_candidate_plans=int(
            _param_int(node, "num_candidate_plans", d("num_candidate_plans", 5))
        ),
        wrist_weight=_param_float(node, "wrist_weight", d("wrist_weight", 50.0)),
        wrist_joint_indices=tuple(
            int(v)
            for v in _param_list(
                node,
                "wrist_joint_indices",
                d("wrist_joint_indices", [2, 3, 4]),
            )
        ),
        require_marker_tf=_param_bool(
            node, "require_marker_tf", d("require_marker_tf", False)
        ),
        settle_time=_param_float(node, "settle_time", d("settle_time", 1.0)),
        recenter_gain=_param_float(node, "recenter_gain", d("recenter_gain", 0.55)),
        max_recenter_iters=max(
            0,
            int(_param_int(node, "max_recenter_iters", d("max_recenter_iters", 4))),
        ),
        recenter_max_step_m=_param_float(
            node, "recenter_max_step_m", d("recenter_max_step_m", 0.005)
        ),
        recenter_min_step_m=_param_float(
            node, "recenter_min_step_m", d("recenter_min_step_m", 0.0015)
        ),
        recenter_max_total_translation_m=_param_float(
            node,
            "recenter_max_total_translation_m",
            d("recenter_max_total_translation_m", 0.015),
        ),
        recenter_max_total_translation_sphere_anchor_m=_param_float(
            node,
            "recenter_max_total_translation_sphere_anchor_m",
            d("recenter_max_total_translation_sphere_anchor_m", 0.040),
        ),
        recenter_max_total_translation_sphere_height_m=_param_float(
            node,
            "recenter_max_total_translation_sphere_height_m",
            d("recenter_max_total_translation_sphere_height_m", 0.020),
        ),
        recenter_max_total_translation_sphere_shell_m=_param_float(
            node,
            "recenter_max_total_translation_sphere_shell_m",
            d("recenter_max_total_translation_sphere_shell_m", 0.020),
        ),
        recenter_improvement_ratio=_param_float(
            node, "recenter_improvement_ratio", d("recenter_improvement_ratio", 0.90)
        ),
        recenter_axis_frame=_param_str(
            node,
            "recenter_axis_frame",
            d("recenter_axis_frame", "ee"),
        ),
        recenter_right_sign=_param_float(
            node,
            "recenter_right_sign",
            d("recenter_right_sign", 1.0),
        ),
        recenter_up_sign=_param_float(
            node,
            "recenter_up_sign",
            d("recenter_up_sign", 1.0),
        ),
        recenter_depth_scale_gain=_param_float(
            node,
            "recenter_depth_scale_gain",
            d("recenter_depth_scale_gain", 1.0),
        ),
        precision_recenter_trigger_center_error_px=_param_float(
            node,
            "precision_recenter_trigger_center_error_px",
            d("precision_recenter_trigger_center_error_px", 45.0),
        ),
        precision_recenter_success_center_error_px=_param_float(
            node,
            "precision_recenter_success_center_error_px",
            d("precision_recenter_success_center_error_px", 35.0),
        ),
        precision_recenter_max_total_translation_sphere_height_m=_param_float(
            node,
            "precision_recenter_max_total_translation_sphere_height_m",
            d("precision_recenter_max_total_translation_sphere_height_m", 0.025),
        ),
        precision_recenter_max_total_translation_sphere_shell_m=_param_float(
            node,
            "precision_recenter_max_total_translation_sphere_shell_m",
            d("precision_recenter_max_total_translation_sphere_shell_m", 0.030),
        ),
        recover_last_good_on_marker_loss=_param_bool(
            node,
            "recover_last_good_on_marker_loss",
            d("recover_last_good_on_marker_loss", True),
        ),
        original_place_attempts=max(
            1,
            int(_param_int(node, "original_place_attempts", d("original_place_attempts", 3))),
        ),
        original_place_motion_timeout=_param_float(
            node,
            "original_place_motion_timeout",
            d("original_place_motion_timeout", 30.0),
        ),
        original_place_retry_wait=_param_float(
            node, "original_place_retry_wait", d("original_place_retry_wait", 2.0)
        ),
        recovery_motion_timeout=_param_float(
            node, "recovery_motion_timeout", d("recovery_motion_timeout", 30.0)
        ),
        recenter_max_velocity=_param_float(
            node, "recenter_max_velocity", d("recenter_max_velocity", 0.08)
        ),
        recenter_max_acceleration=_param_float(
            node, "recenter_max_acceleration", d("recenter_max_acceleration", 0.08)
        ),
        recenter_motion_timeout=_param_float(
            node, "recenter_motion_timeout", d("recenter_motion_timeout", 20.0)
        ),
        standby_retry_wait=_param_float(
            node, "standby_retry_wait", d("standby_retry_wait", 1.0)
        ),
        keyboard_poll_period=_param_float(
            node, "keyboard_poll_period", d("keyboard_poll_period", 0.1)
        ),
        start_wait_poll_period=_param_float(
            node, "start_wait_poll_period", d("start_wait_poll_period", 0.1)
        ),
    )


def _load_sampling_config(node, d, base_offsets):
    """加载采样与门控相关配置。"""
    return CollectorSamplingConfig(
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
        camera_model_max_pixel_error=_param_float(
            node, "camera_model_max_pixel_error", d("camera_model_max_pixel_error", 50.0)
        ),
        precision_gate_enabled=_param_bool(
            node, "precision_gate_enabled", d("precision_gate_enabled", True)
        ),
        precision_max_center_error_px=_param_float(
            node, "precision_max_center_error_px", d("precision_max_center_error_px", 50.0)
        ),
        precision_coverage_center_error_px=_param_float(
            node,
            "precision_coverage_center_error_px",
            d("precision_coverage_center_error_px", 75.0),
        ),
        precision_max_camera_model_error_px=_param_float(
            node,
            "precision_max_camera_model_error_px",
            d("precision_max_camera_model_error_px", 12.0),
        ),
        precision_max_center_std_px=_param_float(
            node, "precision_max_center_std_px", d("precision_max_center_std_px", 4.0)
        ),
        precision_max_depth_std_m=_param_float(
            node, "precision_max_depth_std_m", d("precision_max_depth_std_m", 0.0025)
        ),
        precision_max_angle_std_deg=_param_float(
            node, "precision_max_angle_std_deg", d("precision_max_angle_std_deg", 0.8)
        ),
        precision_reject_non_strict_recenter_non_anchor=_param_bool(
            node,
            "precision_reject_non_strict_recenter_non_anchor",
            d("precision_reject_non_strict_recenter_non_anchor", True),
        ),
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
        sample_min_translation_delta=_param_float(node, "sample_min_translation_delta_m", d("sample_min_translation_delta_m", 0.006)),
        sample_min_rotation_delta_deg=_param_float(node, "sample_min_rotation_delta_deg", d("sample_min_rotation_delta_deg", 3.0)),
        orientation_sample_min_rotation_delta_deg=_param_float(node, "orientation_sample_min_rotation_delta_deg", d("orientation_sample_min_rotation_delta_deg", 2.0)),
        nominal_translation_delta_scale=_param_float(
            node, "nominal_translation_delta_scale", d("nominal_translation_delta_scale", 0.8)
        ),
        nominal_rotation_delta_scale=_param_float(
            node, "nominal_rotation_delta_scale", d("nominal_rotation_delta_scale", 0.6)
        ),
        base_offsets=base_offsets,
        min_pitch_span_deg=_param_float(node, "min_pitch_span_deg", d("min_pitch_span_deg", 4.0)),
        min_yaw_span_deg=_param_float(node, "min_yaw_span_deg", d("min_yaw_span_deg", 4.0)),
        min_roll_span_deg=_param_float(node, "min_roll_span_deg", d("min_roll_span_deg", 10.0)),
        min_sphere_anchor_samples=max(
            1, int(_param_int(node, "min_sphere_anchor_samples", d("min_sphere_anchor_samples", 4)))
        ),
        min_sphere_height_samples=max(
            1, int(_param_int(node, "min_sphere_height_samples", d("min_sphere_height_samples", 3)))
        ),
        min_sphere_shell_samples=max(
            1, int(_param_int(node, "min_sphere_shell_samples", d("min_sphere_shell_samples", 4)))
        ),
        solver_subset_min_samples=max(
            6, int(_param_int(node, "solver_subset_min_samples", d("solver_subset_min_samples", 12)))
        ),
        solver_subset_max_samples=max(
            6, int(_param_int(node, "solver_subset_max_samples", d("solver_subset_max_samples", 18)))
        ),
        max_successful_samples=max(
            1, int(_param_int(node, "max_successful_samples", d("max_successful_samples", 22)))
        ),
        absolute_max_successful_samples=max(
            1,
            int(
                _param_int(
                    node,
                    "absolute_max_successful_samples",
                    d("absolute_max_successful_samples", 28),
                )
            ),
        ),
        calibration_algorithms=tuple(
            str(v) for v in _param_list(
                node, "calibration_algorithms",
                d("calibration_algorithms", ["Park", "Horaud", "Tsai-Lenz"])
            )
        ),
        sample_consistency_max_translation_m=_param_float(
            node, "sample_consistency_max_translation_m", d("sample_consistency_max_translation_m", 0.002)
        ),
        sample_consistency_max_rotation_deg=_param_float(
            node, "sample_consistency_max_rotation_deg", d("sample_consistency_max_rotation_deg", 0.5)
        ),
        sample_consistency_timeout=_param_float(
            node, "sample_consistency_timeout", d("sample_consistency_timeout", 0.5)
        ),
        recenter_weak_allowance_sphere_anchor_pitch=max(
            0, int(_param_int(node, "recenter_weak_allowance_sphere_anchor_pitch", d("recenter_weak_allowance_sphere_anchor_pitch", 2)))
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
    )


def load_collector_config(node):
    """加载完整的采集器配置，返回三个不可变配置对象。

    顺序：先加载 YAML 默认值作为回退，再从 ROS 参数服务器读取实际值，
    最后组装为 CollectorFramesConfig, CollectorMotionConfig, CollectorSamplingConfig。
    """
    defaults = _load_yaml_defaults()
    d = defaults.get  # 快捷方式：从默认字典中取值

    # 解析基础偏移配置（必须存在，否则报错）
    raw_offsets = d("base_offsets", {})
    base_offsets = _parse_base_offsets(raw_offsets)
    if not base_offsets:
        raise RuntimeError(
            "base_offsets is empty or missing in auto_calibration_collector.yaml. "
            "The family-based config is required."
        )

    return (
        _load_frames_config(node, d),
        _load_motion_config(node, d),
        _load_sampling_config(node, d, base_offsets),
    )