"""Pure hand-eye solve, fixed-marker refinement, and compact persistence.

该模块负责从采集到的样本（机器人末端位姿 + 视觉观测位姿）中解算手眼矩阵。
核心流程：
  1. 用 Park、Horaud 求解 AX=XB；Tsai-Lenz 仅输出诊断；
  2. 通过 Park/Horaud 一致性选出最佳初始解；
  3. 基于“标定板在世界坐标系中固定不变”的假设，进行 Gauss-Newton 联合优化；
  4. 计算重投影误差，并迭代剔除离群样本；
  5. 若启用仿真真值，则与 TF 树中的真实变换进行比对校验；
  6. 将结果保存为 EasyHandEye 格式或 YAML 样本文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

from .config import CalibrationType, normalize_calibration_type


@dataclass(frozen=True)
class TransformMatrix:
    """表示一个 4x4 齐次变换矩阵的数据类（使用 scipy 的 Rotation 表示旋转）。"""
    rotation: R
    translation: Tuple[float, float, float]

    def matrix(self) -> np.ndarray:
        """转换为 4x4 numpy 数组。"""
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class CalibrationSample:
    """单次采集的完整样本数据，包含关节目标、机器人末端位姿和视觉追踪位姿。"""
    waypoint_index: int                  # 对应的预置位姿索引
    target_joints_deg: Tuple[float, float, float, float, float, float]  # 目标关节角度（度）
    robot_pose: TransformMatrix          # EIH: base_T_ee; EOB: ee_T_base
    tracking_pose: TransformMatrix       # camera_T_marker (标记相对相机)


def transform_from_matrix(matrix) -> TransformMatrix:
    """从 4x4 numpy 数组转换为 TransformMatrix。"""
    array = np.asarray(matrix, dtype=float)
    return TransformMatrix(R.from_matrix(array[:3, :3]), tuple(float(value) for value in array[:3, 3]))


def robot_pose_for_calibration(base_T_ee: TransformMatrix, calibration_type) -> TransformMatrix:
    """Match EasyHandEye/OpenCV robot-pose semantics for each calibration type."""
    kind = normalize_calibration_type(calibration_type)
    if kind is CalibrationType.EYE_IN_HAND:
        return base_T_ee
    return transform_from_matrix(np.linalg.inv(base_T_ee.matrix()))


def rotation_delta_deg(left: R, right: R) -> float:
    """计算两个旋转之间的角度差（度）。"""
    return math.degrees(float((left.inv() * right).magnitude()))


def sample_coverage(records: Sequence[CalibrationSample]) -> Tuple[float, float]:
    """计算已采集样本中机器人末端位姿的平移跨度和旋转跨度。
    用于评估样本是否覆盖了足够的工作空间。
    """
    translation_span = 0.0
    rotation_span = 0.0
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            translation_span = max(translation_span, float(np.linalg.norm(np.asarray(left.robot_pose.translation) - np.asarray(right.robot_pose.translation))))
            rotation_span = max(rotation_span, rotation_delta_deg(left.robot_pose.rotation, right.robot_pose.rotation))
    return translation_span, rotation_span


def coverage_status(records: Sequence[CalibrationSample], config, *, minimum_count: int) -> Tuple[bool, str]:
    """检查样本的覆盖度是否满足最低要求（数量、最小平移跨度、最小旋转跨度）。"""
    translation_span, rotation_span = sample_coverage(records)
    ok = (
        len(records) >= minimum_count
        and translation_span >= config.minimum_translation_span_m
        and rotation_span >= config.minimum_rotation_span_deg
    )
    return ok, (
        f"count={len(records)}/{minimum_count} translation_span={translation_span:.4f}/"
        f"{config.minimum_translation_span_m:.4f}m rotation_span={rotation_span:.1f}/"
        f"{config.minimum_rotation_span_deg:.1f}deg"
    )


def _methods():
    """返回 OpenCV 支持的手眼标定算法枚举值。"""
    import cv2

    return {
        "OpenCV/Park": cv2.CALIB_HAND_EYE_PARK,
        "OpenCV/Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "OpenCV/Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
    }


_DIAGNOSTIC_ALGORITHM = "OpenCV/Tsai-Lenz"


def _finite_transform(transform: TransformMatrix) -> bool:
    return bool(
        np.all(np.isfinite(transform.rotation.as_matrix()))
        and np.all(np.isfinite(np.asarray(transform.translation, dtype=float)))
    )


def _validate_installation_transform(name: str, transform: TransformMatrix, maximum_norm_m: float) -> None:
    if not _finite_transform(transform):
        raise RuntimeError(f"{name} returned non-finite values")
    if float(np.linalg.norm(transform.translation)) > maximum_norm_m:
        raise RuntimeError(f"{name} camera translation exceeds the installation limit")


def solve_algorithms(records: Sequence[CalibrationSample], algorithms: Iterable[str]) -> dict[str, TransformMatrix]:
    """使用多种算法求解手眼矩阵 AX = XB。

    注意：
      - A: 机器人末端相对基座的变换 (base_T_ee)
      - B: 视觉标记相对相机的变换 (camera_T_marker)
      - X: 相机相对末端的变换 (ee_T_camera)
    """
    import cv2

    if len(records) < 3:
        raise ValueError("at least three samples are required")

    # 提取机器人末端位姿（作为 AX=XB 中的 A 矩阵的平移和旋转）
    robot_rotations = [sample.robot_pose.rotation.as_matrix() for sample in records]
    robot_translations = [np.asarray(sample.robot_pose.translation, dtype=float) for sample in records]

    # 提取视觉标记位姿（作为 AX=XB 中的 B 矩阵的平移和旋转）
    marker_rotations = [sample.tracking_pose.rotation.as_matrix() for sample in records]
    marker_translations = [np.asarray(sample.tracking_pose.translation, dtype=float) for sample in records]

    methods = _methods()
    results = {}
    for name in algorithms:
        if name not in methods:
            raise ValueError(f"unsupported hand-eye algorithm: {name}")
        rotation, translation = cv2.calibrateHandEye(
            robot_rotations, robot_translations, marker_rotations, marker_translations, method=methods[name],
        )
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
        results[name] = transform_from_matrix(matrix)
    return results


def consensus(results: dict[str, TransformMatrix]) -> Tuple[str, TransformMatrix, float, float]:
    """从多种算法结果中选出“共识”解（一致性最高的那个）。

    评判标准：计算每个解与其他所有解的距离（平移误差*100 + 旋转误差），
    选取总距离最小的解作为种子。
    """
    if not results:
        raise ValueError("no calibration transforms were computed")
    items = tuple(results.items())

    def distance(left: TransformMatrix, right: TransformMatrix) -> float:
        # 加权距离：平移误差放大100倍，以平衡平移和旋转的量纲差异
        return 100.0 * float(np.linalg.norm(np.asarray(left.translation) - np.asarray(right.translation))) + rotation_delta_deg(left.rotation, right.rotation)

    name, selected = min(items, key=lambda item: sum(distance(item[1], other) for _, other in items))
    translation_delta = max(float(np.linalg.norm(np.asarray(left.translation) - np.asarray(right.translation))) for _, left in items for _, right in items)
    rotation_delta = max(rotation_delta_deg(left.rotation, right.rotation) for _, left in items for _, right in items)
    return name, selected, translation_delta, rotation_delta


def _marker_reference(records: Sequence[CalibrationSample], handeye: TransformMatrix) -> Tuple[np.ndarray, R]:
    """计算固定标定板在世界坐标系中的参考位姿（作为高斯-牛顿优化的锚点）。

    公式：标定板在世界系下的位姿 = base_T_ee * ee_T_camera * camera_T_marker
    取所有样本的中位数作为固定参考。
    """
    implied = [sample.robot_pose.matrix() @ handeye.matrix() @ sample.tracking_pose.matrix() for sample in records]
    translation = np.median(np.asarray([matrix[:3, 3] for matrix in implied], dtype=float), axis=0)
    rotations = [R.from_matrix(matrix[:3, :3]) for matrix in implied]
    rotation = min(rotations, key=lambda candidate: sum(rotation_delta_deg(candidate, other) for other in rotations))
    return translation, rotation


def marker_metrics(records: Sequence[CalibrationSample], handeye: TransformMatrix) -> dict:
    """计算固定标定板假设下的重投影误差（位置 RMS 和旋转 RMS）。

    用于评估当前手眼矩阵的质量。
    """
    reference_translation, reference_rotation = _marker_reference(records, handeye)
    positions, rotations = [], []
    for sample in records:
        implied = sample.robot_pose.matrix() @ handeye.matrix() @ sample.tracking_pose.matrix()
        positions.append(float(np.linalg.norm(implied[:3, 3] - reference_translation)))
        rotations.append(rotation_delta_deg(reference_rotation, R.from_matrix(implied[:3, :3])))
    return {
        "position_rms_m": float(math.sqrt(np.mean(np.square(positions)))),
        "rotation_rms_deg": float(math.sqrt(np.mean(np.square(rotations)))),
        "per_sample_position_m": positions,
        "per_sample_rotation_deg": rotations,
    }


def _se3_increment(values) -> np.ndarray:
    """将 6 维增量向量（平移 + 旋转向量）转换为 4x4 变换矩阵。"""
    values = np.asarray(values, dtype=float).reshape(6)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = R.from_rotvec(values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def _rotation_vector(matrix) -> np.ndarray:
    """从旋转矩阵提取旋转向量（用于计算残差）。"""
    return R.from_matrix(np.asarray(matrix, dtype=float)).as_rotvec()


def refine_handeye_fixed_marker(records: Sequence[CalibrationSample], seed: TransformMatrix, *, translation_sigma_m: float, rotation_sigma_deg: float, max_iterations: int):
    """固定标定板的高斯-牛顿优化（有限差分法）。

    核心思想：标定板在世界坐标系中位置固定，所以所有样本经过手眼矩阵变换后，
    标定板位姿应该重合。优化变量为 12 维（手眼矩阵 6 自由度 + 标定板位姿 6 自由度）。

    注：这与 WVCSC 算法中的局部求解方案保持一致。
    """
    if len(records) < 3:
        raise ValueError("at least three samples are required for refinement")
    translation_sigma = float(translation_sigma_m)
    rotation_sigma = math.radians(float(rotation_sigma_deg))
    if translation_sigma <= 0.0 or rotation_sigma <= 0.0 or max_iterations < 1:
        raise ValueError("fixed-marker refinement parameters are invalid")

    robots = [sample.robot_pose.matrix() for sample in records]
    markers = [sample.tracking_pose.matrix() for sample in records]
    mount = seed.matrix()

    # 初始化固定标定板的世界位姿（取平均）
    implied = [robot @ mount @ marker for robot, marker in zip(robots, markers)]
    fixed_marker = np.eye(4, dtype=float)
    fixed_marker[:3, 3] = np.mean([item[:3, 3] for item in implied], axis=0)
    fixed_marker[:3, :3] = implied[0][:3, :3]

    def residual(values):
        """计算残差向量：
           位置残差 = (估计的标记世界位姿 - 固定标记位姿) / 平移 sigma
           旋转残差 = (固定标记旋转的逆 * 估计标记旋转) / 旋转 sigma
        """
        camera = mount @ _se3_increment(values[:6])
        fixed = fixed_marker @ _se3_increment(values[6:])
        rows = []
        for robot, marker in zip(robots, markers):
            estimate = robot @ camera @ marker
            rows.extend((estimate[:3, 3] - fixed[:3, 3]) / translation_sigma)
            rows.extend(_rotation_vector(fixed[:3, :3].T @ estimate[:3, :3]) / rotation_sigma)
        return np.asarray(rows, dtype=float)

    values = np.zeros(12, dtype=float)
    initial_cost = float(residual(values) @ residual(values))
    final_cost = initial_cost
    completed = 0
    for iteration in range(int(max_iterations)):
        current = residual(values)
        current_cost = float(current @ current)
        # 有限差分计算雅可比矩阵
        jacobian = np.empty((len(current), len(values)), dtype=float)
        for index in range(len(values)):
            epsilon = 1.0e-6 if index % 6 < 3 else 1.0e-5
            plus, minus = values.copy(), values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            jacobian[:, index] = (residual(plus) - residual(minus)) / (2.0 * epsilon)
        step, *_ = np.linalg.lstsq(jacobian, -current, rcond=None)
        accepted = False
        # 回溯线搜索（Armijo 风格）
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = values + scale * step
            trial_values = residual(trial)
            trial_cost = float(trial_values @ trial_values)
            if math.isfinite(trial_cost) and trial_cost < current_cost:
                values, final_cost, completed, accepted = trial, trial_cost, iteration + 1, True
                break
        if not accepted or float(np.linalg.norm(step)) < 1.0e-8:
            break
    matrix = mount @ _se3_increment(values[:6])
    result = transform_from_matrix(matrix)
    if not np.all(np.isfinite(matrix)) or not math.isfinite(final_cost):
        raise RuntimeError("fixed-marker refinement returned non-finite values")
    return result, {"success": True, "initial_cost": initial_cost, "final_cost": final_cost, "iterations": completed}


def _solve_once(records: Sequence[CalibrationSample], config, *, maximum_translation_norm_m=None):
    """单次求解管道：算法求解 → 共识 → 优化 → 计算指标。"""
    results = solve_algorithms(records, config.algorithm_names)
    maximum_norm = (
        config.maximum_camera_translation_norm_m
        if maximum_translation_norm_m is None
        else float(maximum_translation_norm_m)
    )
    for name, transform in results.items():
        _validate_installation_transform(name, transform, maximum_norm)
    algorithm, seed, translation_delta, rotation_delta = consensus(results)
    diagnostic = "not run"
    try:
        tsai = solve_algorithms(records, (_DIAGNOSTIC_ALGORITHM,))[_DIAGNOSTIC_ALGORITHM]
        if not _finite_transform(tsai):
            diagnostic = "Tsai-Lenz diagnostic returned non-finite values"
        else:
            diagnostic = (
                f"Tsai-Lenz norm={np.linalg.norm(tsai.translation) * 1000.0:.3f}mm "
                f"delta={np.linalg.norm(np.asarray(tsai.translation) - np.asarray(seed.translation)) * 1000.0:.3f}mm/"
                f"{rotation_delta_deg(seed.rotation, tsai.rotation):.3f}deg"
            )
    except Exception as exc:
        diagnostic = f"Tsai-Lenz diagnostic failed: {exc}"
    refined, details = refine_handeye_fixed_marker(
        records,
        seed,
        translation_sigma_m=config.fixed_marker_refinement_translation_sigma_m,
        rotation_sigma_deg=config.fixed_marker_refinement_rotation_sigma_deg,
        max_iterations=config.fixed_marker_refinement_max_iterations,
    )
    details = {
        **details,
        "algorithm_norms_m": {name: float(np.linalg.norm(transform.translation)) for name, transform in results.items()},
        "tsai_diagnostic": diagnostic,
    }
    metrics = marker_metrics(records, refined)
    valid = (
        details["success"]
        and translation_delta <= config.maximum_algorithm_translation_delta_m
        and rotation_delta <= config.maximum_algorithm_rotation_delta_deg
        and metrics["position_rms_m"] <= config.maximum_marker_position_rms_m
        and metrics["rotation_rms_deg"] <= config.maximum_marker_rotation_rms_deg
        and _finite_transform(refined)
        and np.linalg.norm(refined.translation) <= maximum_norm
    )
    return valid, refined, algorithm, translation_delta, rotation_delta, metrics, details


def _worst_sample(metrics: dict, config) -> int:
    """根据固定标定板残差找出最差的样本（用于迭代剔除）。

    评分 = 位置残差/阈值 + 旋转残差/阈值，分值越高越差。
    """
    scores = [
        position / config.maximum_marker_position_rms_m + rotation / config.maximum_marker_rotation_rms_deg
        for position, rotation in zip(metrics["per_sample_position_m"], metrics["per_sample_rotation_deg"])
    ]
    return int(np.argmax(scores))


def _yaml_transform(transform: TransformMatrix) -> dict:
    """将 TransformMatrix 转换为 YAML 友好的字典（平移和四元数）。"""
    quaternion = transform.rotation.as_quat()
    return {
        "translation": {key: float(value) for key, value in zip(("x", "y", "z"), transform.translation)},
        "rotation": {key: float(value) for key, value in zip(("x", "y", "z", "w"), quaternion)},
    }


def _write_yaml(path: Path, data: dict) -> None:
    """原子方式写入 YAML 文件（先用临时文件，再 rename）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _calibration_type(session) -> CalibrationType:
    return normalize_calibration_type(session.frames_config.calibration_type)


def _stem(session) -> Path:
    """生成存储文件的时间戳前缀。"""
    cached = getattr(session, "_collector_output_stem", None)
    if cached is not None:
        return Path(cached)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = Path(session.sampling_config.calibration_output_directory) / (
        f"{session.sampling_config.calibration_file_prefix}_{stamp}_{_calibration_type(session).value}"
    )
    session._collector_output_stem = stem
    return stem


def save_samples(session, records: Sequence[CalibrationSample], status: str) -> str:
    """将采集的样本保存为 YAML 文件（供后续调试或重算使用）。"""
    path = _stem(session).with_suffix(".samples")
    kind = _calibration_type(session)
    robot_pose_key = "base_T_ee" if kind is CalibrationType.EYE_IN_HAND else "ee_T_base"
    _write_yaml(path, {
        "calibration_type": kind.value,
        "status": str(status),
        "samples": [
            {
                "waypoint_index": int(sample.waypoint_index),
                "target_joints_deg": [float(value) for value in sample.target_joints_deg],
                robot_pose_key: _yaml_transform(sample.robot_pose),
                "camera_T_marker": _yaml_transform(sample.tracking_pose),
            }
            for sample in records
        ],
    })
    session.get_logger().info(f"SAMPLES: {path}")
    return str(path)


def _save_calibration(session, transform: TransformMatrix) -> str:
    """将最终标定结果保存为 EasyHandEye2 标准格式。"""
    kind = _calibration_type(session)
    path = _stem(session).with_suffix(".calib")
    _write_yaml(path, {
        "parameters": {
            "name": session.sampling_config.calibration_file_prefix,
            "calibration_type": kind.value,
            "robot_base_frame": session.frames_config.base_frame,
            "robot_effector_frame": session.frames_config.ee_frame,
            "tracking_base_frame": session.frames_config.tracking_base_frame,
            "tracking_marker_frame": session.frames_config.tracking_marker_frame,
            "freehand_robot_movement": True,
            "move_group_namespace": session.motion_config.move_group_ns_fairino or "/",
            "move_group": session.motion_config.move_group_name,
        },
        "transform": _yaml_transform(transform),
    })
    return str(path)


def freeze_simulation_truth(session):
    """从 TF 树中获取仿真环境下的真值（仅当 use_sim_time 为 true 时有效）。

    在 Gazebo 等仿真环境中，相机到基座的变换是已知的，可用于校验标定结果。
    """
    if not bool(getattr(session, "_use_sim_time", False)):
        return None, "real hardware"
    try:
        from rclpy.duration import Duration
        from rclpy.time import Time

        parent = (
            session.frames_config.ee_frame
            if _calibration_type(session) is CalibrationType.EYE_IN_HAND
            else session.frames_config.base_frame
        )
        transform = session.tf_buffer.lookup_transform(
            parent, session.frames_config.tracking_base_frame, Time(), timeout=Duration(seconds=1.0),
        )
        return session.tf_to_matrix(transform), "frozen Fairino mount TF"
    except Exception as exc:
        return None, f"truth TF lookup failed: {exc}"


def truth_status(estimate: TransformMatrix, truth: TransformMatrix, config) -> Tuple[bool, str]:
    """比较估计值与真值，判断是否在允许误差范围内。"""
    delta = np.asarray(estimate.translation) - np.asarray(truth.translation)
    translation = float(np.linalg.norm(delta))
    rotation = rotation_delta_deg(truth.rotation, estimate.rotation)
    ok = (
        translation <= config.ground_truth_max_translation_error_m
        and all(abs(float(value)) <= config.ground_truth_max_axis_error_m for value in delta)
        and rotation <= config.ground_truth_max_rotation_error_deg
    )
    return ok, (
        f"truth dx={delta[0] * 1000.0:.3f}mm dy={delta[1] * 1000.0:.3f}mm dz={delta[2] * 1000.0:.3f}mm "
        f"translation={translation * 1000.0:.3f}mm rotation={rotation:.3f}deg"
    )


def finalize_calibration(session, records: Sequence[CalibrationSample]) -> bool:
    """完整的标定求解主流程：
       1. 获取真值（仿真环境）；
       2. 检查覆盖度；
       3. 尝试求解并计算指标；
       4. 若指标不合格且样本数充足，则剔除最差样本并重试；
       5. 若合格则保存结果。
    """
    config = session.sampling_config
    kind = _calibration_type(session)
    maximum_norm = (
        config.maximum_camera_translation_norm_m
        if kind is CalibrationType.EYE_IN_HAND
        else config.maximum_eye_on_base_camera_translation_norm_m
    )
    retained = list(records)
    truth, truth_note = freeze_simulation_truth(session) if config.ground_truth_check_enabled else (None, "truth check disabled")
    if bool(getattr(session, "_use_sim_time", False)) and config.ground_truth_check_enabled and truth is None:
        session.get_logger().error(f"SOLVE: {truth_note}")
        save_samples(session, retained, "truth_unavailable")
        return False

    # 迭代剔除离群样本（基于固定标记重投影残差）
    while len(retained) >= config.minimum_solution_samples:
        coverage_ok, coverage_note = coverage_status(retained, config, minimum_count=config.minimum_solution_samples)
        if not coverage_ok:
            session.get_logger().error(f"SOLVE: insufficient coverage after pruning: {coverage_note}")
            save_samples(session, retained, "coverage_failed")
            return False
        try:
            valid, transform, algorithm, spread_translation, spread_rotation, metrics, details = _solve_once(
                retained, config, maximum_translation_norm_m=maximum_norm,
            )
        except Exception as exc:
            session.get_logger().error(f"SOLVE: algorithm/refinement failed: {exc}")
            save_samples(session, retained, "solver_failed")
            return False
        hard_norms = ",".join(
            f"{name}={norm * 1000.0:.3f}mm"
            for name, norm in details.get("algorithm_norms_m", {}).items()
        )
        session.get_logger().info(
            f"SOLVE: algorithm={algorithm}+fixed-marker samples={len(retained)} "
            f"hard_norms={hard_norms} "
            f"spread={spread_translation * 1000.0:.3f}mm/{spread_rotation:.3f}deg "
            f"marker_rms={metrics['position_rms_m'] * 1000.0:.3f}mm/{metrics['rotation_rms_deg']:.3f}deg "
            f"iterations={details['iterations']} {details.get('tsai_diagnostic', '')}"
        )
        if valid:
            if truth is not None:
                truth_ok, note = truth_status(transform, truth, config)
                session.get_logger().info(f"GROUND_TRUTH: {note}")
                if not truth_ok:
                    save_samples(session, retained, "truth_failed")
                    return False
            try:
                path = _save_calibration(session, transform)
            except Exception as exc:
                session.get_logger().error(f"CALIBRATION SAVE failed: {exc}")
                save_samples(session, retained, "calibration_save_failed")
                return False
            save_samples(session, retained, "saved")
            session.get_logger().info(f"CALIBRATION SAVED: {path}")
            return True
        if len(retained) == config.minimum_solution_samples:
            session.get_logger().error(f"SOLVE: quality gates failed at the {config.minimum_solution_samples}-sample minimum")
            save_samples(session, retained, "quality_failed")
            return False
        worst = _worst_sample(metrics, config)
        removed = retained.pop(worst)
        session.get_logger().warn(
            f"SOLVE: removing waypoint {removed.waypoint_index} by fixed-marker residual "
            f"({metrics['per_sample_position_m'][worst] * 1000.0:.3f}mm/"
            f"{metrics['per_sample_rotation_deg'][worst]:.3f}deg)"
        )
    save_samples(session, retained, "insufficient_samples")
    return False
