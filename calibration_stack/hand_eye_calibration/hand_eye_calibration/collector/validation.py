"""Calibration validator: sanity checks for hand-eye calibration results.

本模块提供手眼标定结果的合理性验证：
- 基于多个样本对计算标定变换后标记点的重投影残差
- 检查标定结果的平移范数是否过大（可能发散）
- 可选地与 TF 静态挂载真值进行比对
- 支持软/硬两种门控模式
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np


class CalibrationValidator:
    """
    手眼标定验证器。

    执行多项检查来确保标定结果的物理合理性和数值稳定性：
    1. 标记残差检查 —— 标定得到的相机-末端变换应使所有样本中的
       标记在基座坐标系下重合（残差小表示标定准确）。
    2. 平移范数检查 —— 标定结果的平移向量长度应在合理范围内。
    3. TF 挂载真值比对（可选）—— 将标定结果与 URDF 中的静态 TF
       进行比较，如果偏差过大可触发硬失败或仅发出警告。
    """

    def __init__(
        self,
        *,
        enable_calibration_sanity_check: bool,
        validate_calibration_against_tf_mount: bool,
        calibration_tf_mount_check_hard_gate: bool,
        max_calibration_translation_norm_m: float,
        max_calibration_tf_translation_error_m: float,
        max_calibration_tf_rotation_error_deg: float,
        max_calibration_marker_span_m: float,
        logger_warn: Callable[[str], None],
    ):
        """
        初始化验证器参数。

        参数说明：
        - enable_calibration_sanity_check: 是否启用整体合理性检查。
        - validate_calibration_against_tf_mount: 是否与 TF 真值比对。
        - calibration_tf_mount_check_hard_gate: 若为 True，则 TF 比对
          不通过会导致标定失败（硬门控）；否则仅发出警告。
        - max_calibration_translation_norm_m: 允许的最大平移范数（米），
          超出则判定求解发散。
        - max_calibration_tf_translation_error_m: 与 TF 真值比对允许
          的最大平移误差（米）。
        - max_calibration_tf_rotation_error_deg: 与 TF 真值比对允许
          的最大旋转误差（度）。
        - max_calibration_marker_span_m: 标记重投影的最大允许跨度
          和 RMSE（米）。
        - logger_warn: 用于输出警告信息的可调用对象。
        """
        self.enable_calibration_sanity_check = bool(enable_calibration_sanity_check)
        self.validate_calibration_against_tf_mount = bool(validate_calibration_against_tf_mount)
        self.calibration_tf_mount_check_hard_gate = bool(calibration_tf_mount_check_hard_gate)
        self.max_calibration_translation_norm_m = float(max_calibration_translation_norm_m)
        self.max_calibration_tf_translation_error_m = float(max_calibration_tf_translation_error_m)
        self.max_calibration_tf_rotation_error_deg = float(max_calibration_tf_rotation_error_deg)
        self.max_calibration_marker_span_m = float(max_calibration_marker_span_m)
        self._logger_warn = logger_warn

    def calibration_marker_residual(
        self,
        ee_T_cam,
        accepted_sample_poses,
        accepted_tracking_poses,
        compose: Callable,
        rotation_delta_deg: Callable,
    ):
        """
        计算给定相机-末端变换下，所有样本中的标记在基座坐标系中的
        重投影残差。

        原理：
        对于每个样本对 (base_T_ee, cam_T_marker)，通过
            base_T_marker = base_T_ee * ee_T_cam * cam_T_marker
        得到标记在基座坐标系中的理论位姿。
        如果 ee_T_cam 正确，所有样本推算出的 base_T_marker 应该重合。
        残差即各标记点与平均位置的偏离程度。

        参数：
        - ee_T_cam: 末端执行器到相机的变换 (TransformMatrix)
        - accepted_sample_poses: 机器人末端位姿列表 (TransformMatrix)
        - accepted_tracking_poses: 相机观测到的标记位姿列表 (TransformMatrix)
        - compose: 变换组合函数 (a, b) -> a * b
        - rotation_delta_deg: 计算两个旋转之间角距离的函数

        返回：
        - 残差度量字典，包含 xyz_span, span_norm, rmse, max_error,
          max_rot_delta_deg, residuals, mean_xyz
        - 错误信息字符串（如果有）
        """
        if len(accepted_sample_poses) != len(accepted_tracking_poses):
            return None, (
                f"sample pair mismatch robot={len(accepted_sample_poses)} "
                f"tracking={len(accepted_tracking_poses)}"
            )
        if not accepted_sample_poses:
            return None, "no accepted sample pairs"

        # 计算每个样本对应的基座到标记变换
        base_T_markers = []
        for base_T_ee, cam_T_marker in zip(accepted_sample_poses, accepted_tracking_poses):
            base_T_marker = compose(compose(base_T_ee, ee_T_cam), cam_T_marker)
            base_T_markers.append(base_T_marker)

        # 提取所有标记位置（平移部分）
        marker_xyz = np.array([m.translation for m in base_T_markers], dtype=float)
        # 计算所有标记位置的平均值作为理想重合点
        mean_xyz = np.mean(marker_xyz, axis=0)
        # 每个样本点到平均点的欧氏距离
        residuals = np.linalg.norm(marker_xyz - mean_xyz, axis=1)
        # 标记位置的极差（各轴跨度）
        xyz_span = np.ptp(marker_xyz, axis=0)
        # 以第一个标记的旋转为参考，计算所有标记的旋转偏差
        rot_ref = base_T_markers[0].rotation
        rot_deltas = [
            rotation_delta_deg(rot_ref, marker.rotation)
            for marker in base_T_markers
        ]
        return {
            "xyz_span": xyz_span,
            "span_norm": float(np.linalg.norm(xyz_span)),  # 跨度的欧氏长度
            "rmse": float(math.sqrt(np.mean(residuals * residuals))),  # 均方根误差
            "max_error": float(np.max(residuals)),          # 最大偏差
            "max_rot_delta_deg": float(max(rot_deltas)),    # 最大旋转偏差
            "residuals": residuals,
            "mean_xyz": mean_xyz,
        }, ""

    def calibration_sanity_status(
        self,
        calibration,
        *,
        accepted_sample_poses,
        accepted_tracking_poses,
        transform_to_matrix: Callable,
        lookup_tf: Callable,
        compose: Callable,
        rotation_delta_deg: Callable,
        ee_frame: str,
        tracking_base_frame: str,
    ) -> Tuple[bool, str]:
        """
        对标定结果进行完整合理性检查。

        检查步骤：
        1. 检查标定变换矩阵是否包含非有限值（NaN/Inf）。
        2. 检查平移向量的范数是否超过阈值（防止求解发散）。
        3. 计算标记重投影残差，检查跨度范数和 RMSE 是否在允许范围。
        4. （可选）与 TF 挂载真值比较，若超出误差限制：
           - 若 hard_gate=True，则直接判定失败；
           - 否则仅记录警告。

        返回：
        - (是否通过所有检查, 详细描述字符串)
        """
        # 如果禁用了合理性检查，直接返回通过
        if not self.enable_calibration_sanity_check:
            return True, "calibration sanity check disabled"

        # 将标定结果中的变换消息转换为 TransformMatrix
        ee_T_cam = transform_to_matrix(calibration.transform)
        translation = np.array(ee_T_cam.translation, dtype=float)
        quat = ee_T_cam.rotation.as_quat()
        # 检查变换中是否包含 NaN 或 Inf
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quat)):
            return False, "calibration contains non-finite translation or rotation"

        translation_norm = float(np.linalg.norm(translation))
        notes = [f"translation_norm={translation_norm:.3f}m"]
        # 平移范数过大通常意味着求解发散或样本严重退化
        if translation_norm > self.max_calibration_translation_norm_m:
            return (
                False,
                f"{'; '.join(notes)} > {self.max_calibration_translation_norm_m:.3f}m; "
                "solver likely diverged due to a degenerate hand-eye sample set "
                "(for example, too much single-axis rotation and not enough multi-axis orientation excitation)"
            )

        # 计算标记重投影残差
        residual, error = self.calibration_marker_residual(
            ee_T_cam,
            accepted_sample_poses,
            accepted_tracking_poses,
            compose,
            rotation_delta_deg,
        )
        if residual is None:
            return False, error
        notes.append(
            "marker_span="
            f"({residual['xyz_span'][0]:.3f},{residual['xyz_span'][1]:.3f},{residual['xyz_span'][2]:.3f})m"
        )
        notes.append(
            f"marker_span_norm={residual['span_norm']:.3f}m "
            f"marker_rmse={residual['rmse']:.3f}m "
            f"marker_rot_span={residual['max_rot_delta_deg']:.1f}deg"
        )
        # 标记跨度或 RMSE 超出阈值，说明标定精度不足
        if residual["span_norm"] > self.max_calibration_marker_span_m:
            return False, (
                f"{'; '.join(notes)}; marker span exceeds "
                f"{self.max_calibration_marker_span_m:.3f}m"
            )
        if residual["rmse"] > self.max_calibration_marker_span_m:
            return False, (
                f"{'; '.join(notes)}; marker RMSE exceeds "
                f"{self.max_calibration_marker_span_m:.3f}m"
            )

        # 可选的 TF 挂载真值比对
        if self.validate_calibration_against_tf_mount:
            try:
                # 查询末端到跟踪基准的 TF（通常为静态挂载）
                tf_ee_T_cam = lookup_tf(ee_frame, tracking_base_frame, timeout_sec=1.0)
            except Exception as exc:
                return False, f"cannot validate calibration against TF mount: {exc}"
            tf_translation = np.array(tf_ee_T_cam.translation, dtype=float)
            translation_error = float(np.linalg.norm(translation - tf_translation))
            rotation_error = rotation_delta_deg(tf_ee_T_cam.rotation, ee_T_cam.rotation)
            notes.append(f"tf_mount_error={translation_error:.3f}m/{rotation_error:.1f}deg")
            if translation_error > self.max_calibration_tf_translation_error_m:
                warning = (
                    f"{'; '.join(notes)}; TF translation error exceeds "
                    f"{self.max_calibration_tf_translation_error_m:.3f}m"
                )
                if self.calibration_tf_mount_check_hard_gate:
                    return False, warning
                self._logger_warn(f"Calibration TF mount check warning: {warning}")
            if rotation_error > self.max_calibration_tf_rotation_error_deg:
                warning = (
                    f"{'; '.join(notes)}; TF rotation error exceeds "
                    f"{self.max_calibration_tf_rotation_error_deg:.1f}deg"
                )
                if self.calibration_tf_mount_check_hard_gate:
                    return False, warning
                self._logger_warn(f"Calibration TF mount check warning: {warning}")

        return True, f"{'; '.join(notes)}; sanity PASS"