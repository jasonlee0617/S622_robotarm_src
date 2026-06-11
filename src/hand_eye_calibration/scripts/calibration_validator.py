from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np


class CalibrationValidator:
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
        if len(accepted_sample_poses) != len(accepted_tracking_poses):
            return None, (
                f"sample pair mismatch robot={len(accepted_sample_poses)} "
                f"tracking={len(accepted_tracking_poses)}"
            )
        if not accepted_sample_poses:
            return None, "no accepted sample pairs"

        base_T_markers = []
        for base_T_ee, cam_T_marker in zip(accepted_sample_poses, accepted_tracking_poses):
            base_T_marker = compose(compose(base_T_ee, ee_T_cam), cam_T_marker)
            base_T_markers.append(base_T_marker)

        marker_xyz = np.array([m.translation for m in base_T_markers], dtype=float)
        mean_xyz = np.mean(marker_xyz, axis=0)
        residuals = np.linalg.norm(marker_xyz - mean_xyz, axis=1)
        xyz_span = np.ptp(marker_xyz, axis=0)
        rot_ref = base_T_markers[0].rotation
        rot_deltas = [
            rotation_delta_deg(rot_ref, marker.rotation)
            for marker in base_T_markers
        ]
        return {
            "xyz_span": xyz_span,
            "span_norm": float(np.linalg.norm(xyz_span)),
            "rmse": float(math.sqrt(np.mean(residuals * residuals))),
            "max_error": float(np.max(residuals)),
            "max_rot_delta_deg": float(max(rot_deltas)),
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
        if not self.enable_calibration_sanity_check:
            return True, "calibration sanity check disabled"

        ee_T_cam = transform_to_matrix(calibration.transform)
        translation = np.array(ee_T_cam.translation, dtype=float)
        quat = ee_T_cam.rotation.as_quat()
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quat)):
            return False, "calibration contains non-finite translation or rotation"

        translation_norm = float(np.linalg.norm(translation))
        notes = [f"translation_norm={translation_norm:.3f}m"]
        if translation_norm > self.max_calibration_translation_norm_m:
            return False, f"{'; '.join(notes)} > {self.max_calibration_translation_norm_m:.3f}m"

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

        if self.validate_calibration_against_tf_mount:
            try:
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
