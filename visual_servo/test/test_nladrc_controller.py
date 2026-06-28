import os
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from visual_servo.controllers.nladrc_controller import NLADRCController3D, NLADRC_1st_Order, fal
from visual_servo.servo.target_estimator import SimpleTargetPredictor2D
from visual_servo.servo.visual_servo_params import ServoRuntimeConfig


_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
_SPLIT_PARAM_FILES = [
    "visual_servo_params.yaml",
    "visual_servo_ladrc_params.yaml",
    "visual_servo_nladrc_params.yaml",
    "visual_servo_mpc_params.yaml",
    "visual_servo_pid_params.yaml",
    "visual_servo_adaptive_pid_params.yaml",
]


def _load_ros_params_file(name: str) -> dict:
    with open(os.path.join(_CONFIG_DIR, name), "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data["/**"]["ros__parameters"]


def _merged_visual_servo_params() -> dict:
    params = {}
    for name in _SPLIT_PARAM_FILES:
        params.update(_load_ros_params_file(name))
    return params


class _FakeNode:
    def __init__(self, params=None):
        self.params = dict(params or {})

    def has_parameter(self, name):
        return name in self.params

    def declare_parameter(self, name, default):
        self.params[name] = default

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])


def _controller(**overrides):
    """Build a 1-D NLADRC controller with sensible test defaults."""
    params = dict(
        wc=10.0,
        wo=25.0,
        b0=0.5,
        dt=0.004,
        alpha_obs=0.90,
        alpha_obs2=0.70,
        delta_obs=0.004,
        obs_error_clip=0.02,
        obs_transition=0.004,
        z2_clip=0.12,
        u_fb_clip=0.22,
        z2_decay_band=0.004,
        z2_decay_gain=6.0,
        z2_gain=0.35,
        u_rate_max=0.75,
        u_ema_alpha=1.0,
        u_clip=0.28,
        internal_shape=False,
    )
    params.update(overrides)
    return NLADRC_1st_Order(**params)


class NLADRCControllerTest(unittest.TestCase):
    # -- fal function ------------------------------------------------------------

    def test_fal_is_linear_in_small_error_region(self):
        delta = 0.01
        alpha = 0.5
        e = 0.002
        self.assertAlmostEqual(fal(e, alpha, delta), e / (delta ** (1.0 - alpha)))

    def test_fal_is_nonlinear_and_symmetric_outside_delta(self):
        delta = 0.01
        alpha = 0.5
        e = 0.04
        self.assertAlmostEqual(fal(e, alpha, delta), e ** alpha)
        self.assertAlmostEqual(fal(-e, alpha, delta), -(e ** alpha))

    # -- basic controller behaviour ---------------------------------------------

    def test_nladrc_step_returns_finite_clipped_output(self):
        ctrl = _controller()
        outputs = [ctrl.step(0.0105) for _ in range(20)]
        self.assertTrue(all(np.isfinite(outputs)))
        self.assertLessEqual(max(abs(u) for u in outputs), 0.29)

    def test_nladrc_reset_clears_state(self):
        ctrl = _controller()
        ctrl.step(0.02)
        ctrl.commit_applied_command(0.1)
        ctrl.reset()
        self.assertEqual(ctrl.z1, 0.0)
        self.assertEqual(ctrl.z2, 0.0)
        self.assertEqual(ctrl.u_last, 0.0)
        self.assertEqual(ctrl.u_applied_last, 0.0)

    def test_nladrc_uses_applied_command_in_observer(self):
        ctrl = _controller()
        ctrl.commit_applied_command(0.2)
        ctrl.step(0.01)
        self.assertNotEqual(ctrl.z1, 0.0)
        self.assertAlmostEqual(ctrl.last_debug.u_applied_last, 0.2)

    # -- ESO mixing -------------------------------------------------------------

    def test_linear_mix_is_high_for_small_error_and_lower_for_large_error(self):
        ctrl = _controller()
        ctrl.z1 = 0.001
        ctrl.step(0.001)  # e_obs ≈ 0 → linear_mix ≈ 1.0
        small_mix = ctrl.last_debug.linear_mix
        self.assertGreater(small_mix, 0.9)
        ctrl.reset()
        ctrl.z1 = 0.03
        ctrl.step(0.001)  # e_obs clipped to obs_error_clip → linear_mix ≈ 0.0
        large_mix = ctrl.last_debug.linear_mix
        self.assertLess(large_mix, 0.1)

    def test_small_error_observer_stays_close_to_linear(self):
        ctrl = _controller()
        ctrl.z1 = 0.0012
        ctrl.step(0.0010)
        e_obs = ctrl.last_debug.e_obs
        self.assertAlmostEqual(ctrl.last_debug.fal_obs, e_obs, delta=abs(e_obs) * 0.25 + 1e-9)

    def test_repeated_small_error_keeps_z2_bounded(self):
        ctrl = _controller()
        for _ in range(200):
            ctrl.step(0.0015)
            ctrl.commit_applied_command(ctrl.last_debug.u)
        self.assertLessEqual(abs(ctrl.z2), 0.12 + 1e-9)

    def test_large_observer_error_saturates_eobs(self):
        ctrl = _controller()
        ctrl.z1 = 0.05
        ctrl.z2 = 0.12
        ctrl.step(0.0)
        self.assertAlmostEqual(ctrl.last_debug.e_obs, 0.02)

    # -- LADRC control law ------------------------------------------------------

    def test_default_feedback_matches_ladrc_main_law(self):
        ctrl = _controller()
        ctrl.z1 = 0.007
        ctrl.z2 = 0.010
        ctrl.step(0.007)
        expected_u0 = 10.0 * ctrl.z1
        dist_weight = max(0.0, 1.0 - abs(ctrl.last_debug.e_obs) / 0.02)
        expected_u_fb = (expected_u0 + ctrl.z2_gain * dist_weight * ctrl.z2) / 0.5
        self.assertAlmostEqual(ctrl.last_debug.u0, expected_u0)
        self.assertAlmostEqual(ctrl.last_debug.u_fb, expected_u_fb)

    def test_feedback_is_not_silently_zeroed_at_moderate_error(self):
        """NLADRC must keep producing output at 4 mm so residual is not locked."""
        ctrl = _controller()
        ctrl.z1 = 0.004  # 4 mm
        ctrl.z2 = 0.002
        u = ctrl.step(0.004)
        self.assertGreater(abs(u), 1e-6)
        self.assertGreater(abs(ctrl.last_debug.u_fb), 1e-6)

    # -- z2 decay ---------------------------------------------------------------

    def test_z2_gain_scales_disturbance_injection(self):
        """z2_gain < 1.0 should reduce the z2 contribution to u_fb."""
        ctrl_full = _controller(z2_gain=1.0)
        ctrl_half = _controller(z2_gain=0.35)
        ctrl_full.z1 = 0.005; ctrl_full.z2 = 0.05
        ctrl_half.z1 = 0.005; ctrl_half.z2 = 0.05
        u_full = ctrl_full.step(0.005)
        u_half = ctrl_half.step(0.005)
        # With same (z1,z2), u_fb should have different z2 contribution
        self.assertNotAlmostEqual(u_full, u_half, delta=1e-9)
        self.assertLess(abs(u_half), abs(u_full) + 1e-9)

    def test_z2_decay_releases_residual_disturbance(self):
        ctrl = _controller()
        ctrl.z1 = 0.002
        ctrl.z2 = 0.08
        ctrl.commit_applied_command(0.0)
        before = ctrl.z2
        ctrl.step(0.002)
        self.assertLess(ctrl.z2, before)

    def test_large_obs_error_does_not_silence_z2(self):
        """z2 should remain active (not forced to zero) when obs error is large."""
        ctrl = _controller()
        ctrl.z1 = 0.03  # far from true error → e_obs clipped to 0.02
        ctrl.z2 = 0.08
        ctrl.step(0.01)  # e_obs > z2_decay_band → decay NOT active
        # z2 updated by ESO (not decayed), should be non-zero
        self.assertNotAlmostEqual(ctrl.z2, 0.0, delta=1e-6)

    # -- 3D controller ----------------------------------------------------------

    def test_3d_xy_bypasses_internal_shape_but_z_keeps_it(self):
        ctrl = NLADRCController3D(
            dt=0.004,
            u_rate_max_xy=0.20,
            u_ema_alpha=1.0,
            u_clip_xy=1.0,
            u_fb_clip_xy=1.0,
        )
        vx, vy, vz, debug = ctrl.step(np.array([0.08, -0.08, 0.08]), 0.004)
        self.assertAlmostEqual(debug["u_cmd_pre_x"], debug["u_cmd_shaped_x"])
        self.assertAlmostEqual(debug["u_cmd_pre_y"], debug["u_cmd_shaped_y"])
        self.assertAlmostEqual(vx, debug["u_cmd_pre_x"])
        self.assertAlmostEqual(vy, debug["u_cmd_pre_y"])
        max_z_step = 0.20 * 0.004
        self.assertLessEqual(abs(vz), max_z_step + 1e-9)
        self.assertGreater(abs(debug["u_cmd_pre_z"]), abs(debug["u_z"]))

    def test_3d_debug_field_count_is_stable(self):
        """PlotJuggler layout depends on field count and order."""
        ctrl = NLADRCController3D(dt=0.004)
        _, _, _, debug = ctrl.step(np.array([0.01, -0.01, 0.005]), 0.004)
        expected_fields_per_axis = 13
        # 3 axes x 13 fields
        self.assertEqual(len(debug), 3 * expected_fields_per_axis)
        for axis in ("x", "y", "z"):
            for suffix in (
                "z1", "z2", "u0", "u_fb", "u_ff",
                "u_cmd_pre", "u_cmd_shaped", "u_applied_last",
                "u", "fal_obs", "fal_ctrl", "linear_mix", "e_obs",
            ):
                self.assertIn(f"{suffix}_{axis}", debug)

    # -- RuntimeConfig compatibility --------------------------------------------

    def test_runtime_config_accepts_nladrc(self):
        cfg = ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NLADRC"}))
        self.assertEqual(cfg.servo_controller_type, "NLADRC")
        self.assertEqual(cfg.servo_controller_family, "NLADRC")
        self.assertEqual(cfg.pid_variant, "NONE")
        self.assertEqual(cfg.nladrc_obs_transition_xy, 0.004)
        self.assertEqual(cfg.nladrc_z2_clip_xy, 0.12)
        self.assertEqual(cfg.nladrc_u_fb_clip_xy, 0.22)
        self.assertEqual(cfg.nladrc_z2_decay_band_xy, 0.004)
        self.assertEqual(cfg.nladrc_z2_decay_gain_xy, 6.0)
        self.assertEqual(cfg.nladrc_z2_gain_xy, 0.35)
        self.assertEqual(cfg.nladrc_ff_mix_gain, 0.20)
        self.assertEqual(cfg.nladrc_u_rate_max_xy, 0.75)
        self.assertEqual(cfg.nladrc_u_ema_alpha, 1.0)
        self.assertEqual(cfg.nladrc_u_clip_xy, 0.28)

    def test_runtime_config_accepts_split_yaml_for_all_controller_families(self):
        params = _merged_visual_servo_params()
        cases = {
            "NLADRC": ("NLADRC", "NONE"),
            "LADRC": ("LADRC", "NONE"),
            "MPC": ("MPC", "NONE"),
            "PID": ("PID", "PID"),
            "ADAPTIVE_PID": ("PID", "ADAPTIVE_PID"),
        }
        for controller_type, (family, pid_variant) in cases.items():
            with self.subTest(controller_type=controller_type):
                cfg = ServoRuntimeConfig.from_node(_FakeNode({**params, "servo_controller_type": controller_type}))
                self.assertEqual(cfg.servo_controller_family, family)
                self.assertEqual(cfg.pid_variant, pid_variant)

    def test_runtime_config_rejects_unknown_controller(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NOT_A_CONTROLLER"}))

    def test_runtime_config_rejects_invalid_nladrc_wc(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_wc_xy": 0.0,
            }))

    def test_runtime_config_rejects_invalid_z2_clip(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_z2_clip_xy": 0.0,
            }))


class ServoControllerStabilityTest(unittest.TestCase):
    def _servo_controller_class(self):
        try:
            from visual_servo.servo.servo_controller import ServoController
        except ModuleNotFoundError as exc:
            if exc.name == "rclpy":
                self.skipTest("rclpy is not available on PYTHONPATH")
            raise
        return ServoController

    def _controller_with_predictor_state(self):
        ServoController = self._servo_controller_class()
        ctrl = object.__new__(ServoController)
        ctrl.target_predictor = SimpleTargetPredictor2D()
        ctrl.target_predictor.update([1.0, 2.0], [0.3, -0.2], 10.0)
        ctrl._obs_last_meas_xy = np.array([1.0, 2.0], dtype=float)
        ctrl._obs_last_meas_stamp_sec = 10.0
        ctrl._ff_vel_filt = np.array([0.3, -0.2], dtype=float)
        ctrl._target_xy_pred = np.array([1.1, 1.9], dtype=float)
        ctrl._target_vxy_pred = np.array([0.3, -0.2], dtype=float)
        ctrl._target_axy_pred = np.array([0.1, -0.1], dtype=float)
        ctrl._predict_horizon = 0.05
        return ctrl

    def test_target_stale_resets_predictor_state(self):
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()))
        ctrl = self._controller_with_predictor_state()
        ctrl.node = SimpleNamespace(_get_latest_target_msgs=lambda: (msg, {"yaw": 0.0}, {"yaw_offset": 0.0}))
        ctrl.servo_detection_timeout = 0.14
        ctrl._last_msg_age = -1.0
        ctrl._msg_age_sec = lambda _stamp: 0.50

        obj_msg, obj_rpy, prof = ctrl._get_fresh_obj()

        self.assertIsNone(obj_msg)
        self.assertIsNone(obj_rpy)
        self.assertIsNone(prof)
        self.assertFalse(ctrl.target_predictor.initialized)
        self.assertIsNone(ctrl._obs_last_meas_xy)
        self.assertIsNone(ctrl._obs_last_meas_stamp_sec)
        self.assertTrue(np.allclose(ctrl._ff_vel_filt, 0.0))
        self.assertTrue(np.allclose(ctrl._target_xy_pred, 0.0))
        self.assertTrue(np.allclose(ctrl._target_vxy_pred, 0.0))
        self.assertTrue(np.allclose(ctrl._target_axy_pred, 0.0))
        self.assertEqual(ctrl._predict_horizon, 0.0)

    def test_status_decel_scales_final_command(self):
        ServoController = self._servo_controller_class()
        ctrl = object.__new__(ServoController)
        ctrl.controller_family = "NLADRC"
        ctrl.v_xy_max = 1.0
        ctrl._v_last = np.zeros(4, dtype=float)
        ctrl._status_decel_active = True
        ctrl.status1_speed_scale = 0.4

        vx, vy, vz, _, _, _, u_slew = ctrl._postprocess_command(0.50, -0.25, 0.0, 0.004)

        self.assertAlmostEqual(vx, 0.20)
        self.assertAlmostEqual(vy, -0.10)
        self.assertAlmostEqual(vz, 0.0)
        self.assertTrue(np.allclose(u_slew, [0.20, -0.10, 0.0]))


if __name__ == "__main__":
    unittest.main()
