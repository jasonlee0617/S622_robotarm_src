import os
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from visual_servo.controllers.nladrc_controller import NLADRCController3D, NLADRC_1st_Order, fal
from visual_servo.servo.target_estimator import SimpleTargetPredictor2D
from visual_servo.servo.visual_servo_params import ServoRuntimeConfig


_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
_ALGO_PARAM_KEYS = {
    "v_xy_max",
    "a_xy_max",
    "v_z_max",
    "a_z_max",
    "twist_norm_max",
    "status1_speed_scale",
    "predict_lead_sec",
    "max_predict_horizon",
    "servo_detection_timeout",
    "cmd_lpf_alpha",
    "vel_ff_gain",
    "rel_vel_damping_gain",
    "ff_vel_ema_alpha",
    "max_target_speed",
    "target_vxy_clip",
    "meas_jump_clip_xy",
    "ee_vel_ema_alpha",
    "rel_vel_clip",
    "ff_term_clip",
    "target_accel_ema_alpha",
    "ff_age_start_sec",
    "ff_age_ref_sec",
    "ff_age_window_sec",
    "ff_age_floor_scale",
    "ff_err_norm_threshold",
    "ff_large_err_scale",
    "slew_dv_trigger",
    "slew_alpha_high",
    "slew_alpha_low",
    "handoff_target_delta_max",
    "servo_handoff_zero_twist_count",
}


def _load_ros_params_file(name: str) -> dict:
    with open(os.path.join(_CONFIG_DIR, name), "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data["/**"]["ros__parameters"]


def _params_for_controller(controller_type: str) -> dict:
    controller_type = controller_type.strip().upper()
    file_map = {
        "PID": ["visual_servo_params.yaml", "visual_servo_pid_params.yaml"],
        "PD": ["visual_servo_params.yaml", "visual_servo_pid_params.yaml"],
        "PI_FF": ["visual_servo_params.yaml", "visual_servo_pid_params.yaml"],
        "ADAPTIVE_PID": [
            "visual_servo_params.yaml",
            "visual_servo_pid_params.yaml",
            "visual_servo_adaptive_pid_params.yaml",
        ],
        "LADRC": ["visual_servo_params.yaml", "visual_servo_ladrc_params.yaml"],
        "NLADRC": ["visual_servo_params.yaml", "visual_servo_nladrc_params.yaml"],
        "MPC": ["visual_servo_params.yaml", "visual_servo_mpc_params.yaml"],
    }
    params = {}
    for name in file_map[controller_type]:
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

    def dbg_throttle(self, *args, **kwargs):
        return False

    def get_logger(self):
        return SimpleNamespace(warn=lambda *args, **kwargs: None)


class _FakeTfTools:
    def __init__(self, pos_base):
        self.pos_base = pos_base

    def camera_point_to_base(self, _msg):
        return self.pos_base


def _controller(**overrides):
    """Build a 1-D NLADRC controller with sensible test defaults."""
    params = dict(
        wc=12.5,
        wo=25.0,
        b0=0.5,
        dt=0.004,
        alpha_obs=0.98,
        alpha_obs2=0.95,
        delta_obs=0.004,
        obs_error_clip=0.02,
        obs_transition=0.012,
        z2_clip=0.14,
        u_fb_clip=0.24,
        z2_decay_band=0.004,
        z2_decay_gain=3.0,
        z2_gain=1.0,
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

    def test_base_origin_visual_target_is_rejected(self):
        try:
            from visual_servo.servo.servo_controller import ServoController
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        fake = SimpleNamespace(node=_FakeNode())
        fake.node.tf_tools = _FakeTfTools(SimpleNamespace(x=0.0, y=0.0, z=0.0))
        pos, pos_base = ServoController._target_msg_to_base_position(fake, object())
        self.assertIsNone(pos)
        self.assertIsNone(pos_base)

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
        self.assertLessEqual(abs(ctrl.z2), 0.14 + 1e-9)

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
        expected_u0 = 12.5 * ctrl.z1
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
        self.assertEqual(cfg.nladrc_obs_transition_xy, 0.012)
        self.assertEqual(cfg.nladrc_z2_clip_xy, 0.14)
        self.assertEqual(cfg.nladrc_u_fb_clip_xy, 0.24)
        self.assertEqual(cfg.nladrc_z2_decay_band_xy, 0.004)
        self.assertEqual(cfg.nladrc_z2_decay_gain_xy, 3.0)
        self.assertEqual(cfg.nladrc_z2_gain_xy, 1.0)
        self.assertEqual(cfg.nladrc_ff_mix_gain, 0.30)
        self.assertEqual(cfg.nladrc_u_rate_max_xy, 0.75)
        self.assertEqual(cfg.nladrc_u_ema_alpha, 1.0)
        self.assertEqual(cfg.nladrc_u_clip_xy, 0.28)

    def test_runtime_config_accepts_split_yaml_for_all_controller_families(self):
        cases = {
            "NLADRC": ("NLADRC", "NONE"),
            "LADRC": ("LADRC", "NONE"),
            "MPC": ("MPC", "NONE"),
            "PID": ("PID", "PID"),
            "ADAPTIVE_PID": ("PID", "ADAPTIVE_PID"),
        }
        for controller_type, (family, pid_variant) in cases.items():
            with self.subTest(controller_type=controller_type):
                cfg = ServoRuntimeConfig.from_node(
                    _FakeNode({**_params_for_controller(controller_type), "servo_controller_type": controller_type})
                )
                self.assertEqual(cfg.servo_controller_family, family)
                self.assertEqual(cfg.pid_variant, pid_variant)

    def test_common_yaml_keeps_algorithm_tuning_outside(self):
        common_params = _load_ros_params_file("visual_servo_params.yaml")
        self.assertTrue(_ALGO_PARAM_KEYS.isdisjoint(common_params))

    def test_each_algorithm_yaml_carries_shared_tracking_keys(self):
        cases = {
            "visual_servo_pid_params.yaml": _ALGO_PARAM_KEYS,
            "visual_servo_adaptive_pid_params.yaml": _ALGO_PARAM_KEYS,
            "visual_servo_ladrc_params.yaml": _ALGO_PARAM_KEYS,
            "visual_servo_nladrc_params.yaml": _ALGO_PARAM_KEYS,
            "visual_servo_mpc_params.yaml": _ALGO_PARAM_KEYS,
        }
        for name, expected_keys in cases.items():
            with self.subTest(name=name):
                params = _load_ros_params_file(name)
                self.assertTrue(expected_keys.issubset(params))

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

        obj_msg, obj_rpy, prof = ctrl._get_fresh_grasp_target()

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
        ctrl.slew_dv_trigger = 0.03
        ctrl.slew_alpha_high = 1.0
        ctrl.slew_alpha_low = 0.70

        vx, vy, vz, _, _, _, u_slew = ctrl._shape_servo_command(0.50, -0.25, 0.0, 0.004)

        self.assertAlmostEqual(vx, 0.20)
        self.assertAlmostEqual(vy, -0.10)
        self.assertAlmostEqual(vz, 0.0)
        self.assertTrue(np.allclose(u_slew, [0.20, -0.10, 0.0]))

    def test_nladrc_smooths_final_command_with_existing_slew_alpha(self):
        ServoController = self._servo_controller_class()
        ctrl = object.__new__(ServoController)
        ctrl.controller_family = "NLADRC"
        ctrl.v_xy_max = 1.0
        ctrl._v_last = np.array([0.10, -0.10, 0.0, 0.0], dtype=float)
        ctrl._status_decel_active = False
        ctrl.slew_dv_trigger = 0.03
        ctrl.slew_alpha_high = 0.90
        ctrl.slew_alpha_low = 0.70

        vx, vy, vz, _, _, _, u_slew = ctrl._shape_servo_command(0.30, -0.20, 0.0, 0.004)

        self.assertAlmostEqual(vx, 0.90 * 0.30 + 0.10 * 0.10)
        self.assertAlmostEqual(vy, 0.90 * -0.20 + 0.10 * -0.10)
        self.assertAlmostEqual(vz, 0.0)
        self.assertTrue(np.allclose(u_slew, [vx, vy, 0.0]))

    def test_nladrc_mixes_feedforward_without_extra_damping(self):
        ServoController = self._servo_controller_class()
        ctrl = object.__new__(ServoController)
        ctrl.controller_family = "NLADRC"
        ctrl.nladrc_ff_mix_gain = 0.35
        ctrl.nladrc_controller = SimpleNamespace(
            step=lambda _error, _dt: (0.10, -0.20, 0.0, {"z1_x": 0.0})
        )
        ctrl.node = SimpleNamespace(
            messages_publishers=SimpleNamespace(publish_servo_nladrc_debug=lambda _debug: None)
        )

        vx, vy, vz = ctrl._run_selected_controller(
            0.0,
            0.0,
            0.0,
            0.004,
            np.array([0.02, -0.03]),
            np.array([0.04, 0.01]),
        )

        self.assertAlmostEqual(vx, 0.10 + 0.35 * 0.02)
        self.assertAlmostEqual(vy, -0.20 + 0.35 * -0.03)
        self.assertAlmostEqual(vz, 0.0)

    def test_invalid_predicted_visual_target_stops_cycle_and_resets_nladrc(self):
        ServoController = self._servo_controller_class()
        ctrl = object.__new__(ServoController)
        calls = {"zero": 0, "commit": [], "reset_pred": 0, "reset_nladrc": 0}

        def zero_twist():
            calls["zero"] += 1

        def reset_prediction():
            calls["reset_pred"] += 1

        def reset_nladrc():
            calls["reset_nladrc"] += 1

        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()))
        ctrl.node = SimpleNamespace(
            abort=SimpleNamespace(is_set=lambda: False),
            _get_state=lambda: "SERVO_TRACK_ABOVE",
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)),
        )
        ctrl.io = SimpleNamespace(publish_zero_twist=zero_twist)
        ctrl.controller_family = "NLADRC"
        ctrl.nladrc_controller = SimpleNamespace(reset=reset_nladrc)
        ctrl._apply_servo_status_policy = lambda: True
        ctrl._resolve_active_target = lambda _state, _cur_q: (msg, 0.0)
        ctrl._target_msg_to_base_position = lambda _msg: (np.array([0.30, 0.47, 0.0]), SimpleNamespace())
        ctrl._stamp_to_sec = lambda _stamp: 0.0
        ctrl._predict_visual_target_state = lambda _pos, _msg: (np.zeros(2), np.zeros(2), 0.0)
        ctrl._reset_target_prediction_state = reset_prediction
        ctrl._commit_nladrc_applied_command = lambda vx, vy, vz: calls["commit"].append((vx, vy, vz))
        ctrl._compute_visual_tracking_error = lambda *_args: self.fail("invalid predicted target reached control law")

        ctrl._run_visual_servo_cycle(cur_p=np.array([0.30, 0.47, 0.0]), cur_q=np.array([0.0, 0.0, 0.0, 1.0]), dt=0.004)

        self.assertEqual(calls["zero"], 1)
        self.assertEqual(calls["reset_pred"], 1)
        self.assertEqual(calls["reset_nladrc"], 1)
        self.assertEqual(calls["commit"], [(0.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
