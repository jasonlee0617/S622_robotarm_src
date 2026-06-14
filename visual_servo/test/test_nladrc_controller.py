from types import SimpleNamespace
import unittest

import numpy as np

from visual_servo.controllers.nladrc_controller import NLADRC_1st_Order, fal
from visual_servo.servo.servo_runtime_config import ServoRuntimeConfig


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
    params = dict(
        wc=11.0,
        wo=24.0,
        b0=0.5,
        dt=0.004,
        alpha_obs=0.85,
        alpha_obs2=0.70,
        alpha_ctrl=0.90,
        delta_obs=0.004,
        delta_ctrl=0.0045,
        err_transition=0.0090,
        obs_error_clip=0.02,
        obs_transition=0.004,
        z2_clip=0.12,
        u_fb_clip=0.24,
        tail_error_band=0.0045,
        tail_u_fb_clip=0.13,
        tail_u_rate_max=0.35,
        tail_ff_scale=0.00,
        ff_enable_err_band=0.006,
        ff_disable_err_band=0.003,
        ff_age_disable_sec=0.16,
        ff_z2_conflict_band=0.10,
        wc_tail=15.0,
        delta_ctrl_tail=0.0025,
        err_transition_tail=0.0035,
        z2_decay_band=0.004,
        z2_decay_gain=3.5,
        ff_mix_gain=0.95,
        wc_boost=12.0,
        wo_boost=24.0,
        delta_ctrl_boost=0.0035,
        err_transition_boost=0.0060,
        u_fb_clip_boost=0.26,
        u_clip_boost=0.29,
        ff_mix_gain_boost=1.00,
        ff_boost_ref=0.024,
        ff_motion_ref=0.010,
        ff_motion_floor=0.60,
        ff_motion_exit=0.006,
        ff_boost_exit=0.018,
        ff_lead_time=0.024,
        ff_lead_clip=0.0012,
        mode_blend_alpha=0.30,
        err_rate_ema_alpha=0.25,
        u_damp_gain=0.010,
        u_damp_gain_boost=0.018,
        u_damp_clip=0.045,
        u_rate_max=0.75,
        u_ema_alpha=1.0,
        u_clip=0.28,
        internal_shape=False,
    )
    params.update(overrides)
    return NLADRC_1st_Order(**params)


class NLADRCControllerTest(unittest.TestCase):
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

    def test_nladrc_step_returns_finite_clipped_output(self):
        ctrl = _controller()
        outputs = [ctrl.step(0.01, u_ff=0.005) for _ in range(20)]
        self.assertTrue(all(np.isfinite(outputs)))
        self.assertLessEqual(max(abs(u) for u in outputs), 0.29)
        self.assertTrue(all(abs(ctrl.last_debug.u_cmd_pre - ctrl.last_debug.u_cmd_shaped) <= 1e-9 for _ in [0]))

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
        ctrl.step(0.01, u_ff=0.0)
        self.assertNotEqual(ctrl.z1, 0.0)
        self.assertAlmostEqual(ctrl.last_debug.u_applied_last, 0.2)

    def test_linear_mix_is_high_for_small_error_and_lower_for_large_error(self):
        ctrl = _controller()
        ctrl.z1 = 0.001
        ctrl.step(0.001, u_ff=0.0)
        small_mix = ctrl.last_debug.linear_mix
        ctrl.reset()
        ctrl.z1 = 0.05
        ctrl.step(0.05, u_ff=0.0)
        large_mix = ctrl.last_debug.linear_mix
        self.assertGreater(small_mix, large_mix)

    def test_small_error_observer_stays_close_to_linear(self):
        ctrl = _controller()
        ctrl.z1 = 0.0012
        ctrl.step(0.0010, u_ff=0.0)
        e_obs = ctrl.last_debug.e_obs
        self.assertAlmostEqual(ctrl.last_debug.fal_obs, e_obs, delta=abs(e_obs) * 0.25 + 1e-9)

    def test_repeated_small_error_keeps_z2_bounded(self):
        ctrl = _controller()
        for _ in range(200):
            ctrl.step(0.0015, u_ff=0.0)
            ctrl.commit_applied_command(ctrl.last_debug.u)
        self.assertLessEqual(abs(ctrl.z2), 0.12 + 1e-9)

    def test_large_observer_error_reduces_disturbance_influence(self):
        ctrl = _controller()
        ctrl.z1 = 0.05
        ctrl.z2 = 0.12
        ctrl.step(0.0, u_ff=0.0)
        self.assertLessEqual(abs(ctrl.last_debug.u_fb), 0.26 + 1e-9)
        self.assertAlmostEqual(ctrl.last_debug.e_obs, 0.02)

    def test_command_pre_and_shaped_stay_close_on_step(self):
        ctrl = _controller()
        diffs = []
        for _ in range(200):
            u = ctrl.step(0.02, u_ff=0.015)
            ctrl.commit_applied_command(u)
            diffs.append(abs(ctrl.last_debug.u_cmd_pre - ctrl.last_debug.u_cmd_shaped))
        self.assertLess(max(diffs[-20:]), 1e-9)

    def test_feedforward_is_retained_in_dynamic_tracking_region(self):
        ctrl = _controller()
        ctrl.step(0.02, u_ff=0.03, ff_age=0.0, error_norm=0.02, ff_norm=0.015)
        self.assertGreater(ctrl.last_debug.u_ff, 0.025)

    def test_feedforward_is_suppressed_in_tail_region_when_target_is_almost_stopped(self):
        ctrl = _controller()
        ctrl.step(0.0025, u_ff=0.003, ff_age=0.20, error_norm=0.0025, ff_norm=0.003)
        self.assertLess(abs(ctrl.last_debug.u_ff), 1e-4)

    def test_dynamic_feedforward_is_not_suppressed_by_age_or_z2_conflict(self):
        ctrl = _controller()
        ctrl.z2 = 0.08
        ctrl.step(0.02, u_ff=0.03, ff_age=1.0, error_norm=0.02, ff_norm=0.03)
        self.assertGreater(ctrl.last_debug.u_ff, 0.025)

    def test_boost_mode_raises_feedforward_and_clip_authority(self):
        ctrl = _controller()
        ctrl.step(0.02, u_ff=0.03, ff_age=0.0, error_norm=0.02, ff_norm=0.01)
        track_ff = ctrl.last_debug.u_ff
        ctrl.reset()
        ctrl.step(0.02, u_ff=0.03, ff_age=0.0, error_norm=0.02, ff_norm=0.03)
        boost_ff = ctrl.last_debug.u_ff
        self.assertGreater(boost_ff, track_ff)
        self.assertGreaterEqual(boost_ff, 0.028)

    def test_motion_hold_keeps_feedforward_alive_with_small_error(self):
        ctrl = _controller()
        ctrl.step(0.0025, u_ff=0.03, ff_age=0.0, error_norm=0.0025, ff_norm=0.03)
        self.assertGreater(ctrl.last_debug.u_ff, 0.010)

    def test_motion_hold_hysteresis_keeps_feedforward_until_exit(self):
        ctrl = _controller()
        ctrl.step(0.0025, u_ff=0.03, ff_age=0.0, error_norm=0.0025, ff_norm=0.012)
        keep_ff = ctrl.last_debug.u_ff
        ctrl.commit_applied_command(ctrl.last_debug.u)
        ctrl.step(0.0025, u_ff=0.03, ff_age=0.0, error_norm=0.0025, ff_norm=0.008)
        hold_ff = ctrl.last_debug.u_ff
        self.assertGreaterEqual(hold_ff, 0.010)
        ctrl.commit_applied_command(ctrl.last_debug.u)
        ctrl.step(0.0025, u_ff=0.004, ff_age=0.0, error_norm=0.0025, ff_norm=0.004)
        self.assertLess(ctrl.last_debug.u_ff, hold_ff)
        self.assertGreater(keep_ff, 0.0)

    def test_boost_hysteresis_stays_active_until_exit(self):
        ctrl = _controller()
        ctrl.step(0.02, u_ff=0.03, ff_age=0.0, error_norm=0.02, ff_norm=0.03)
        self.assertTrue(ctrl.boost_hold_active)
        prev_blend = ctrl.boost_blend
        ctrl.commit_applied_command(ctrl.last_debug.u)
        ctrl.step(0.02, u_ff=0.03, ff_age=0.0, error_norm=0.02, ff_norm=0.020)
        self.assertTrue(ctrl.boost_hold_active)
        self.assertGreater(ctrl.boost_blend, 0.0)
        ctrl.commit_applied_command(ctrl.last_debug.u)
        ctrl.step(0.02, u_ff=0.015, ff_age=0.0, error_norm=0.02, ff_norm=0.015)
        self.assertFalse(ctrl.boost_hold_active)
        self.assertLess(ctrl.boost_blend, prev_blend)

    def test_feedforward_lead_term_is_bounded_and_disabled_in_settle(self):
        ctrl = _controller()
        ctrl.step(0.02, u_ff=0.20, ff_age=0.0, error_norm=0.02, ff_norm=0.03)
        self.assertLessEqual(abs(ctrl.last_e_lead), 0.0012 + 1e-9)
        ctrl.reset()
        ctrl.step(0.002, u_ff=0.003, ff_age=0.0, error_norm=0.002, ff_norm=0.003)
        self.assertAlmostEqual(ctrl.last_e_lead, 0.0, delta=1e-6)

    def test_ctrl_error_rate_filter_stays_bounded(self):
        ctrl = _controller()
        for _ in range(100):
            ctrl.step(0.002 + 0.0005, u_ff=0.0)
            ctrl.commit_applied_command(ctrl.last_debug.u)
        self.assertTrue(np.isfinite(ctrl.ctrl_error_rate_filt))
        self.assertLess(abs(ctrl.ctrl_error_rate_filt), 5.0)

    def test_u_damp_is_active_in_track_and_releases_in_settle(self):
        ctrl = _controller()
        ctrl.step(0.015, u_ff=0.03, ff_age=0.0, error_norm=0.015, ff_norm=0.03)
        ctrl.commit_applied_command(ctrl.last_debug.u)
        ctrl.step(0.025, u_ff=0.03, ff_age=0.0, error_norm=0.025, ff_norm=0.03)
        self.assertGreater(abs(ctrl.last_u_damp), 1e-5)
        ctrl.commit_applied_command(ctrl.last_debug.u)
        for _ in range(20):
            ctrl.step(0.002, u_ff=0.0, ff_age=0.0, error_norm=0.002, ff_norm=0.0)
            ctrl.commit_applied_command(ctrl.last_debug.u)
        self.assertLess(abs(ctrl.last_u_damp), 1e-4)

    def test_damping_reduces_output_delta_for_dynamic_sequence(self):
        ctrl_damped = _controller(u_clip=1.0, u_clip_boost=1.0, u_fb_clip=1.0, u_fb_clip_boost=1.0, tail_u_fb_clip=1.0)
        ctrl_plain = _controller(
            u_clip=1.0,
            u_clip_boost=1.0,
            u_fb_clip=1.0,
            u_fb_clip_boost=1.0,
            tail_u_fb_clip=1.0,
            u_damp_gain=0.0,
            u_damp_gain_boost=0.0,
            u_damp_clip=0.0,
        )
        seq = [0.004, 0.012, 0.003, 0.013, 0.004, 0.011, 0.0035, 0.0125]
        diffs_damped = []
        diffs_plain = []
        last_damped = None
        last_plain = None
        for e in seq:
            u_damped = ctrl_damped.step(e, u_ff=0.03, ff_age=0.0, error_norm=abs(e), ff_norm=0.03)
            u_plain = ctrl_plain.step(e, u_ff=0.03, ff_age=0.0, error_norm=abs(e), ff_norm=0.03)
            ctrl_damped.commit_applied_command(u_damped)
            ctrl_plain.commit_applied_command(u_plain)
            if last_damped is not None:
                diffs_damped.append(abs(u_damped - last_damped))
                diffs_plain.append(abs(u_plain - last_plain))
            last_damped = u_damped
            last_plain = u_plain
        self.assertLess(sum(diffs_damped) / len(diffs_damped), sum(diffs_plain) / len(diffs_plain))

    def test_tail_z2_decay_releases_residual_disturbance(self):
        ctrl = _controller()
        ctrl.z1 = 0.002
        ctrl.z2 = 0.08
        ctrl.commit_applied_command(0.0)
        before = ctrl.z2
        ctrl.step(0.002, u_ff=0.0, ff_age=0.0)
        self.assertLess(ctrl.z2, before)

    def test_track_to_settle_transition_stays_bounded(self):
        ctrl = _controller()
        first = ctrl.step(0.015, u_ff=0.02, ff_age=0.0, error_norm=0.015, ff_norm=0.02)
        ctrl.commit_applied_command(first)
        second = ctrl.step(0.0050, u_ff=0.02, ff_age=0.0, error_norm=0.0050, ff_norm=0.02)
        self.assertLess(abs(second - first), 0.07)

    def test_settle_mode_uses_tail_feedback_parameters(self):
        ctrl = _controller()
        ctrl.z1 = 0.0048
        ctrl.step(0.0048, u_ff=0.0, ff_age=0.0)
        settle_u0 = ctrl.last_debug.u0 / 0.0048
        ctrl.reset()
        ctrl.z1 = 0.0062
        ctrl.step(0.0062, u_ff=0.0, ff_age=0.0)
        track_u0 = ctrl.last_debug.u0 / 0.0062
        self.assertGreater(settle_u0, track_u0)

    def test_runtime_config_accepts_nladrc(self):
        cfg = ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NLADRC"}))
        self.assertEqual(cfg.servo_controller_type, "NLADRC")
        self.assertEqual(cfg.servo_controller_family, "NLADRC")
        self.assertEqual(cfg.pid_variant, "NONE")
        self.assertEqual(cfg.nladrc_obs_transition_xy, 0.004)
        self.assertEqual(cfg.nladrc_z2_clip_xy, 0.12)
        self.assertEqual(cfg.nladrc_u_fb_clip_xy, 0.24)
        self.assertEqual(cfg.nladrc_tail_error_band_xy, 0.0045)
        self.assertEqual(cfg.nladrc_ff_age_disable_sec, 0.16)
        self.assertEqual(cfg.nladrc_wc_xy_tail, 15.0)
        self.assertEqual(cfg.nladrc_ff_mix_gain, 0.95)
        self.assertEqual(cfg.nladrc_ff_mix_gain_boost, 1.00)
        self.assertEqual(cfg.nladrc_ff_motion_floor_xy, 0.60)
        self.assertEqual(cfg.nladrc_ff_motion_exit_xy, 0.006)
        self.assertEqual(cfg.nladrc_ff_boost_exit_xy, 0.018)
        self.assertEqual(cfg.nladrc_ff_lead_clip_xy, 0.0012)
        self.assertEqual(cfg.nladrc_mode_blend_alpha_xy, 0.30)
        self.assertEqual(cfg.nladrc_u_damp_gain_xy, 0.010)
        self.assertEqual(cfg.nladrc_u_damp_gain_boost_xy, 0.018)
        self.assertEqual(cfg.nladrc_u_damp_clip_xy, 0.045)

    def test_runtime_config_rejects_unknown_controller(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NOT_A_CONTROLLER"}))

    def test_runtime_config_rejects_invalid_new_nladrc_limits(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_z2_clip_xy": 0.0,
            }))

    def test_runtime_config_rejects_invalid_tail_gate_parameters(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_tail_u_fb_clip_xy": 0.5,
            }))

    def test_runtime_config_rejects_invalid_tail_feedback_parameters(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_delta_ctrl_xy_tail": 0.0,
            }))

    def test_runtime_config_rejects_invalid_boost_parameters(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_u_fb_clip_xy_boost": 0.20,
            }))

    def test_runtime_config_rejects_invalid_motion_hold_parameters(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_ff_motion_floor_xy": 1.5,
            }))

    def test_runtime_config_rejects_invalid_hysteresis_parameters(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_ff_motion_exit_xy": 0.02,
            }))

    def test_runtime_config_rejects_invalid_damping_parameters(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_u_damp_clip_xy": -0.01,
            }))


if __name__ == "__main__":
    unittest.main()
