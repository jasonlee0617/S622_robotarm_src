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


def _controller():
    return NLADRC_1st_Order(
        wc=14.0,
        wo=20.0,
        b0=0.5,
        dt=0.004,
        alpha_obs=0.85,
        alpha_obs2=0.70,
        alpha_ctrl=0.90,
        delta_obs=0.004,
        delta_ctrl=0.003,
        err_transition=0.015,
        obs_error_clip=0.02,
        obs_transition=0.004,
        z2_clip=0.12,
        u_fb_clip=0.28,
        u_rate_max=0.60,
        u_ema_alpha=0.95,
        u_clip=0.30,
    )


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
        outputs = [ctrl.step(0.02, u_ff=0.01) for _ in range(20)]
        self.assertTrue(all(np.isfinite(outputs)))
        self.assertLessEqual(max(abs(u) for u in outputs), 0.30)
        self.assertTrue(all(abs(outputs[i] - outputs[i - 1]) <= (0.60 * 0.004 + 1e-9) for i in range(1, len(outputs))))

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
        self.assertLessEqual(abs(ctrl.last_debug.u_fb), 0.28 + 1e-9)
        self.assertAlmostEqual(ctrl.last_debug.e_obs, 0.02)

    def test_command_pre_and_shaped_stay_close_on_step(self):
        ctrl = _controller()
        diffs = []
        for _ in range(200):
            u = ctrl.step(0.02, u_ff=0.015)
            ctrl.commit_applied_command(u)
            diffs.append(abs(ctrl.last_debug.u_cmd_pre - ctrl.last_debug.u_cmd_shaped))
        self.assertLess(max(diffs[-20:]), 0.05)

    def test_runtime_config_accepts_nladrc(self):
        cfg = ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NLADRC"}))
        self.assertEqual(cfg.servo_controller_type, "NLADRC")
        self.assertEqual(cfg.servo_controller_family, "NLADRC")
        self.assertEqual(cfg.pid_variant, "NONE")
        self.assertEqual(cfg.nladrc_obs_transition_xy, 0.004)
        self.assertEqual(cfg.nladrc_z2_clip_xy, 0.12)
        self.assertEqual(cfg.nladrc_u_fb_clip_xy, 0.28)

    def test_runtime_config_rejects_unknown_controller(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NOT_A_CONTROLLER"}))

    def test_runtime_config_rejects_invalid_new_nladrc_limits(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({
                "servo_controller_type": "NLADRC",
                "nladrc_z2_clip_xy": 0.0,
            }))


if __name__ == "__main__":
    unittest.main()
