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
        wc=12.0,
        wo=32.0,
        b0=0.5,
        dt=0.004,
        alpha_obs=0.50,
        alpha_obs2=0.25,
        alpha_ctrl=0.75,
        delta_obs=0.0025,
        delta_ctrl=0.0020,
        err_transition=0.008,
        obs_error_clip=0.02,
        u_rate_max=0.60,
        u_ema_alpha=0.35,
        u_clip=0.24,
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
        outputs = [ctrl.step(0.02) for _ in range(20)]
        self.assertTrue(all(np.isfinite(outputs)))
        self.assertLessEqual(max(abs(u) for u in outputs), 0.24)
        self.assertTrue(all(abs(outputs[i] - outputs[i - 1]) <= (0.60 * 0.004 + 1e-9) for i in range(1, len(outputs))))

    def test_nladrc_reset_clears_state(self):
        ctrl = _controller()
        ctrl.step(0.02)
        ctrl.reset()
        self.assertEqual(ctrl.z1, 0.0)
        self.assertEqual(ctrl.z2, 0.0)
        self.assertEqual(ctrl.u_last, 0.0)

    def test_runtime_config_accepts_nladrc(self):
        cfg = ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NLADRC"}))
        self.assertEqual(cfg.servo_controller_type, "NLADRC")
        self.assertEqual(cfg.servo_controller_family, "NLADRC")
        self.assertEqual(cfg.pid_variant, "NONE")

    def test_runtime_config_rejects_unknown_controller(self):
        with self.assertRaises(RuntimeError):
            ServoRuntimeConfig.from_node(_FakeNode({"servo_controller_type": "NOT_A_CONTROLLER"}))


if __name__ == "__main__":
    unittest.main()
