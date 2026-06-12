#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


@dataclass
class ServoMetrics:
    samples: int = 0
    sum_err_sq: float = 0.0
    sum_dx_sq: float = 0.0
    sum_dy_sq: float = 0.0
    dv_samples: int = 0
    sum_dv_sq: float = 0.0
    max_err_norm: float = 0.0
    align_start_sec: float | None = None
    settle_time_sec: float | None = None
    status_fault_count: int = 0


class ServoMetricsMonitor(Node):
    def __init__(self):
        super().__init__("servo_metrics_monitor")
        self.declare_parameter("report_period_sec", 1.0)
        self.declare_parameter("aligned_hold_sec", 0.20)
        self.declare_parameter("active_states", ["SERVO_TRACK_ABOVE", "SERVO_TRACK_TO_BOX"])

        self.report_period_sec = float(self.get_parameter("report_period_sec").value)
        self.aligned_hold_sec = float(self.get_parameter("aligned_hold_sec").value)
        self.active_states = {str(x) for x in self.get_parameter("active_states").value}

        self.current_state = ""
        self.metrics = ServoMetrics()
        self._active = False
        self._last_cmd_xy: tuple[float, float] | None = None
        self._last_status_fault = False

        self.create_subscription(String, "/task_state", self._on_state, 10)
        self.create_subscription(Float32MultiArray, "/servo_error_xyyaw", self._on_error, 10)
        self.create_subscription(Float32MultiArray, "/servo_cmd_stages", self._on_cmd, 10)
        self.create_subscription(Float32MultiArray, "/servo_exec_feedback", self._on_exec_feedback, 10)
        self.create_timer(self.report_period_sec, self._report_metrics)

        self.get_logger().info(
            "Servo metrics monitor started: "
            f"report_period={self.report_period_sec:.2f}s aligned_hold={self.aligned_hold_sec:.2f}s"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _reset_window(self):
        self.metrics = ServoMetrics()
        self._last_cmd_xy = None
        self._last_status_fault = False

    def _on_state(self, msg: String):
        new_state = str(msg.data).strip()
        new_active = new_state in self.active_states
        if new_active and not self._active:
            self._reset_window()
            self.get_logger().info(f"Servo metrics active in state={new_state}")
        elif self._active and not new_active:
            self._report_metrics(final=True)
            self.get_logger().info(f"Servo metrics inactive, leaving state={self.current_state}")
            self._reset_window()
        self.current_state = new_state
        self._active = new_active

    def _on_error(self, msg: Float32MultiArray):
        if not self._active or len(msg.data) < 4:
            return
        dx = float(msg.data[0])
        dy = float(msg.data[1])
        aligned = bool(msg.data[3] > 0.5)
        err_norm = math.hypot(dx, dy)

        self.metrics.samples += 1
        self.metrics.sum_err_sq += err_norm * err_norm
        self.metrics.sum_dx_sq += dx * dx
        self.metrics.sum_dy_sq += dy * dy
        self.metrics.max_err_norm = max(self.metrics.max_err_norm, err_norm)

        now_sec = self._now_sec()
        if self.metrics.align_start_sec is None:
            self.metrics.align_start_sec = now_sec

        if aligned:
            if self.metrics.settle_time_sec is None:
                if not hasattr(self, "_aligned_since_sec") or self._aligned_since_sec is None:
                    self._aligned_since_sec = now_sec
                elif now_sec - self._aligned_since_sec >= self.aligned_hold_sec:
                    self.metrics.settle_time_sec = now_sec - self.metrics.align_start_sec
            return

        self._aligned_since_sec = None

    def _on_cmd(self, msg: Float32MultiArray):
        if not self._active or len(msg.data) < 8:
            return
        vx = float(msg.data[6])
        vy = float(msg.data[7])
        if self._last_cmd_xy is not None:
            dvx = vx - self._last_cmd_xy[0]
            dvy = vy - self._last_cmd_xy[1]
            dv = math.hypot(dvx, dvy)
            self.metrics.dv_samples += 1
            self.metrics.sum_dv_sq += dv * dv
        self._last_cmd_xy = (vx, vy)

    def _on_exec_feedback(self, msg: Float32MultiArray):
        if not self._active or len(msg.data) < 1:
            return
        status_code = int(msg.data[0])
        is_fault = status_code not in (-1, 0)
        if is_fault and not self._last_status_fault:
            self.metrics.status_fault_count += 1
        self._last_status_fault = is_fault

    def _report_metrics(self, final: bool = False):
        if not self._active or self.metrics.samples <= 0:
            return
        err_rms = math.sqrt(self.metrics.sum_err_sq / max(1, self.metrics.samples))
        dx_rms = math.sqrt(self.metrics.sum_dx_sq / max(1, self.metrics.samples))
        dy_rms = math.sqrt(self.metrics.sum_dy_sq / max(1, self.metrics.samples))
        dv_rms = math.sqrt(self.metrics.sum_dv_sq / max(1, self.metrics.dv_samples)) if self.metrics.dv_samples > 0 else 0.0
        settle = self.metrics.settle_time_sec
        prefix = "[FINAL]" if final else "[LIVE]"
        settle_text = "pending" if settle is None else f"{settle:.3f}s"
        self.get_logger().info(
            f"{prefix} state={self.current_state} "
            f"xy_rms={err_rms * 1000.0:.2f}mm "
            f"dx_rms={dx_rms * 1000.0:.2f}mm "
            f"dy_rms={dy_rms * 1000.0:.2f}mm "
            f"max_err={self.metrics.max_err_norm * 1000.0:.2f}mm "
            f"dv_rms={dv_rms:.5f} "
            f"settle={settle_text} "
            f"status_faults={self.metrics.status_fault_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ServoMetricsMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node._active:
            node._report_metrics(final=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
