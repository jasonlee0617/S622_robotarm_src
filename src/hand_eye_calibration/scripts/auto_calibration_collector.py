#!/usr/bin/env python3
"""
自动手眼标定数据采集 (eye_in_hand)

按键控制:
  s / Enter  → 开始自动采集 20 个标定位姿
  Space      → 暂停 / 恢复
  q          → 复位到 home 并退出

每个位姿: MoveIt2 运动到位 → 等待静止 → 检查 marker 可见 → take_sample
"""

import math
import sys
import termios
import threading
import time
from typing import List, Optional, Tuple

import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from ros2_aruco_interfaces.msg import ArucoMarkers

from easy_handeye2_msgs.srv import TakeSample
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R


# ── 20 个标定位姿定义 ──────────────────────────────────────────────
# 格式: (dx, dy, dz, droll_deg, dpitch_deg, dyaw_deg) 叠加到基准位姿

_CALIBRATION_OFFSETS = [
    #  样本  描述                  dx     dy     dz    roll  pitch  yaw
    ( 1,  "正中",                 0.0,   0.0,   0.0,   0,    0,    0),
    ( 2,  "左侧 yaw+15°",        0.0,   0.05,  0.0,   0,    0,   15),
    ( 3,  "右侧 yaw-15°",        0.0,  -0.05,  0.0,   0,    0,  -15),
    ( 4,  "上方 pitch+20°",      0.0,   0.0,   0.08,  0,   20,    0),
    ( 5,  "下方 pitch-20°",      0.0,   0.0,  -0.05,  0,  -20,    0),
    ( 6,  "左上 roll+20°",       0.0,   0.04,  0.04, 20,    0,    0),
    ( 7,  "右上 roll-20°",       0.0,  -0.04,  0.04,-20,    0,    0),
    ( 8,  "左下 pitch+roll混合", 0.0,   0.04, -0.04, 15,  -15,    0),
    ( 9,  "右下 pitch+yaw混合",  0.0,  -0.04, -0.04,  0,   15,  -15),
    (10,  "近距离正对",           0.08,  0.0,  -0.05,  0,    0,    0),
    (11,  "远距离正对",          -0.08,  0.0,   0.05,  0,    0,    0),
    (12,  "近距左侧 yaw+pitch",   0.06,  0.04,  0.0,   0,   10,   15),
    (13,  "近距右侧 yaw-pitch",   0.06, -0.04,  0.0,   0,  -10,  -15),
    (14,  "高位斜视 pitch大",     0.0,   0.0,   0.10,  0,   30,    0),
    (15,  "低位斜视 pitch大",     0.0,   0.0,  -0.08,  0,  -30,    0),
    (16,  "左侧 roll+30°",       0.0,   0.06,  0.0,  30,    0,    0),
    (17,  "右侧 roll-30°",       0.0,  -0.06,  0.0, -30,    0,    0),
    (18,  "斜上方 yaw+roll",      0.0,   0.03,  0.06,  15,    0,   15),
    (19,  "斜下方 yaw-roll",      0.0,  -0.03, -0.06, -15,    0,  -15),
    (20,  "回到初始位(检查重复性)", 0.0,   0.0,   0.0,   0,    0,    0),
]


# Home 位姿（安全关节角，单位 rad）
_HOME_JOINTS = [0.0, -1.57, 0.0, -0.785, 0.0, 0.0]


class AutoCalibrationCollector(Node):
    """自动手眼标定数据采集节点"""

    def __init__(self):
        super().__init__("auto_calibration_collector")

        # ── TF 参数 ──
        self.base_frame = (
            self.declare_parameter("base_frame", "base_link")
            .get_parameter_value().string_value
        )
        self.ee_frame = (
            self.declare_parameter("ee_frame", "grasp_frame")
            .get_parameter_value().string_value
        )

        # ── TF 监听 ──
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── 基准位姿（按 s 时从 TF 捕获） ──
        self._base_xyz = None
        self._base_rpy = None

        # ── MoveIt2 初始化 ──
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name="robot_arm",
            callback_group=ReentrantCallbackGroup(),
        )
        self.moveit2.pipeline_id = "ompl"
        self.moveit2.max_velocity = 0.15
        self.moveit2.max_acceleration = 0.10
        self.moveit2.allowed_planning_time = 5.0

        # ── 建图: 偏移索引 → 样本编号 ──
        self.pose_map = {
            idx: (desc, dx, dy, dz, dr, dp, dyaw)
            for idx, desc, dx, dy, dz, dr, dp, dyaw in _CALIBRATION_OFFSETS
        }

        # ── easy_handeye2 TakeSample 服务 ──
        self.sample_cli = self.create_client(
            TakeSample, "/easy_handeye2/calibration/take_sample"
        )

        # ── Marker 可见性检查 ──
        self._last_marker_msg: Optional[ArucoMarkers] = None
        self.create_subscription(
            ArucoMarkers, "/aruco_markers", self._on_markers, 1
        )

        # ── 键盘状态 ──
        self._start_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._quit_requested = threading.Event()
        self._resume_requested = threading.Event()

        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self._keyboard_thread.start()
        self._keyboard_help()

        # ── 采集结果 ──
        self.results: List[Tuple[int, str, bool, str]] = []  # (idx, desc, ok, note)

    # ── 键盘处理 ──────────────────────────────────────────────────

    def _keyboard_help(self):
        self.get_logger().info(
            "\n╔══════════════════════════════════════════════╗\n"
            "║  手眼标定自动采集 (eye_in_hand)              ║\n"
            "║                                              ║\n"
            "║  [s]  开始采集 20 个位姿                      ║\n"
            "║  [Space]  暂停 / 恢复                         ║\n"
            "║  [q]  复位到 home 并退出                       ║\n"
            "╚══════════════════════════════════════════════╝"
        )

    def _keyboard_loop(self):
        """独立线程: /dev/tty 非阻塞轮询"""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            import tty
            tty.setcbreak(fd)
            while rclpy.ok():
                import select
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("s", "\r", "\n"):
                    self._start_requested.set()
                elif ch == " ":
                    if self._pause_requested.is_set():
                        self._pause_requested.clear()
                        self._resume_requested.set()
                        self.get_logger().info("▶  恢复采集")
                    else:
                        self._pause_requested.set()
                        self._resume_requested.clear()
                        self.get_logger().info("⏸  暂停 (按 Space 恢复)")
                elif ch == "q":
                    self._quit_requested.set()
                    self.get_logger().info("🛑 请求退出...")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _wait_for_start_or_quit(self) -> bool:
        """阻塞直到 s(q) 按下。返回 True=开始, False=退出"""
        self.get_logger().info("等待按下 [s] 开始采集...")
        while rclpy.ok():
            if self._quit_requested.is_set():
                return False
            if self._start_requested.is_set():
                self._start_requested.clear()
                return True
            time.sleep(0.1)
        return False

    def _check_pause_or_quit(self) -> bool:
        """检查暂停/退出。返回 True=继续, False=退出"""
        while self._pause_requested.is_set() and not self._quit_requested.is_set():
            time.sleep(0.1)
        if self._quit_requested.is_set():
            return False
        return True

    def _on_markers(self, msg: ArucoMarkers):
        self._last_marker_msg = msg

    # ── 位姿构建 ──────────────────────────────────────────────────

    def _capture_base_pose(self) -> bool:
        """从 TF 读取当前末端位姿作为基准位姿"""
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, Time())
            p = t.transform.translation
            q = t.transform.rotation

            # 提取 RPY
            r = R.from_quat([q.x, q.y, q.z, q.w])
            rpy = r.as_euler("xyz", degrees=True)

            self._base_xyz = (float(p.x), float(p.y), float(p.z))
            self._base_rpy = tuple(float(v) for v in rpy)

            self.get_logger().info(
                f"📌 捕获基准位姿: xyz=({self._base_xyz[0]:.4f}, "
                f"{self._base_xyz[1]:.4f}, {self._base_xyz[2]:.4f}), "
                f"rpy=({self._base_rpy[0]:.1f}°, {self._base_rpy[1]:.1f}°, "
                f"{self._base_rpy[2]:.1f}°)"
            )
            return True
        except Exception as exc:
            self.get_logger().error(
                f"无法获取末端位姿 {self.base_frame}→{self.ee_frame}: {exc}"
            )
            return False

    def _build_pose(self, dx: float, dy: float, dz: float,
                    dr: float, dp: float, dyaw: float) -> PoseStamped:
        """基准位姿 + 偏移 → PoseStamped (base_link)"""
        # 基准方向: RPY (度) → 四元数
        base_r = R.from_euler("xyz", [math.radians(a) for a in self._base_rpy])
        # 偏移方向: 在基准坐标系下的增量旋转
        offset_r = R.from_euler("xyz", [math.radians(a) for a in (dr, dp, dyaw)])
        # 组合: R_final = R_base * R_offset
        final_r = base_r * offset_r
        q = final_r.as_quat()  # xyzw

        pose = Pose()
        pose.position = Point(
            x=float(self._base_xyz[0] + dx),
            y=float(self._base_xyz[1] + dy),
            z=float(self._base_xyz[2] + dz),
        )
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        return ps

    # ── Marker 可见性 ────────────────────────────────────────────

    def _check_marker_visible(self, timeout: float = 2.0) -> bool:
        """等待 /aruco_markers 中出现 ID=1"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._last_marker_msg is not None:
                for mid in self._last_marker_msg.marker_ids:
                    if mid == 1:
                        return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    # ── 采样触发 ──────────────────────────────────────────────────

    def _take_sample(self) -> bool:
        """调用 easy_handeye2 TakeSample 服务"""
        if not self.sample_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("TakeSample service not available")
            return False
        req = TakeSample.Request()
        future = self.sample_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.done() and future.result() is not None

    # ── 运动控制 ──────────────────────────────────────────────────

    def _wait_for_moveit(self, timeout: float = 30.0):
        """等待 move_group 就绪"""
        from moveit_msgs.srv import GetPositionIK
        tmp = self.create_client(GetPositionIK, "/compute_ik")
        self.get_logger().info("等待 /compute_ik 就绪 (move_group)...")
        if tmp.wait_for_service(timeout_sec=timeout):
            self.get_logger().info("✓ move_group 已就绪")
            self.destroy_client(tmp)
            return True
        self.destroy_client(tmp)
        self.get_logger().error(
            f"move_group 超时({timeout}s)未就绪，请确认 calibrate.launch.py 已启动"
        )
        return False

    def _go_home(self):
        """移动到安全 home 位姿"""
        self.get_logger().info("🏠 回到 home 位姿...")
        self.moveit2.move_to_configuration(
            joint_positions=list(_HOME_JOINTS),
            tolerance=0.02,
        )
        self.moveit2.wait_until_executed()

    def _move_and_sample(self, pose_idx: int, desc: str,
                         dx, dy, dz, dr, dp, dyaw) -> bool:
        """移动到指定位姿并采集样本"""
        if self._quit_requested.is_set():
            return False

        # 1. 构建目标位姿
        target = self._build_pose(dx, dy, dz, dr, dp, dyaw)
        self.get_logger().info(
            f"  ▶ [{pose_idx:2d}/20] {desc} "
            f"target=({target.pose.position.x:.3f},{target.pose.position.y:.3f},{target.pose.position.z:.3f})"
        )

        # 2. 运动到位
        try:
            self.moveit2.move_to_pose(pose=target)
        except Exception as exc:
            self.get_logger().error(f"  ✗ 规划失败: {exc}")
            return False

        # 3. 等待完成 + 残余振动衰减
        ok = self.moveit2.wait_until_executed()
        if not ok:
            self.get_logger().warn(f"  ⚠ 运动未完成, 跳过此位姿")
            return False
        time.sleep(1.0)  # 消除残余振动

        # 4. 检查 marker 是否可见
        if not self._check_marker_visible(timeout=3.0):
            self.get_logger().warn(f"  ⚠ marker 不可见, 跳过")
            self.results.append((pose_idx, desc, False, "marker not visible"))
            return False

        # 5. 触发 easy_handeye2 采集
        if not self._take_sample():
            self.get_logger().error(f"  ✗ TakeSample 失败")
            self.results.append((pose_idx, desc, False, "take_sample failed"))
            return False

        self.get_logger().info(f"  ✓ [{pose_idx:2d}/20] {desc} — sampled")
        self.results.append((pose_idx, desc, True, "ok"))
        return True

    # ── 主流程 ────────────────────────────────────────────────────

    def run(self):
        """主入口：等待开始 → 执行采集 → 打印结果"""
        if not self._wait_for_start_or_quit():
            return

        # 捕获当前末端位姿作为基准
        if not self._capture_base_pose():
            self.get_logger().error("基准位姿捕获失败，退出")
            return

        if not self._wait_for_moveit():
            self.get_logger().error("move_group 未就绪，请先启动 calibrate.launch.py")
            return

        self.get_logger().info("=" * 50)
        self.get_logger().info("开始自动采集 20 个标定位姿...")
        self.get_logger().info("=" * 50)

        total = len(_CALIBRATION_OFFSETS)
        for pose_idx, desc, dx, dy, dz, dr, dp, dyaw in _CALIBRATION_OFFSETS:
            if not self._check_pause_or_quit():
                break

            self._move_and_sample(pose_idx, desc, dx, dy, dz, dr, dp, dyaw)

            # 检查退出
            if self._quit_requested.is_set():
                break

        # 打印总结
        ok_count = sum(1 for _, _, ok, _ in self.results if ok)
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"采集完成: {ok_count}/{total} 成功")
        for idx, desc, ok, note in self.results:
            status = "✓" if ok else "✗"
            self.get_logger().info(f"  [{idx:2d}] {status} {desc}" + (f" ({note})" if not ok else ""))

        if ok_count < 12:
            self.get_logger().warn("成功率偏低，建议重新 jog 到一个 marker 清晰可见的位姿再试")

        # 回到 home
        if not self._quit_requested.is_set():
            self._go_home()
        self.get_logger().info("退出。可在 easy_handeye2 GUI 中点击 Compute → Save。")


def main():
    rclpy.init()
    node = AutoCalibrationCollector()
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)

    # 在 executor 线程中运行主流程
    def _run():
        time.sleep(1.0)  # 等待 MoveIt2 就绪
        node.run()

    import threading
    runner = threading.Thread(target=_run, daemon=True)
    runner.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
