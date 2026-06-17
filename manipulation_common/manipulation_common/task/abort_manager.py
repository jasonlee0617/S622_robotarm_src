import time
import threading
import rclpy


class AbortManager:
    """
    把以下功能集中管理：
    - /manual_abort 回调触发 abort event
    - cancel arm/gripper 的 moveit action
    - 可中断 wait（轮询 query_state）
    - recover：停下 -> disable keepout -> open gripper -> go_home -> reset cache -> clear abort flag
    """

    def __init__(self, node, arm, gripper):
        self._node = node
        self.arm = arm
        self.gripper = gripper

        self._event = threading.Event()
        self.reason = ""

    # --------- basic ---------
    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self):
        self._event.clear()
        self.reason = ""

    def request_abort(self, reason: str):
        self.reason = reason
        self._event.set()
        self._node.get_logger().warn(f"!!! Abort requested: {reason} !!!")

    # --------- ROS callback ---------
    def on_manual_abort(self, msg):
        if not msg.data:
            return
        self.request_abort("manual abort (/manual_abort)")
        self.cancel_all_motion_now()

    # --------- cancel ---------
    def _cancel_moveit(self, m, name: str):
        # 1) pymoveit2 stop
        try:
            m.cancel_execution()
        except Exception as e:
            self._node.get_logger().warn(f"[{name}] cancel_execution failed: {e}")

        # 2) best-effort cancel current goal_handle
        try:
            mutex = getattr(m, "_MoveIt2__execution_mutex", None)
            if mutex:
                mutex.acquire()
            gh = getattr(m, "_MoveIt2__execution_goal_handle", None)
            if mutex:
                mutex.release()

            if gh is not None:
                gh.cancel_goal_async()
        except Exception as e:
            self._node.get_logger().warn(f"[{name}] goal_handle cancel failed: {e}")

    def cancel_all_motion_now(self):
        self._node.get_logger().warn("Stopping arm & gripper...")
        try:
            self._cancel_moveit(self.arm, "arm")
        except Exception:
            pass
        try:
            self._cancel_moveit(self.gripper, "gripper")
        except Exception:
            pass

    # --------- interruptible wait ---------
    def wait_idle_or_abort(self, m, action_name: str, timeout_sec: float) -> bool:
        """
        代替 wait_until_executed()
        - abort 优先：触发时立即 cancel 并返回 False
        - query_state()==IDLE 时返回 motion_suceeded
        """
        t0 = time.time()
        while rclpy.ok():
            if self.is_set():
                self._node.get_logger().warn(f"ABORT while waiting: {action_name}")
                self.cancel_all_motion_now()
                return False

            try:
                st = m.query_state()
                st_val = st.value if hasattr(st, "value") else int(st)
            except Exception:
                st_val = 0

            # IDLE == 0
            if st_val == 0:
                return bool(getattr(m, "motion_suceeded", False))

            if (time.time() - t0) > timeout_sec:
                self._node.get_logger().error(f"{action_name} timeout -> force stop")
                self.cancel_all_motion_now()
                try:
                    m.force_reset_executing_state()
                except Exception:
                    pass
                return False

            time.sleep(0.02)

        return False

    # --------- recovery ---------
    def recover(
        self,
        keepout=None,
        open_gripper_fn=None,
        go_home_fn=None,
        reset_fn=None,
        restore_arm_limits_fn=None,
    ) -> bool:
        """
        返回：是否成功回 home
        """
        self._node.get_logger().warn(f"=== RECOVER FROM ABORT: {self.reason} ===")

        # 1) stop now
        self.cancel_all_motion_now()

        # 2) disable keepout
        try:
            if keepout is not None and getattr(keepout, "enabled", False):
                keepout.disable()
        except Exception:
            pass

        # 3) open gripper (safer)
        try:
            if open_gripper_fn is not None:
                open_gripper_fn()
        except Exception:
            pass

        # 4) restore arm limits (optional)
        try:
            if restore_arm_limits_fn is not None:
                restore_arm_limits_fn()
        except Exception:
            pass

        # 5) go home
        ok_home = False
        try:
            if go_home_fn is not None:
                ok_home = bool(go_home_fn())
        except Exception:
            ok_home = False

        # 6) reset caches
        try:
            if reset_fn is not None:
                reset_fn()
        except Exception:
            pass

        # 7) clear abort flag
        self.clear()
        return ok_home
