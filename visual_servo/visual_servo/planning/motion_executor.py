import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Any

from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath


@dataclass
class PlanScoreConfig:
    num_candidates: int = 5
    wrist_weight: float = 50.0
    wrist_joint_indices: Tuple[int, int, int] = (2, 3, 4)


class MoveItMotion:
    def __init__(
        self,
        node,
        arm,
        pose_tools,
        arm_clients: Optional[dict] = None,
        abort=None,
        gripper=None,
        select_best_path: Optional[Callable[..., Any]] = None,
        score_cfg: PlanScoreConfig = PlanScoreConfig(),
        action_delay: float = 0.5,
    ):
        self.node = node
        self.arm = arm
        self.arm_clients = dict(arm_clients or {})
        self.gripper = gripper
        self.pose_tools = pose_tools
        self.abort = abort
        self.select_best_path = select_best_path
        self.score_cfg = score_cfg
        self.action_delay = float(action_delay)
        self._fairino_cartesian_client = node.create_client(GetCartesianPath, "/fairino_cartesian_path")
        self._fairino_cartesian_start_guard_tol = self._node_param_float(
            "fairino.cartesian_path_planner.start_state_guard_tolerance_rad", 0.005
        )
        self.node.get_logger().info(
            "Motion planning policy: cartesian=True => no select_best_path; "
            "cartesian=False => select_best_path enabled."
        )

    def _node_param_float(self, name: str, fallback: float) -> float:
        try:
            if not self.node.has_parameter(name):
                self.node.declare_parameter(name, float(fallback))
            return float(self.node.get_parameter(name).value)
        except Exception:
            return float(fallback)

    def _select_arm(self, planning_client: Optional[str]):
        if not planning_client:
            return self.arm
        key = str(planning_client).strip().lower()
        selected = self.arm_clients.get(key)
        if selected is not None:
            return selected
        self.node.get_logger().warn(
            f"Unknown planning_client='{planning_client}', fallback to default arm client."
        )
        return self.arm

    def _planning_client_key(self, planning_client: Optional[str], arm) -> str:
        if planning_client:
            return str(planning_client).strip().lower()
        if self.arm_clients.get("fairino") is arm or self.arm is arm:
            return "fairino"
        if self.arm_clients.get("kdl") is arm:
            return "kdl"
        return ""

    def wait_client_ready(self, planning_client: Optional[str], timeout_sec: float = 3.0) -> bool:
        arm = self._select_arm(planning_client)
        cli = getattr(arm, "_plan_kinematic_path_service", None)
        if cli is None:
            return True
        return bool(cli.wait_for_service(timeout_sec=float(timeout_sec)))

    def _aborted(self) -> bool:
        return bool(self.abort.is_set()) if self.abort is not None else False

    def move_to_pose(
        self,
        target_pose,
        planning_client: Optional[str] = None,
        cartesian: bool = False,
        action_name: str = "move",
        max_velocity: float = 0.05,
        max_acceleration: float = 0.05,
        timeout_sec: float = 60.0,
        joint_constraint: Optional[dict] = None,
    ) -> bool:
        arm = self._select_arm(planning_client)
        if isinstance(target_pose, Pose):
            target_pose = self.pose_tools.to_pose_stamped(target_pose)
        planning_key = self._planning_client_key(planning_client, arm)
        if cartesian and planning_key == "fairino":
            planner_mode = "fairino_cartesian"
        elif cartesian:
            planner_mode = "moveit_cartesian"
        elif planning_key == "fairino":
            planner_mode = "fairino_global_single"
        else:
            planner_mode = "ompl_global_candidate_scored"
        self.node.get_logger().info(
            f"{action_name}: ({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}), "
            f"cartesian={bool(cartesian)}, planner_mode={planner_mode}"
        )
        try:
            arm.max_velocity = float(max_velocity)
            arm.max_acceleration = float(max_acceleration)
            num_trials = 1 if cartesian or planning_key == "fairino" else int(self.score_cfg.num_candidates)
            paths = []
            for _ in range(max(1, num_trials)):
                if self._aborted():
                    return False
                try:
                    arm.clear_path_constraints()
                    if joint_constraint is not None:
                        arm.set_path_joint_constraint(
                            joint_positions=joint_constraint["joint_positions"],
                            joint_names=joint_constraint["joint_names"],
                            tolerance=joint_constraint.get("tolerance", 0.0),
                            weight=joint_constraint.get("weight", 1.0),
                        )
                    if cartesian and planning_key == "fairino":
                        p = self._plan_fairino_cartesian(
                            arm=arm,
                            target_pose=target_pose,
                            action_name=action_name,
                            fraction_threshold=0.98,
                        )
                    else:
                        p = arm.plan(
                            target_pose,
                            cartesian=cartesian,
                            cartesian_fraction_threshold=0.98 if cartesian else 0.0,
                        )
                    if p:
                        paths.append(p)
                except Exception as exc:
                    self.node.get_logger().warn(f"{action_name}: plan failed: {exc}")
            if not paths:
                self.node.get_logger().error(f"{action_name}: No valid plan generated.")
                return False
            if self._aborted():
                self.node.get_logger().warn(f"{action_name}: aborted before execute")
                return False
            best_path = self._pick_path(paths=paths, cartesian=cartesian, action_name=action_name)
            best_path = arm._retime_trajectory_if_needed(best_path, cartesian=cartesian)
            arm.execute(best_path)
            if self.abort is not None:
                ok = self.abort.wait_idle_or_abort(arm, action_name, timeout_sec=float(timeout_sec))
            else:
                t0 = time.time()
                while time.time() - t0 < float(timeout_sec):
                    time.sleep(0.05)
                ok = True
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            self.node.get_logger().info(f"✓ {action_name} done.")
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().error(f"✗ {action_name} exception: {exc}")
            return False

    def _plan_fairino_cartesian(self, arm, target_pose, action_name: str, fraction_threshold: float):
        if not self._fairino_cartesian_client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().error(
                f"{action_name}: /fairino_cartesian_path service not available."
            )
            return None
        if arm.joint_state is None:
            self.node.get_logger().error(f"{action_name}: no joint state for Fairino Cartesian planner.")
            return None

        req = GetCartesianPath.Request()
        req.header = target_pose.header
        req.start_state.joint_state = arm.joint_state
        req.group_name = getattr(arm, "group_name", "robot_arm")
        req.link_name = arm.end_effector_name
        req.waypoints = [target_pose.pose]
        req.max_step = 0.0025
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        future = self._fairino_cartesian_client.call_async(req)
        rate = self.node.create_rate(200)
        while not future.done():
            if self._aborted():
                return None
            rate.sleep()
        res = future.result()
        if res is None:
            self.node.get_logger().error(f"{action_name}: Fairino Cartesian service returned None.")
            return None
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().warn(
                f"{action_name}: Fairino Cartesian planner failed: fraction={res.fraction:.3f}, "
                f"error={res.error_code.val}."
            )
            return None
        if res.fraction < float(fraction_threshold):
            self.node.get_logger().warn(
                f"{action_name}: Fairino Cartesian planner completed {res.fraction:.3f}, "
                f"less than threshold {fraction_threshold:.3f}."
            )
            return None
        if not self._fairino_cartesian_start_is_valid(res.solution.joint_trajectory, arm.joint_state, action_name):
            return None
        return res.solution.joint_trajectory

    def _fairino_cartesian_start_is_valid(self, trajectory, joint_state, action_name: str) -> bool:
        if not trajectory.points:
            self.node.get_logger().error(f"{action_name}: Fairino Cartesian trajectory has no points.")
            return False
        if not trajectory.points[0].positions:
            self.node.get_logger().error(f"{action_name}: Fairino Cartesian trajectory first point has no positions.")
            return False
        current_by_name = {
            name: joint_state.position[i]
            for i, name in enumerate(joint_state.name)
            if i < len(joint_state.position)
        }
        max_delta = 0.0
        max_joint = ""
        for i, name in enumerate(trajectory.joint_names):
            if i >= len(trajectory.points[0].positions) or name not in current_by_name:
                self.node.get_logger().error(
                    f"{action_name}: Fairino Cartesian start check missing joint '{name}'."
                )
                return False
            delta = abs(float(trajectory.points[0].positions[i]) - float(current_by_name[name]))
            if delta > max_delta:
                max_delta = delta
                max_joint = name
        if max_delta > self._fairino_cartesian_start_guard_tol:
            self.node.get_logger().error(
                f"{action_name}: Fairino Cartesian trajectory start mismatch: "
                f"joint={max_joint}, delta={max_delta:.6f} rad, "
                f"tol={self._fairino_cartesian_start_guard_tol:.6f} rad."
            )
            return False
        return True

    def _pick_path(self, paths, cartesian: bool, action_name: str):
        if not paths:
            raise ValueError("paths must not be empty")
        if cartesian:
            if len(paths) > 1:
                self.node.get_logger().warn(
                    f"{action_name}: cartesian=True requires direct path mode; "
                    f"received {len(paths)} candidates, using first and skipping scoring."
                )
            return paths[0]
        if self.select_best_path is not None and len(paths) > 1:
            try:
                best_path, best_score = self.select_best_path(
                    paths,
                    wrist_weight=self.score_cfg.wrist_weight,
                    wrist_joint_indices=self.score_cfg.wrist_joint_indices,
                    return_score=True,
                )
                if best_score is not None:
                    self.node.get_logger().info(
                        f"{action_name}: selected best trajectory from {len(paths)} candidates: "
                        f"cost={best_score.total_cost:.4f}, path={best_score.path_length:.4f}, "
                        f"wrist={best_score.wrist_length:.4f}, max_step={best_score.max_joint_step:.4f}, "
                        f"smooth={best_score.smoothness:.4f}, duration={best_score.duration:.3f}, "
                        f"valid={best_score.valid}"
                    )
                    if not best_score.valid and best_score.reason:
                        self.node.get_logger().warn(
                            f"{action_name}: best trajectory score warning: {best_score.reason}"
                        )
                return best_path if best_path is not None else paths[0]
            except TypeError:
                return self.select_best_path(
                    paths,
                    wrist_weight=self.score_cfg.wrist_weight,
                    wrist_joint_indices=self.score_cfg.wrist_joint_indices,
                )
        return paths[0]

    def move_to_joints(
        self,
        joint_positions: Sequence[float],
        action_name: str = "move_joints",
        timeout_sec: float = 60.0,
        planning_client: Optional[str] = None,
    ) -> bool:
        if self._aborted():
            return False
        arm = self._select_arm(planning_client)
        try:
            self.node.get_logger().info(action_name)
            arm.move_to_configuration(list(joint_positions))
            if self.abort is not None:
                ok = self.abort.wait_idle_or_abort(arm, action_name, timeout_sec=float(timeout_sec))
            else:
                time.sleep(min(self.action_delay, 0.5))
                ok = True
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().error(f"✗ {action_name} exception: {exc}")
            return False

    def control_gripper(
        self,
        open_gripper: bool = True,
        open_positions=(0.0305, -0.0305),
        close_positions=(0.01, -0.01),
        action_name: Optional[str] = None,
        timeout_sec: float = 20.0,
    ) -> bool:
        if self.gripper is None:
            self.node.get_logger().warn("control_gripper called but gripper MoveIt2 is None.")
            return False
        if action_name is None:
            action_name = "Open gripper" if open_gripper else "Close gripper"
        positions = list(open_positions if open_gripper else close_positions)
        self.node.get_logger().info(action_name)
        try:
            self.gripper.move_to_configuration(positions)
            if self.abort is not None:
                ok = self.abort.wait_idle_or_abort(self.gripper, action_name, timeout_sec=float(timeout_sec))
            else:
                time.sleep(min(self.action_delay, 0.5))
                ok = True
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().warn(f"{action_name} exception: {exc}")
            time.sleep(self.action_delay)
            return False


__all__ = ["MoveItMotion", "PlanScoreConfig"]
