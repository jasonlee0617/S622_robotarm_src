import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Any

from geometry_msgs.msg import Pose


@dataclass
class PlanScoreConfig:
    num_candidates: int = 5
    wrist_weight: float = 50.0
    wrist_joint_indices: Tuple[int, int, int] = (2, 3, 4)


class PlannerSwitch:
    FAIRINO_PLANNERS = {"birrt*", "rrt*", "aapf_birrt*"}

    @staticmethod
    def normalize_pipeline(value: str) -> str:
        token = str(value or "").strip().lower()
        if token in ("fairino", "fairino_planning"):
            return "fairino"
        if token in ("kdl", "ompl"):
            return "ompl"
        return token

    @staticmethod
    def normalize_ik(value: str) -> str:
        token = str(value or "").strip().lower()
        if token in ("fairino", "fr", "fairino_ik"):
            return "fairino"
        if token in ("kdl", "ompl"):
            return "kdl"
        return token

    @staticmethod
    def normalize_planner(pipeline: str, planner: str) -> str:
        raw = str(planner or "").strip()
        key = raw.lower()
        if PlannerSwitch.normalize_pipeline(pipeline) == "fairino":
            fairino_aliases = {
                "aapf": "aapf_birrt*",
                "aapf_birrt": "aapf_birrt*",
                "aapf-birrt": "aapf_birrt*",
                "aapf_birrt*": "aapf_birrt*",
                "aapf-birrt*": "aapf_birrt*",
                "birrt": "birrt*",
                "birrt*": "birrt*",
                "rrt": "rrt*",
                "rrt*": "rrt*",
            }
            return fairino_aliases.get(key, raw)
        if key in ("rrtconnect", "rrtconnectkconfigdefault"):
            return "RRTConnect"
        return raw

    @staticmethod
    def is_valid(pipeline: str, planner: str) -> bool:
        normalized_pipeline = PlannerSwitch.normalize_pipeline(pipeline)
        normalized_planner = PlannerSwitch.normalize_planner(normalized_pipeline, planner)
        if normalized_pipeline == "fairino":
            return normalized_planner in PlannerSwitch.FAIRINO_PLANNERS
        return bool(normalized_planner)


class MoveItMotion:
    def __init__(
        self,
        node,
        arm_clients: dict,
        default_client: str,
        gripper,
        pose_tools,
        abort=None,
        select_best_path: Optional[Callable[..., Any]] = None,
        score_cfg: PlanScoreConfig = PlanScoreConfig(),
        action_delay: float = 0.5,
        joint_constraint: Optional[dict] = None,
        open_positions: Sequence[float] = (0.0305, -0.0305),
        close_positions: Sequence[float] = (0.0, 0.0),
    ):
        self.node = node
        self.arm_clients = dict(arm_clients)
        self.current_client = PlannerSwitch.normalize_ik(default_client)
        self.gripper = gripper
        self.pose_tools = pose_tools
        self.abort = abort
        self.select_best_path = select_best_path
        self.score_cfg = score_cfg
        self.action_delay = float(action_delay)
        self.joint_constraint = joint_constraint
        self.open_positions = tuple(open_positions)
        self.close_positions = tuple(close_positions)

    @property
    def arm(self):
        return self._select_arm(None)

    def _select_arm(self, planning_client: Optional[str]):
        key = PlannerSwitch.normalize_ik(planning_client or self.current_client)
        selected = self.arm_clients.get(key)
        if selected is not None:
            return selected
        self.node.get_logger().warn(f"Unknown planning client '{planning_client}', fallback to fairino.")
        return self.arm_clients.get("fairino", next(iter(self.arm_clients.values())))

    def set_ik(self, plugin: str) -> bool:
        key = PlannerSwitch.normalize_ik(plugin)
        if key not in self.arm_clients:
            self.node.get_logger().error(f"Unsupported IK plugin '{plugin}'. Use fairino or kdl.")
            return False
        self.current_client = key
        if self.abort is not None:
            self.abort.arm = self._select_arm(key)
        self.node.get_logger().info(f"IK client switched: {key}")
        return True

    def set_planner(self, pipeline: str, planner: str, raw_algorithm: Optional[str] = None) -> bool:
        normalized_pipeline = PlannerSwitch.normalize_pipeline(pipeline)
        normalized_planner = PlannerSwitch.normalize_planner(normalized_pipeline, planner)
        raw = raw_algorithm if raw_algorithm is not None else planner
        if not PlannerSwitch.is_valid(normalized_pipeline, normalized_planner):
            self.node.get_logger().error(
                f"Unsupported planner command: pipeline={pipeline}, algorithm={raw}. "
                "Fairino supports birrt*, rrt*, aapf_birrt*."
            )
            return False

        if normalized_pipeline == "fairino":
            self.set_ik("fairino")
            targets = [self.arm_clients.get("fairino")]
        else:
            self.set_ik("kdl")
            targets = [self.arm_clients.get("kdl")]

        for arm in targets:
            if arm is None:
                continue
            arm.pipeline_id = normalized_pipeline
            arm.planner_id = normalized_planner

        self.node.get_logger().info(
            f"Planner switched: pipeline={normalized_pipeline}, "
            f"raw_algorithm={raw}, algorithm={normalized_planner}"
        )
        return True

    def handle_command(self, msg):
        text = str(msg.data).strip()
        if not text:
            return
        parts = text.split()
        if len(parts) >= 2 and parts[0].lower() == "ik":
            self.set_ik(parts[1])
            return
        if len(parts) >= 3 and parts[0].lower() == "planner":
            self.set_planner(parts[1], parts[2], raw_algorithm=parts[2])
            return
        self.node.get_logger().warn(
            "Unsupported planner command. Use 'ik fairino', 'ik kdl', "
            "'planner fairino birrt*', or 'planner ompl RRTConnect'."
        )

    def move_to_pose(
        self,
        target_pose,
        planning_client: Optional[str] = None,
        cartesian: bool = False,
        action_name: str = "move",
        max_velocity: float = 0.05,
        max_acceleration: float = 0.05,
        timeout_sec: float = 30.0,
        joint_constraint: Optional[dict] = None,
    ) -> bool:
        arm = self._select_arm(planning_client)
        if isinstance(target_pose, Pose):
            target_pose = self.pose_tools.to_pose_stamped(target_pose)

        self.node.get_logger().info(
            f"{action_name}: ({target_pose.pose.position.x:.3f}, "
            f"{target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f}), "
            f"cartesian={bool(cartesian)}, client={self._client_name(arm)}"
        )

        try:
            arm.max_velocity = float(max_velocity)
            arm.max_acceleration = float(max_acceleration)
            paths = []
            for _ in range(max(1, int(self.score_cfg.num_candidates))):
                if self._aborted():
                    return False
                try:
                    arm.clear_path_constraints()
                    constraint = joint_constraint if joint_constraint is not None else self.joint_constraint
                    if constraint is not None:
                        arm.set_path_joint_constraint(
                            joint_positions=constraint["joint_positions"],
                            joint_names=constraint["joint_names"],
                            tolerance=constraint.get("tolerance", 0.0),
                            weight=constraint.get("weight", 1.0),
                        )
                    plan = arm.plan(target_pose, cartesian=cartesian)
                    if plan:
                        paths.append(plan)
                except Exception as exc:
                    self.node.get_logger().warn(f"{action_name}: plan failed: {exc}")

            if not paths:
                self.node.get_logger().error(f"{action_name}: No valid plan generated.")
                return False
            if self._aborted():
                self.node.get_logger().warn(f"{action_name}: aborted before execute")
                return False

            best_path = self._pick_path(paths, cartesian)
            best_path = arm._retime_trajectory_if_needed(best_path, cartesian=cartesian)
            arm.execute(best_path)
            ok = self._wait(arm, action_name, timeout_sec)
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            self.node.get_logger().info(f"✓ {action_name} done.")
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().error(f"✗ {action_name} exception: {exc}")
            return False

    def move_to_joints(
        self,
        joint_positions: Sequence[float],
        action_name: str = "move_joints",
        planning_client: Optional[str] = None,
        timeout_sec: float = 30.0,
    ) -> bool:
        arm = self._select_arm(planning_client)
        try:
            self.node.get_logger().info(action_name)
            arm.move_to_configuration(list(joint_positions))
            ok = self._wait(arm, action_name, timeout_sec)
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
        action_name: Optional[str] = None,
        timeout_sec: float = 10.0,
    ) -> bool:
        if action_name is None:
            action_name = "Open gripper" if open_gripper else "Close gripper"
        positions = self.open_positions if open_gripper else self.close_positions
        self.node.get_logger().info(action_name)
        try:
            self.gripper.move_to_configuration(list(positions))
            ok = self._wait(self.gripper, action_name, timeout_sec)
            if not ok:
                self.node.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as exc:
            self.node.get_logger().warn(f"{action_name} exception: {exc}")
            time.sleep(self.action_delay)
            return False

    def _pick_path(self, paths, cartesian: bool):
        if cartesian or self.select_best_path is None or len(paths) == 1:
            return paths[0]
        return self.select_best_path(
            paths,
            wrist_weight=self.score_cfg.wrist_weight,
            wrist_joint_indices=self.score_cfg.wrist_joint_indices,
        )

    def _wait(self, moveit_obj, action_name: str, timeout_sec: float) -> bool:
        if self.abort is not None:
            return self.abort.wait_idle_or_abort(moveit_obj, action_name, timeout_sec=float(timeout_sec))
        time.sleep(min(float(timeout_sec), 0.5))
        return True

    def _aborted(self) -> bool:
        return bool(self.abort.is_set()) if self.abort is not None else False

    def _client_name(self, arm) -> str:
        for key, value in self.arm_clients.items():
            if value is arm:
                return key
        return "unknown"
