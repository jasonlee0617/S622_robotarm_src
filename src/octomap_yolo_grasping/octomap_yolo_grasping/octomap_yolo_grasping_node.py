#!/usr/bin/env python3
import time
from enum import Enum
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose, PointStamped
from std_msgs.msg import String, Float32MultiArray
from scipy.spatial.transform import Rotation as R
from pymoveit2 import MoveIt2
# keepout + scoring modules (moved out)
from manipulation_common.planning.keepout_manager import KeepoutManager, KeepoutConfig
from manipulation_common.planning.trajectory_scoring import select_best_path
from moveit_msgs.msg import CollisionObject, PlanningScene

from manipulation_common.utils.tf_tools import TfTools
from manipulation_common.utils.pose_tools import PoseTools
from manipulation_common.task.abort_manager import AbortManager
from std_msgs.msg import Bool

class TargetType(Enum):
    PEN = "pen"
    CUBE = "cube"


class TaskState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    MOVING_TO_TARGET_ABOVE = "moving_to_target_above"
    MOVING_TO_TARGET = "moving_to_target"
    GRASPING = "grasping"
    LIFTING_TARGET = "lifting_target"
    MOVING_TO_BOX = "moving_to_box"
    RELEASING = "releasing"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ERROR = "error"


class PenCubeBoxGraspingNode(Node):

    def __init__(self):
        super().__init__("pen_cube_box_grasping")

        self.callback_group = ReentrantCallbackGroup()
        # control_loop 不允许重入（状态机安全）
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        # abort 回调必须能在运动阻塞时也执行（与 control_loop 分组隔离）
        self.abort_cb_group = MutuallyExclusiveCallbackGroup()

        # --- 可手动设置优先级 ---
        # auto: 同时存在时用 preferred_target 选择；如果 preferred_target 无效则默认 pen
        # pen: 强制优先 pen
        # cube: 强制优先 cube
        self.declare_parameter("preferred_target", "pen")  # "pen" or "cube"
        self.preferred_target = str(self.get_parameter("preferred_target").value).lower().strip()

        #frames
        self.base_frame = "base_link"
        self.camera_frame = "camera_color_optical_frame"
        #F tools
        self.tf_tools = TfTools(self, base_frame=self.base_frame, camera_frame=self.camera_frame)
        
        #Pose tools 
        self.pose_tools = PoseTools(self, base_frame=self.base_frame)

        # 初始化各个子系统
        self.setup_detection_subscribers()
        self.setup_moveit()
        self.setup_params()

        # keepout manager
        cfg = KeepoutConfig(object_id="z_keepout",frame_id="base_link",thickness=self.KEEP_OUT_THICKNESS,xy_size=self.KEEP_OUT_XY_SIZE,)
        self.keepout = KeepoutManager(self, self.collision_obj_pub, self.planning_scene_pub, cfg)

        # 任务状态
        self.current_state = TaskState.IDLE
        self.active_target: TargetType | None = None  # 当前执行的目标（pen/cube）

        # NEW: Abort manager 
        self.abort = AbortManager(self, arm=self.moveit2_arm, gripper=self.moveit2_gripper)
        # 检测结果缓存
        self.target_pen_position: PointStamped | None = None
        self.target_cube_position: PointStamped | None = None
        self.target_box_position: PointStamped | None = None
        self.target_pen_rpy: dict | None = None
        self.target_cube_rpy: dict | None = None
        self.poses = {}

        # --- PATCH: subscribe manual abort ---
        self.create_subscription(Bool,"/manual_abort",self.abort.on_manual_abort,10,callback_group=self.abort_cb_group,)

        # 状态发布和控制循环
        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        # --- PATCH: faster control loop ---
        self.create_timer(0.2, self.control_loop, callback_group=self.control_cb_group)
        
    # ---------------- Subscribers ----------------
    def setup_detection_subscribers(self):
        qos_reliable_latest = QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1, reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
        self.collision_obj_pub = self.create_publisher(CollisionObject, "/collision_object", qos_reliable_latest)
        self.planning_scene_pub = self.create_publisher(PlanningScene, "/planning_scene", qos_reliable_latest)

        self.create_subscription(PointStamped, "/pen_position_3d", self._on_pen_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/cube_position_3d", self._on_cube_pos, qos_reliable_latest)
        self.create_subscription(PointStamped, "/box_position_3d", self._on_box_pos, qos_reliable_latest)
        self.create_subscription(Float32MultiArray, "/pen_rpy", self._on_pen_rpy, qos_reliable_latest)
        self.create_subscription(Float32MultiArray, "/cube_rpy", self._on_cube_rpy, qos_reliable_latest)
        self.get_logger().info("✓ Detection subscribers set")

    def _on_pen_pos(self, msg: PointStamped):
        self.target_pen_position = msg
    def _on_cube_pos(self, msg: PointStamped):
        self.target_cube_position = msg
    def _on_box_pos(self, msg: PointStamped):
        self.target_box_position = msg
    def _on_pen_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.target_pen_rpy = {"roll": msg.data[0], "pitch": msg.data[1], "yaw": msg.data[2]}
        else:
            self.get_logger().warn("⚠ /pen_rpy format error")
    def _on_cube_rpy(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.target_cube_rpy = {"roll": msg.data[0], "pitch": msg.data[1], "yaw": msg.data[2]}
        else:
            self.get_logger().warn("⚠ /cube_rpy format error")

    # ---------------- MoveIt ----------------
    def setup_moveit(self):
        try:
            self.moveit2_arm = MoveIt2(
                node=self,
                joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                base_link_name="base_link",
                end_effector_name="grasp_frame",
                # end_effector_name="wrist3_link",
                group_name="robot_arm",
                # group_name="robot_arm_tip",
                callback_group=self.callback_group,
            )
            self.moveit2_arm.planner_id = "RRTConnect"
            self.moveit2_arm.max_step_size = 0.05
            self.moveit2_arm.max_velocity = 0.15
            self.moveit2_arm.max_acceleration = 0.15
            self.moveit2_arm.allowed_planning_time = 15.0
            self.moveit2_arm.position_tolerance = 0.005
            self.moveit2_arm.orientation_tolerance = 0.005

            self.moveit2_gripper = MoveIt2(
                node=self,
                joint_names=["finger1_joint", "finger2_joint"],
                base_link_name="base_link",
                end_effector_name="grasp_frame",
                group_name="hand",
                callback_group=self.callback_group,
            )
            self.get_logger().info("✓ MoveIt2 initialized")
        except Exception as e:
            self.get_logger().error(f"✗ MoveIt init failed: {e}")

    # ---------------- Params ----------------
    def setup_params(self):

        self.safe_height = 0.04
        self.grasp_offset = 0.008
        self.place_offset = 0.20

        self.home_joints = [-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0]
        self.action_delay = 0.5
        self.detection_timeout = 3.0

        # planning scoring
        self.NUM_CANDIDATE_PLANS = 5
        self.WRIST_WEIGHT = 50.0
        self.WRIST_JOINT_INDICES = (2, 3, 4) #约束关节

        self.grasp_profile = {
        TargetType.PEN: {"roll": 0.0,"pitch": -180.0,"yaw_offset": -135.0,"above_z": 0.03,"grasp_z": 0.00,},
        TargetType.CUBE: {"roll": 0.0,"pitch": -180.0,"yaw_offset": -135.0,"above_z": 0.05,"grasp_z": 0.01,},
        }

        # keepout params
        self.Z_MIN = 0.06
        self.KEEP_OUT_THICKNESS = 0.06
        self.KEEP_OUT_XY_SIZE = 0.5

        self.get_logger().info("✓ Params set")

    # ---------- reset helpers (used by abort recovery) ----------
    def _reset_task_cache(self):
        self.active_target = None
        self.poses = {}
        self.target_pen_position = None
        self.target_cube_position = None
        self.target_box_position = None
        self.target_pen_rpy = None
        self.target_cube_rpy = None

    def _restore_arm_limits(self):
        try:
            self.moveit2_arm.max_velocity = 0.15
            self.moveit2_arm.max_acceleration = 0.15
        except Exception:
            pass

    def _msg_age_sec(self, msg_header_stamp):
        now = self.get_clock().now()
        t = rclpy.time.Time.from_msg(msg_header_stamp)
        return (now - t).nanoseconds / 1e9

    # ---------------- Motion ----------------
    def move_to_pose(self, target_pose, cartesian=False, action_name="move", max_velocity=0.05, max_acceleration=0.05):
        if isinstance(target_pose, Pose):
            target_pose = self.pose_tools.to_pose_stamped(target_pose)

        self.get_logger().info(f"{action_name}: "f"({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})")

        try:
            self.moveit2_arm.max_velocity = max_velocity
            self.moveit2_arm.max_acceleration = max_acceleration

            paths = []
            for _ in range(self.NUM_CANDIDATE_PLANS):
                if self.abort.is_set():
                    return False
                try:
                    self.moveit2_arm.clear_path_constraints() 
                    self.moveit2_arm.set_path_joint_constraint(joint_positions=[-1.5708],joint_names=["j2"],tolerance=1.5708,weight=1.0)
                    p = self.moveit2_arm.plan(target_pose, cartesian=cartesian)
                    if p:
                        paths.append(p)
                except Exception as e:
                    self.get_logger().warn(f"plan failed: {e}")

            if not paths:
                self.get_logger().error("No valid plan generated.")
                return False

            # --- PATCH: allow abort before execute ---
            if self.abort.is_set():
                self.get_logger().warn(f"{action_name}: aborted before execute")
                return False
            
            # best_path = self.select_best_path(paths)
            best_path = select_best_path(paths,wrist_weight=self.WRIST_WEIGHT,wrist_joint_indices=self.WRIST_JOINT_INDICES,)
            best_path = self.moveit2_arm._retime_trajectory_if_needed(best_path, cartesian=cartesian)
            self.moveit2_arm.execute(best_path)
            
            # ok = self.moveit2_arm.wait_until_executed()
            # --- PATCH: interruptible wait ---
            ok = self.abort.wait_idle_or_abort(self.moveit2_arm, action_name, timeout_sec=30.0)
            if not ok:
                self.get_logger().error(f"✗ {action_name} aborted/failed.")
                return False

            self.get_logger().info(f"✓ {action_name} done.")
            time.sleep(self.action_delay)
            return True

        except Exception as e:
            self.get_logger().error(f"✗ {action_name} exception: {e}")
            return False

    def control_gripper(self, open_gripper=True):
        action = "Open gripper" if open_gripper else "Close gripper"
        self.get_logger().info(action)

        positions = [0.0305, -0.0305] if open_gripper else [0.0, 0.0]
        try:
            self.moveit2_gripper.move_to_configuration(positions)
            # ok = self.moveit2_gripper.wait_until_executed()
            ok = self.abort.wait_idle_or_abort(self.moveit2_gripper, action, timeout_sec=10.0)
            if not ok:
                self.get_logger().error(f"✗ {action} aborted/failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as e:
            self.get_logger().warn(f"{action} exception: {e}")
            time.sleep(self.action_delay)
            return False

    def go_home(self):
        self.get_logger().info("Go HOME")
        try:
            self.moveit2_arm.move_to_configuration(self.home_joints)
            # ok = self.moveit2_arm.wait_until_executed()
            ok = self.abort.wait_idle_or_abort(self.moveit2_arm, "Go HOME", timeout_sec=30.0)
            if not ok:
                self.get_logger().error("✗ HOME failed.")
                return False
            time.sleep(self.action_delay)
            return True
        except Exception as e:
            self.get_logger().error(f"✗ HOME exception: {e}")
            return False

    # ---------------- Detection validity + target selection ----------------
    def _pair_valid(self, obj_pos: PointStamped, obj_rpy: dict, box_pos: PointStamped) -> bool:
        if obj_pos is None or obj_rpy is None or box_pos is None:
            return False
        obj_age = self._msg_age_sec(obj_pos.header.stamp)
        box_age = self._msg_age_sec(box_pos.header.stamp)
        return (obj_age < self.detection_timeout) and (box_age < self.detection_timeout)

    def select_target(self) -> TargetType | None:
        pen_ok = self._pair_valid(self.target_pen_position, self.target_pen_rpy, self.target_box_position)
        cube_ok = self._pair_valid(self.target_cube_position, self.target_cube_rpy, self.target_box_position)

        if pen_ok and not cube_ok:
            return TargetType.PEN
        if cube_ok and not pen_ok:
            return TargetType.CUBE
        if pen_ok and cube_ok:
            if self.preferred_target == "cube":
                return TargetType.CUBE
            return TargetType.PEN  # 默认 pen
        return None

    # ---------------- Pose generation (pen/cube identical) ----------------
    def setup_poses_for_target(self, target: TargetType, obj_pos_base, box_pos_base):
        if target == TargetType.PEN:
            obj_rpy = self.target_pen_rpy
            obj_name = "pen"
        else:
            obj_rpy = self.target_cube_rpy
            obj_name = "cube"

        obj_y_deg = float(np.degrees(obj_rpy["yaw"]))
        obj_r_deg = float(np.degrees(obj_rpy["roll"]))
        obj_p_deg = float(np.degrees(obj_rpy["pitch"]))
        self.get_logger().info(f"✓ Using {obj_name} RPY(deg): R={obj_r_deg:.1f}, P={obj_p_deg:.1f}, Y={obj_y_deg:.1f}")

        # ✅ 要求：pen/cube 抓取姿态区别
        prof = self.grasp_profile[target]
        obj_roll = prof["roll"]
        obj_pitch = prof["pitch"]
        obj_yaw = prof["yaw_offset"] + obj_y_deg   # pen/cube 各自 offset
        box_roll, box_pitch, box_yaw = 0.0, -180.0, 0.0

        self.poses = {
            "target_above": self.pose_tools.make_pose(
                obj_pos_base.x,
                obj_pos_base.y,
                # obj_pos_base.x - self.grasp_offset,
                # obj_pos_base.y - 2*self.grasp_offset,
                0.03,  
                # obj_pos_base.z,
                obj_roll, obj_pitch, obj_yaw,
            ),
            "target_grasp": self.pose_tools.make_pose(
                obj_pos_base.x,
                obj_pos_base.y,
                # obj_pos_base.x - self.grasp_offset,
                # obj_pos_base.y - 2*self.grasp_offset,
                0.00,  # 你原来写死的抓取高度
                obj_roll, obj_pitch, obj_yaw,
            ),
            "target_lift": self.pose_tools.make_pose(
                obj_pos_base.x,
                obj_pos_base.y,
                # obj_pos_base.x - self.grasp_offset,
                # obj_pos_base.y - 2*self.grasp_offset,
                obj_pos_base.z + self.place_offset,
                obj_roll, obj_pitch, obj_yaw,
            ),
            "box_place": self.pose_tools.make_pose(
                box_pos_base.x,
                box_pos_base.y,
                box_pos_base.z + self.place_offset,
                obj_roll, obj_pitch, obj_yaw,
            ),
        }

        self.get_logger().info(f"✓ Poses ready for {obj_name}")
        for k, p in self.poses.items():
            self.get_logger().info(f"  - {k}: ({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f})")
    
    

    # ---------------- Main loop ----------------
    def control_loop(self):
        
        # --- PATCH: abort has priority ---
        if self.abort.is_set():
            ok_home = self.abort.recover(
                keepout=self.keepout,
                open_gripper_fn=lambda: self.control_gripper(True),
                go_home_fn=self.go_home,
                reset_fn=self._reset_task_cache,
                restore_arm_limits_fn=self._restore_arm_limits,
            )
            self.current_state = TaskState.SEARCHING if ok_home else TaskState.ERROR
            return
        
        if not self.tf_tools.ready:
            return

        # publish state
        s = String()
        s.data = self.current_state.value
        self.state_publisher.publish(s)

        try:
            if self.current_state == TaskState.IDLE:
                self.active_target = None
                if self.go_home():
                    self.current_state = TaskState.SEARCHING
                else:
                    self.current_state = TaskState.ERROR

            elif self.current_state == TaskState.SEARCHING:
                target = self.select_target()
                if target is None:
                    self.get_logger().info("⏳ Waiting for (pen+box) or (cube+box)...")
                    return
                
                # 确定本轮目标
                self.active_target = target

                # 取对应 PointStamped
                if target == TargetType.PEN:
                    obj_msg = self.target_pen_position
                else:
                    obj_msg = self.target_cube_position

                box_msg = self.target_box_position

                # TF -> base
                obj_pos_base = self.tf_tools.camera_point_to_base(obj_msg)
                box_pos_base = self.tf_tools.camera_point_to_base(box_msg)

                if obj_pos_base is None or box_pos_base is None:
                    self.get_logger().warn("⚠ TF transform failed, keep searching...")
                    return

                self.setup_poses_for_target(target, obj_pos_base, box_pos_base)

                # 打开夹爪准备抓
                self.control_gripper(True)
                time.sleep(0.5)
                self.current_state = TaskState.MOVING_TO_TARGET_ABOVE

            elif self.current_state == TaskState.MOVING_TO_TARGET_ABOVE:
                # if not self.keepout.enabled:
                #     self.keepout.enable(self.Z_MIN)

                if self.move_to_pose(
                    self.poses["target_above"],
                    cartesian=False,
                    action_name="Move to target above",
                    max_velocity=0.06,
                    max_acceleration=0.06,
                ):
                    self.current_state = TaskState.MOVING_TO_TARGET
                else:
                    self.current_state = TaskState.ERROR

            elif self.current_state == TaskState.MOVING_TO_TARGET:
                
                # if self.keepout.enabled:
                #     self.keepout.disable()

                if self.move_to_pose(
                    self.poses["target_grasp"],
                    cartesian=True,
                    action_name="Move to target grasp",
                    max_velocity=0.01,
                    max_acceleration=0.01,
                ):
                    self.current_state = TaskState.GRASPING
                else:
                    self.current_state = TaskState.ERROR

            elif self.current_state == TaskState.GRASPING:
                self.control_gripper(False)
                self.current_state = TaskState.LIFTING_TARGET

            elif self.current_state == TaskState.LIFTING_TARGET:
                if self.move_to_pose(
                    self.poses["target_lift"],
                    cartesian=True,
                    action_name="Lift target",
                    max_velocity=0.05,
                    max_acceleration=0.05,
                ):
                    self.current_state = TaskState.MOVING_TO_BOX
                else:
                    self.current_state = TaskState.ERROR

            elif self.current_state == TaskState.MOVING_TO_BOX:
                # if not self.keepout.enabled:
                #     self.keepout.enable(self.Z_MIN)

                if self.move_to_pose(
                    self.poses["box_place"],
                    cartesian=False,
                    action_name="Move to box place",
                    max_velocity=0.1,
                    max_acceleration=0.1,
                ):
                    self.current_state = TaskState.RELEASING
                else:
                    self.current_state = TaskState.ERROR

            elif self.current_state == TaskState.RELEASING:
                self.control_gripper(True)
                self.current_state = TaskState.RETURNING_HOME

            elif self.current_state == TaskState.RETURNING_HOME:
                if self.go_home():
                    self.control_gripper(False)
                    self.current_state = TaskState.COMPLETED
                else:
                    self.current_state = TaskState.ERROR

            elif self.current_state == TaskState.COMPLETED:
                self.get_logger().info("=== Task completed ===")
                # 清空本轮缓存（避免用到旧数据）
                self._reset_task_cache()
                time.sleep(0.5)
                self.current_state = TaskState.IDLE

            elif self.current_state == TaskState.ERROR:
                self.get_logger().error("!!! Task ERROR, recovering ...")
                self.control_gripper(True)
                time.sleep(0.5)

                if self.go_home():
                    self.get_logger().info("✓ Recovered, restart.")
                    # 清空本轮缓存（避免用到旧数据）
                    self._reset_task_cache()
                    self.current_state = TaskState.IDLE
                else:
                    self.get_logger().error("✗ Recovery failed, will retry.")
                    time.sleep(1.0)

        except Exception as e:
            self.get_logger().error(f"control_loop exception: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.current_state = TaskState.ERROR


def main(args=None):
    rclpy.init(args=args)
    node = PenCubeBoxGraspingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted.")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
