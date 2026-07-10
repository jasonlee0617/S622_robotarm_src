#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Float64MultiArray, Empty

from manipulation_common.planning.motion_executor import MoveItMotion, PlanScoreConfig, PlannerSwitch
from manipulation_common.planning.trajectory_scoring import select_best_path
from manipulation_common.task.abort_manager import AbortManager

class Controller(Node):
    def __init__(self):
        super().__init__("yolo_pick_node")

        # ====== 状态变量 ======
        self._lock = threading.Lock()
        self._latest_target = None   # (x, y, yaw)
        self.is_executing = False
        self._worker_thread = None

        self.callback_group = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(Float64MultiArray,"/target_point",self.target_callback,10,)
        self.subscription_trigger = self.create_subscription(Empty,"/pick_trigger",self.trigger_callback,10,)

        self.setup_params()
        self.setup_moveit()

        self.get_logger().info("✅ Controller initialized")
        self.get_logger().info("等待：/target_point 更新目标，/pick_trigger 点击触发一次抓取")

    def setup_params(self):
         # 参数配置
        self.height = 0.12
        self.pick_height = 0.02
        self.carrying_height = 0.15
        self.place_x = 0.2
        self.place_y = 0.25
        self.place_release_height = 0.10
        self.init_angle = 1.5719
        # self.init_angle = -1.5888
        # self.home_joints = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]
        self.home_joints = [-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0]

        # ---------- 多次规划参数 ----------
        self.NUM_CANDIDATE_PLANS = 5

        # ---------- 代价函数权重（越大越抑制腕部乱转） ----------
        self.WRIST_WEIGHT = 50.0
        self.wrist_joint_indices = (2, 3, 4)

        self.arm_group_name = "robot_arm"
        self.hand_group_name = "hand"
        self.base_frame = "base_link"
        self.ee_frame = "grasp_frame"
        self.move_group_ns_fairino = "/move_group_fairino"
        self.move_group_ns_kdl = "/move_group_kdl"
        self.ik_plugin = "fairino"
        self.planning_pipeline_id = PlannerSwitch.normalize_pipeline("fairino")
        self.planner_id = PlannerSwitch.normalize_planner(self.planning_pipeline_id, "tube_birrt*")
        self.max_step_size = 0.05
        self.arm_max_velocity = 0.3
        self.arm_max_acceleration = 0.3
        self.allowed_planning_time = 15.0
        self.position_tolerance = 0.005
        self.orientation_tolerance = 0.005
        self.allowed_start_tolerance = 0.1
        self.action_delay = 1.0
        self.move_group_ready_timeout_sec = 10.0
        self.gripper_open_positions = (0.0305, -0.0305)
        self.gripper_close_positions = (0.0, 0.0)
        self.j2_constraint = {
            "joint_positions": [-1.5708],
            "joint_names": ["j2"],
            "tolerance": 1.5708,
            "weight": 1.0,
        }

    def setup_moveit(self):
        self.moveit2_arm_fairino = self._make_arm_client(self.move_group_ns_fairino)
        self.moveit2_arm_kdl = self._make_arm_client(self.move_group_ns_kdl)
        for arm in (self.moveit2_arm_fairino, self.moveit2_arm_kdl):
            self._configure_arm(arm)

        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.hand_group_name,
            callback_group=self.callback_group,
            move_group_namespace=self.move_group_ns_fairino,
        )
        self.moveit2_gripper.pipeline_id = "ompl"
        self.moveit2_gripper.planner_id = ""
        self.get_logger().info("MoveIt2 initialized")
        self.abort = AbortManager(self, arm=self.moveit2_arm_fairino, gripper=self.moveit2_gripper)
        self.create_subscription(Bool, "/manual_abort", self.abort.on_manual_abort, 10)

        self.motion = MoveItMotion(
            node=self,
            arm_clients={"fairino": self.moveit2_arm_fairino, "kdl": self.moveit2_arm_kdl},
            default_client=self.ik_plugin,
            gripper=self.moveit2_gripper,
            abort=self.abort,
            select_best_path=select_best_path,
            score_cfg=PlanScoreConfig(
                num_candidates=self.NUM_CANDIDATE_PLANS,
                wrist_weight=self.WRIST_WEIGHT,
                wrist_joint_indices=self.wrist_joint_indices,
            ),
            action_delay=self.action_delay,
            joint_constraint=self.j2_constraint,
            open_positions=self.gripper_open_positions,
            close_positions=self.gripper_close_positions,
        )
        self.motion.set_ik(self.ik_plugin)
        self.arm = self.moveit2_arm_fairino
        self.hand = self.moveit2_gripper

    def _make_arm_client(self, move_group_namespace: str):
        return MoveIt2(
            node=self,
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name=self.base_frame,
            end_effector_name=self.ee_frame,
            group_name=self.arm_group_name,
            callback_group=self.callback_group,
            move_group_namespace=move_group_namespace,
        )

    def _configure_arm(self, arm):
        arm.pipeline_id = self.planning_pipeline_id
        arm.planner_id = self.planner_id
        arm.max_step_size = self.max_step_size
        arm.max_velocity = self.arm_max_velocity
        arm.max_acceleration = self.arm_max_acceleration
        arm.allowed_planning_time = self.allowed_planning_time
        arm.position_tolerance = self.position_tolerance
        arm.orientation_tolerance = self.orientation_tolerance
        arm.allowed_start_tolerance = self.allowed_start_tolerance

    # ===================== 回调：只缓存目标 =====================
    def target_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 3:
            return
        x, y, yaw = float(msg.data[0]), float(msg.data[1]), float(msg.data[2])
        with self._lock:
            self._latest_target = (x, y, yaw)


     # ===================== 回调：点击触发一次 =====================
    def trigger_callback(self, _msg: Empty):
        with self._lock:
            if self.is_executing:
                self.get_logger().warn("⚠️ 正在执行中，忽略本次点击触发")
                return
            if self._latest_target is None:
                self.get_logger().warn("⚠️ 还没有收到 /target_point 目标，无法执行")
                return

            # 标记执行中
            self.is_executing = True
            target = self._latest_target

        self.get_logger().info("🖱️ 收到点击触发，开始执行一次抓取任务")
        # 用线程执行，避免阻塞订阅回调 & 避免多线程并发重入
        self._worker_thread = threading.Thread(
            target=self._execute_pick_once,
            args=(target,),
            daemon=True
        )
        self._worker_thread.start()

    def make_pose(self, x, y, z, roll, pitch, yaw):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = float(z)
        
        # 欧拉角转四元数
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        quat = r.as_quat()
        p.orientation.x = float(quat[0])
        p.orientation.y = float(quat[1])
        p.orientation.z = float(quat[2])
        p.orientation.w = float(quat[3])
        return p
    
    
    def pose_to_pose_stamped(self, pose):
        """将Pose转换为PoseStamped"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped
    
    def move_to(self, target_pose, cartesian=False, action_name="移动", max_velocity=None, max_acceleration=None):
        target_pose = self.pose_to_pose_stamped(target_pose)
        if not self.motion.wait_client_ready("fairino", self.move_group_ready_timeout_sec):
            return False
        return self.motion.move_to_pose(
            target_pose,
            planning_client="fairino",
            cartesian=cartesian,
            action_name=action_name,
            max_velocity=self.arm_max_velocity if max_velocity is None else max_velocity,
            max_acceleration=self.arm_max_acceleration if max_acceleration is None else max_acceleration,
            max_step_size=self.max_step_size,
            allowed_planning_time=self.allowed_planning_time,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            allowed_start_tolerance=self.allowed_start_tolerance,
            timeout_sec=180.0,
            joint_constraint=self.j2_constraint,
        )

    
    def gripper_action(self, open_gripper=True):
        """控制夹爪张开或闭合"""
        action = "张开夹爪" if open_gripper else "闭合夹爪"
        return self.motion.control_gripper(
            open_gripper=open_gripper,
            action_name=action,
            timeout_sec=90.0,
        )

    def move_to_home(self):
        """返回 home 位置"""
        if not self.motion.wait_client_ready("fairino", self.move_group_ready_timeout_sec):
            return False
        return self.motion.move_to_joints(
            self.home_joints,
            action_name="返回 home 位置",
            planning_client="fairino",
            timeout_sec=180.0,
        )

    def _execute_pick_once(self, target):
        try:
            x, y, yaw = target
            self.get_logger().info(f"🎯 执行目标: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")

            grasping_roll = 0.0
            grasping_pitch = -180.0
            grasping_yaw = (yaw + self.init_angle) * 180.0 / 3.1415926

            poses = {
                "height": self.make_pose(x, y, self.height, grasping_roll, grasping_pitch, grasping_yaw),
                "pick_height": self.make_pose(x, y, self.pick_height, grasping_roll, grasping_pitch, grasping_yaw),
                "carrying_height": self.make_pose(x, y, self.carrying_height, grasping_roll, grasping_pitch, grasping_yaw),
                "case_above": self.make_pose(self.place_x, self.place_y, self.carrying_height, grasping_roll, grasping_pitch, grasping_yaw),
                "case_release": self.make_pose(self.place_x, self.place_y, self.place_release_height, grasping_roll, grasping_pitch, grasping_yaw),
            }

            steps = (
                ("张开夹爪", lambda: self.gripper_action(True)),
                ("返回 home 位置", self.move_to_home),
                ("height", lambda: self.move_to(poses["height"], action_name="height", max_velocity=0.25, max_acceleration=0.25)),
                ("pick_height", lambda: self.move_to(poses["pick_height"], cartesian=True, action_name="pick_height", max_velocity=0.02, max_acceleration=0.02)),
                ("闭合夹爪", lambda: self.gripper_action(False)),
                ("carrying_height", lambda: self.move_to(poses["carrying_height"], cartesian=True, action_name="carrying_height", max_velocity=0.2, max_acceleration=0.2)),
                ("Case 上方 (0.200, 0.250)", lambda: self.move_to(poses["case_above"], action_name="case_above", max_velocity=0.25, max_acceleration=0.25)),
                ("Case 内低位释放", lambda: self.move_to(poses["case_release"], cartesian=True, action_name="case_release", max_velocity=0.02, max_acceleration=0.02)),
                ("张开夹爪", lambda: self.gripper_action(True)),
                ("离开 Case", lambda: self.move_to(poses["case_above"], cartesian=True, action_name="case_retreat", max_velocity=0.2, max_acceleration=0.2)),
                ("返回 home 位置", self.move_to_home),
                ("闭合夹爪", lambda: self.gripper_action(False)),
            )
            for action_name, run_step in steps:
                if not run_step():
                    self.get_logger().error(f"✗ 本次抓取任务中止：{action_name} 失败")
                    return

            self.get_logger().info("✅ 本次抓取任务结束，等待下一次点击触发")

        finally:
            with self._lock:
                self.is_executing = False

def main():
    rclpy.init(args=None)
    node = Controller()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()  # ← 直接spin，不用单独线程
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
