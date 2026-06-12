#!/usr/bin/env python3
import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String,Float64MultiArray, Empty
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np

class Controller(Node):
    def __init__(self):
        super().__init__("yolo_pick_node")

        # ====== 状态变量 ======
        self._lock = threading.Lock()
        self._latest_target = None   # (x, y, yaw)
        self.is_executing = False
        self._worker_thread = None

        # ========= Keepout 参数（你主要改这里）=========
        self.KEEP_OUT_ID = "z_keepout"
        self.KEEP_OUT_FRAME = "base_link"
        self.Z_MIN = 0.07        # 你希望“最低安全高度”（米），例如 0.08 / 0.10 / 0.12
        self.KEEP_OUT_THICKNESS = 0.06  # 禁入区厚度（米），0.1~0.3 常用
        self.KEEP_OUT_XY_SIZE = 0.8     # 覆盖范围（米），足够大覆盖工作区即可
        self.keepout_enabled = False

        # 用 TRANSIENT_LOCAL，让 move_group 后启动也能收到
        latched_qos = QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE,)

        self.moveit_callback_group = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(Float64MultiArray,"/target_point",self.target_callback,10,)
        self.subscription_trigger = self.create_subscription(Empty,"/pick_trigger",self.trigger_callback,10,)
        self.collision_obj_pub = self.create_publisher(CollisionObject, "/collision_object", latched_qos)
        self.planning_scene_pub = self.create_publisher(PlanningScene, "/planning_scene", latched_qos)

        self.arm = MoveIt2(
            node=self,
            # joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            base_link_name="base_link",
            end_effector_name="grasp_frame",
            # group_name="dummy2_arm",
            group_name="robot_arm",
            callback_group=self.moveit_callback_group,
            use_move_group_action=True,
        )

        self.hand = MoveIt2(
            node=self,
            # joint_names=hand_joint_names,
            # joint_names=["figer1", "figer2"],
            joint_names=["finger1_joint", "finger2_joint"],
            base_link_name="base_link",
            end_effector_name="grasp_frame",
            group_name="hand",
            callback_group=self.moveit_callback_group, 
            use_move_group_action=True,
        )

        self.setup_params()
        # self.setup_poses()  

        self.arm.planner_id = "RRTConnect"
        # self.arm.planner_id = "RRTstar"
        self.arm.max_step_size = 0.05
        self.arm.max_velocity = 1.0
        self.arm.max_acceleration = 1.0
        self.arm.allowed_planning_time = 15.0
        self.arm.position_tolerance = 2
        self.arm.orientation_tolerance = 2

        self.get_logger().info("✅ Controller initialized")
        self.get_logger().info("等待：/target_point 更新目标，/pick_trigger 点击触发一次抓取")

    def setup_params(self):
         # 参数配置
        self.height = 0.12
        self.pick_height = 0.02
        self.carrying_height = 0.15
        self.init_angle = 1.5719
        # self.init_angle = -1.5888
        # self.home_joints = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]
        self.home_joints = [-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0]

        # ---------- 多次规划参数 ----------
        self.NUM_CANDIDATE_PLANS = 5

        # ---------- 代价函数权重（越大越抑制腕部乱转） ----------
        self.WRIST_WEIGHT = 50.0

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
    

     # ------------------------- Keepout：地面禁入区（阶段性启用/移除） -------------------------
    def _make_keepout_collision_object(self, z_min: float) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self.KEEP_OUT_FRAME
        co.id = self.KEEP_OUT_ID

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(self.KEEP_OUT_XY_SIZE), float(self.KEEP_OUT_XY_SIZE), float(self.KEEP_OUT_THICKNESS)]

        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = 0.0
        pose.position.y = 0.0
        # 顶面位于 z_min => 中心 z = z_min - thickness/2
        pose.position.z = float(z_min - self.KEEP_OUT_THICKNESS / 2.0)

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        return co

    def enable_keepout(self, z_min: float):
        """启用禁入区：z < z_min 不可进入（用于保护 forearm/elbow 不下探）"""
        co = self._make_keepout_collision_object(z_min)

        # 1) /collision_object（部分配置可用）
        self.collision_obj_pub.publish(co)

        # 2) /planning_scene（MoveIt2最稳）
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.planning_scene_pub.publish(ps)

        self.keepout_enabled = True
        self.get_logger().info(f"[KEEP_OUT] ENABLED: top_z={z_min:.3f} m")

    def disable_keepout(self):
        """移除禁入区：允许指尖/工具下探用于抓取"""
        co = CollisionObject()
        co.header.frame_id = self.KEEP_OUT_FRAME
        co.id = self.KEEP_OUT_ID
        co.operation = CollisionObject.REMOVE

        self.collision_obj_pub.publish(co)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.planning_scene_pub.publish(ps)

        self.keepout_enabled = False
        self.get_logger().info("[KEEP_OUT] DISABLED")

    def select_best_path(self, paths):
        """选择路径代价最小的路径（最短路径）"""
        best_path = None
        min_distance = float('inf')

        for path in paths:
            # 获取路径的所有点
            trajectory = path
            
            # 计算路径的总长度
            path_length = self.calculate_path_length(trajectory)

            # 计算腕部的路径长度，惩罚大角度的腕部旋转
            wrist_len = self.calculate_wrist_path_length(trajectory)

            # 综合考虑路径长度和腕部旋转的惩罚
            total_cost = path_length + self.WRIST_WEIGHT * wrist_len  # WRIST_WEIGHT 是腕部旋转的惩罚因子

            if total_cost < min_distance:
                min_distance = total_cost
                best_path = path

        return best_path

    def calculate_wrist_path_length(self, trajectory):
        """计算腕部的路径长度（J3,J4, J5）"""
        points = trajectory.points

        total_distance = 0.0
        for i in range(1, len(points)):
            wrist1 = np.array([points[i-1].positions[2],points[i-1].positions[3], points[i-1].positions[4]])  # j3,j4, j5翻转代价函数
            wrist2 = np.array([points[i-1].positions[2],points[i].positions[3], points[i].positions[4]])  # j3,j4, j5翻转代价函数
            distance = np.linalg.norm(wrist2 - wrist1)
            total_distance += distance

        return total_distance
    
    def calculate_path_length(self, trajectory):
        """计算路径的长度"""
        points = trajectory.points  # 获取路径点的列表

        total_distance = 0.0
        for i in range(1, len(points)):
            p1 = np.array([points[i-1].positions])  
            p2 = np.array([points[i].positions])  
            distance = np.linalg.norm(p2 - p1)  # 计算欧几里得距离
            total_distance += distance

        return total_distance


    def move_to(self, target_pose, cartesian=False, action_name="移动"):

        target_pose = self.pose_to_pose_stamped(target_pose)

        try:

            ##########代价路径##########
            # 记录多个路径规划结果
            paths = []
            for _ in range(self.NUM_CANDIDATE_PLANS):  # 尝试多次路径规划
                try:
                    self.arm.set_path_joint_constraint(joint_positions=[-1.5708],joint_names=["j2"],tolerance=1.5708,weight=1.0)
                    # 调用 MoveIt2 执行路径规划
                    path = self.arm.plan(target_pose, cartesian=cartesian)
                    if path:
                        paths.append(path)
                except Exception as e:
                    self.get_logger().error(f"路径规划失败: {e}")
                    continue
            
            if not paths:
                self.get_logger().error("没有成功生成路径")
                return False
            # 从多个路径中选择最短路径
            best_path = self.select_best_path(paths)
            
            # 执行最短路径
            self.arm.execute(best_path)
            ##########代价路径##########

            # self.arm.set_path_joint_constraint(joint_positions=[-1.5708],joint_names=["j2"],tolerance=1.5708,weight=1.0)
            # self.arm.move_to_pose(pose=target_pose, cartesian=cartesian)

            ok = self.arm.wait_until_executed()
            if not ok:
                self.get_logger().error(f"✗ {action_name}失败：执行未成功（action aborted/failed）")
                return False
            
            self.get_logger().info(f'✓ {action_name}完成')
            time.sleep(1.0)
            return True

        except Exception as e:
            self.get_logger().error(f'✗ {action_name}失败: {e}')
            return False

    
    def gripper_action(self, open_gripper=True):
        """控制夹爪张开或闭合"""
        action = "张开夹爪" if open_gripper else "闭合夹爪"
        self.get_logger().info(f'正在执行: {action}')
        
        # positions = [0.028, -0.028] if open_gripper else [0.0, 0.0]
        positions = [0.0305, -0.0305] if open_gripper else [0.0, 0.0]
        try:
            self.hand.move_to_configuration(positions)
            ok = self.arm.wait_until_executed()
            if not ok:
                self.get_logger().error(f"✗ {action}失败：执行未成功（action aborted/failed）")
                return False
            self.get_logger().info(f'✓ {action}完成')
            time.sleep(1.0)
            return True
        except Exception as e:
            time.sleep(1.0)
            return True

    def move_to_home(self):
        """返回 home 位置"""
        self.get_logger().info("正在返回 home 位置")
        try:
            self.arm.move_to_configuration(self.home_joints)  # Move to home joints
            ok = self.arm.wait_until_executed()
            if not ok:
                self.get_logger().error('✓ 返回HOME失败')
                return False
            self.get_logger().info('✓ 返回HOME完成')
            time.sleep(1.0)
            return True
        
        except Exception as e:
            self.get_logger().error(f'返回HOME失败: {e}')
            return False

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
                "place": self.make_pose(0.3, 0.2, self.carrying_height, grasping_roll, grasping_pitch, grasping_yaw),
            }

            # ====== 抓取序列======
            self.gripper_action(True)
            self.move_to_home()
            self.enable_keepout(self.Z_MIN)
            self.move_to(poses["height"], cartesian=False, action_name="height")
            self.disable_keepout()
            self.move_to(poses["pick_height"], cartesian=True, action_name="pick_height")
            self.gripper_action(False)
            self.move_to(poses["carrying_height"], cartesian=True, action_name="carrying_height")
            self.move_to(poses["place"], cartesian=False, action_name="place")
            self.gripper_action(True)
            self.move_to_home()
            self.gripper_action(False)

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()



