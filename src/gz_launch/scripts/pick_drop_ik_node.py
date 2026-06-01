#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R
import time
import math
from enum import Enum
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np


class TaskState(Enum):
    IDLE = "idle"
    MOVING_TO_BOX_ABOVE = "moving_to_box_above"
    MOVING_TO_BOX = "moving_to_box"
    GRASPING = "grasping"
    LIFTING_BOX = "lifting_box"
    MOVING_TO_INTERMEDIATE = "moving_to_intermediate" 
    MOVING_TO_CASE = "moving_to_case"
    RELEASING = "releasing"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ERROR = "error"


class PenBoxGraspingNode(Node):
    def __init__(self):
        super().__init__('pick_drop_ik')
        
        self.callback_group = ReentrantCallbackGroup()
        time.sleep(2.0)
        
        self.setup_moveit()
        self.setup_params()
        self.setup_poses()  

        # ========= Keepout 参数（你主要改这里）=========
        self.KEEP_OUT_ID = "z_keepout"
        self.KEEP_OUT_FRAME = "base_link"
        self.Z_MIN = 0.07        # 你希望“最低安全高度”（米），例如 0.08 / 0.10 / 0.12
        self.KEEP_OUT_THICKNESS = 0.06  # 禁入区厚度（米），0.1~0.3 常用
        self.KEEP_OUT_XY_SIZE = 0.8     # 覆盖范围（米），足够大覆盖工作区即可
        self.keepout_enabled = False

        # 用 TRANSIENT_LOCAL，让 move_group 后启动也能收到
        latched_qos = QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE,)

        self.current_state = TaskState.IDLE
        self.state_publisher = self.create_publisher(String, '/task_state', 10)
        self.collision_obj_pub = self.create_publisher(CollisionObject, "/collision_object", latched_qos)
        self.planning_scene_pub = self.create_publisher(PlanningScene, "/planning_scene", latched_qos)

        # self.enable_keepout(self.Z_MIN)

        self.create_timer(4.0, self.control_loop)
        
        self.get_logger().info('抓取节点启动完成')

    def setup_moveit(self):
        """初始化MoveIt接口"""
        try:
            self.moveit2_arm = MoveIt2(
                node=self, 
                joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                base_link_name="base_link", 
                end_effector_name="grasp_frame",
                group_name="robot_arm", 
                # group_name="robot_arm_tip",
                callback_group=self.callback_group,
                use_move_group_action=True
            )
            
            self.moveit2_arm.planner_id = "RRTConnect"
            # self.moveit2_arm.planner_id = "RRTstar"
            self.moveit2_arm.max_step_size = 0.05
            self.moveit2_arm.max_velocity = 1.0
            self.moveit2_arm.max_acceleration = 1.0
            self.moveit2_arm.allowed_planning_time = 15.0
            self.moveit2_arm.position_tolerance = 2
            self.moveit2_arm.orientation_tolerance = 2
            
            self.moveit2_gripper = MoveIt2(
                node=self, 
                joint_names=["finger1_joint", "finger2_joint"],
                # joint_names=["finger1_joint"],
                base_link_name="base_link", 
                end_effector_name="grasp_frame",
                group_name="hand", 
                callback_group=self.callback_group,
                use_move_group_action=True
            )
            
            self.get_logger().info('MoveIt接口初始化成功')
        except Exception as e:
            self.get_logger().error(f'MoveIt初始化失败: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def setup_params(self):
        """设置基础参数"""
        # 目标位置
        # self.BOX_POS = (0.2, 0.0, 0.04)
        # self.BOX_POS = (-0.29, 0.376, 0.04)
        self.BOX_POS = (-0.19, 0.076, 0.04)
        self.CASE_POS = (0.0, 0.50, 0.08)
        
        # 运动参数
        self.grasp_offset = 0.008
        self.safe_height = 0.05

        # ---------- 多次规划参数 ----------
        self.NUM_CANDIDATE_PLANS = 8

        # ---------- 代价函数权重（越大越抑制腕部乱转） ----------
        self.WRIST_WEIGHT =50.0
        

        # self.home_joints = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]
        self.home_joints = [-1.1170, -1.6214, 1.5465, -1.5877, -1.6368, 0.0]
        self.action_delay = 1.0

        self.INTERMEDIATE_POS = ((self.BOX_POS[0]+self.CASE_POS[0])/2.0, (self.BOX_POS[1]+self.CASE_POS[1])/2.0, self.CASE_POS[2]*2.0)  # 自定义中间位置坐标
      
        
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

    def setup_poses(self):
        """集中定义所有任务姿态"""
        # 抓取box的姿态（垂直向下）
        box_roll, box_pitch, box_yaw = 0, -180, 0
        
        # 中间位置姿态（用于前往中间点）
        intermediate_roll, intermediate_pitch, intermediate_yaw = 0, -180, 0

        # 放置case的姿态（标准姿态）
        case_roll, case_pitch, case_yaw = 0, -180, 0
        
        
        
        # 定义所有关键姿态
        self.poses = {

            # 抓取阶段
            'box_above': self.make_pose(self.BOX_POS[0],self.BOX_POS[1],self.BOX_POS[2] + self.safe_height,box_roll, box_pitch, box_yaw),

            'box_grasp': self.make_pose(self.BOX_POS[0],self.BOX_POS[1],self.BOX_POS[2] - self.grasp_offset,box_roll, box_pitch, box_yaw),
            
            'box_lift': self.make_pose(self.BOX_POS[0],self.BOX_POS[1],self.BOX_POS[2] + self.safe_height,box_roll, box_pitch, box_yaw),
            
            # 中间位置阶段
            'intermediate_place': self.make_pose(self.INTERMEDIATE_POS[0],self.INTERMEDIATE_POS[1],self.INTERMEDIATE_POS[2],intermediate_roll, intermediate_pitch, intermediate_yaw),

            # case放置阶段
            'case_place': self.make_pose(self.CASE_POS[0],self.CASE_POS[1],self.CASE_POS[2] + self.safe_height,case_roll, case_pitch, case_yaw),
        }
        

    def pose_to_pose_stamped(self, pose):
        """将Pose转换为PoseStamped"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped

    def move_to_pose(self, target_pose, cartesian=False, action_name="移动"):

        target_pose = self.pose_to_pose_stamped(target_pose)

        try:

            #代价路径#
            # 记录多个路径规划结果
            paths = []
            for _ in range(self.NUM_CANDIDATE_PLANS):  # 尝试多次路径规划
                try:
                    self.moveit2_arm.set_path_joint_constraint(joint_positions=[-1.5708],joint_names=["j2"],tolerance=1.5708,weight=1.0)
                    # 调用 MoveIt2 执行路径规划
                    path = self.moveit2_arm.plan(target_pose, cartesian=cartesian)
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
            self.moveit2_arm.execute(best_path)
            #代价路径#

            # self.moveit2_arm.move_to_pose(pose=target_pose, cartesian=cartesian)
            ok = self.moveit2_arm.wait_until_executed()
            # self.moveit2_arm.clear_path_constraints()
            if not ok:
                self.get_logger().error(f"✗ {action_name}失败：执行未成功（action aborted/failed）")
                return False
            self.get_logger().info(f'✓ {action_name}完成')
            time.sleep(self.action_delay)
            return True

        except Exception as e:
            self.get_logger().error(f'✗ {action_name}失败: {e}')
            return False
        
    def control_gripper(self, open_gripper=True):
        """控制夹爪张开或闭合"""
        action = "张开夹爪" if open_gripper else "闭合夹爪"
        self.get_logger().info(f'正在执行: {action}')
        
        # pos_open = -0.0305
        # pos_close = 0.0  
        # positions = [pos_open] if open_gripper else [pos_close]
        positions = [0.0305, -0.0305] if open_gripper else [0.0, 0.0]
        try:
            self.moveit2_gripper.move_to_configuration(positions)

            ok = self.moveit2_gripper.wait_until_executed()
            if not ok:
                self.get_logger().error(f"✗ {action}失败：执行未成功（action aborted/failed）")
                return False
            
            self.get_logger().info(f'✓ {action}完成')
            time.sleep(self.action_delay)
            return True
        
        except Exception as e:
            # time.sleep(self.action_delay)
            # return True
            self.get_logger().error(f'✗ {action}失败: {e}')
            return False

    def go_home(self):
        """返回home位置"""
        self.get_logger().info('正在执行: 返回HOME')
        try:
            self.moveit2_arm.move_to_configuration(self.home_joints)
            ok = self.moveit2_arm.wait_until_executed()
            if not ok:
                self.get_logger().error('✓ 返回HOME失败')
                return False
            self.get_logger().info('✓ 返回HOME完成')
            time.sleep(self.action_delay)
            return True
        
        except Exception as e:
            self.get_logger().error(f'返回HOME失败: {e}')
            return False

    def control_loop(self):
        """主控制循环"""
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_publisher.publish(state_msg)
        
        try:
            if self.current_state == TaskState.IDLE:
                self.get_logger().info("=== 开始抓取任务 ===")
                if self.go_home():

                    # self.control_gripper(True)
                    self.current_state = TaskState.MOVING_TO_BOX_ABOVE
                else:
                    self.current_state = TaskState.ERROR
            
            # ===== 阶段1.5: 移动到box抓取位置 =====
            elif self.current_state == TaskState.MOVING_TO_BOX_ABOVE:

                # 阶段：移动到抓取上方 -> keepout 应启用
                if not self.keepout_enabled:
                    self.enable_keepout(self.Z_MIN)

                if self.move_to_pose(
                    self.poses['box_above'], 
                    cartesian=False, 
                    # cartesian=True, 
                    action_name="移动到box_above抓取位置",
                    # j2_center=-1.5708, j2_tol=1.5708
                ):
                    self.current_state = TaskState.RETURNING_HOME
                else:
                    self.current_state = TaskState.ERROR

            
            # ===== 阶段2.0: 移动到box抓取位置 =====
            elif self.current_state == TaskState.MOVING_TO_BOX:

                # 阶段：向下接近抓取 -> 必须移除 keepout，避免指尖被挡住
                # if self.keepout_enabled:
                #     self.disable_keepout()

                if self.move_to_pose(
                    self.poses['box_grasp'], 
                    cartesian=True, 
                    action_name="移动到box抓取位置",
                    # j2_center=-1.5708, j2_tol=1.5708
                ):
                    self.current_state = TaskState.RETURNING_HOME
                else:
                    self.current_state = TaskState.ERROR
            
            # ===== 阶段2: 闭合夹爪 =====
            elif self.current_state == TaskState.GRASPING:
                self.control_gripper(False)
                self.current_state = TaskState.LIFTING_BOX
            
            # ===== 阶段3: 提升box =====
            elif self.current_state == TaskState.LIFTING_BOX:
                if self.move_to_pose(
                    self.poses['box_lift'], 
                    # cartesian=False, 
                    cartesian=True, 
                    action_name="提升box",
                    # j2_center=-1.5708, j2_tol=1.5708
                ):
                    self.current_state = TaskState.MOVING_TO_INTERMEDIATE
                else:
                    self.current_state = TaskState.ERROR
            
            # ===== 阶段3.5: 移动到中间位置 =====
            elif self.current_state == TaskState.MOVING_TO_INTERMEDIATE:

                # 阶段：去中间点 -> keepout 启用
                if not self.keepout_enabled:
                    self.enable_keepout(self.Z_MIN)
                # if self.keepout_enabled:
                #     self.disable_keepout()

                if self.move_to_pose(
                    self.poses['intermediate_place'], 
                    cartesian=False, 
                    action_name="移动到中间位置",
                    # j2_center=-1.5708, j2_tol=1.5708
                ):
                    self.current_state = TaskState.MOVING_TO_CASE
                else:
                    self.current_state = TaskState.ERROR

            # ===== 阶段4: 移动到case放置位置 =====
            elif self.current_state == TaskState.MOVING_TO_CASE:

                 # 阶段：去放置区 -> keepout 启用
                if not self.keepout_enabled:
                    self.enable_keepout(self.Z_MIN)

                if self.move_to_pose(
                    self.poses['case_place'], 
                    cartesian=False, 
                    action_name="移动到case放置位置",
                    # j2_center=-1.5708, j2_tol=1.5708
                ):
                    self.current_state = TaskState.RELEASING
                else:
                    self.current_state = TaskState.ERROR
            
            # ===== 阶段5: 释放box =====
            elif self.current_state == TaskState.RELEASING:
                self.control_gripper(True)
                self.current_state = TaskState.RETURNING_HOME
            
            # ===== 阶段6: 返回home =====
            elif self.current_state == TaskState.RETURNING_HOME:
                if self.go_home():
                    self.control_gripper(False)
                    self.current_state = TaskState.COMPLETED
                else:
                    self.current_state = TaskState.ERROR
                    
            # ===== 任务完成 =====
            elif self.current_state == TaskState.COMPLETED:
                self.get_logger().info("=== 任务完成 ===")
                time.sleep(2.0)
                self.current_state = TaskState.IDLE
            
            # ===== 错误恢复 =====
            elif self.current_state == TaskState.ERROR:
                self.get_logger().error("!!! 任务出错，执行恢复 !!!")
                # self.control_gripper(True)
                time.sleep(self.action_delay)
                if self.go_home():
                    time.sleep(1.0)
                    self.current_state = TaskState.IDLE
                else:
                    time.sleep(2.0)
                    
        except Exception as e:
            self.get_logger().error(f'控制循环异常: {e}')
            import traceback
            self.get_logger().error(f'详细错误:\n{traceback.format_exc()}')
            self.current_state = TaskState.ERROR


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PenBoxGraspingNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        node.get_logger().info("=== 抓取节点就绪 ===")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("任务被中断")
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()





