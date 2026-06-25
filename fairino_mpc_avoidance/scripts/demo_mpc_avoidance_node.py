#!/usr/bin/env python3
"""
MPC 动态避障演示节点
流程：
  1. 初始位置 → box_above: birrt*/rrt* 规划 + MoveIt2 直接执行
  2. box_above → case_place: birrt*/rrt* 规划 + MPC 实时跟踪避障
"""
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint,
    BoundingVolume, RobotState, MoveItErrorCodes
)
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from shape_msgs.msg import SolidPrimitive

from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import threading
import math
import xml.etree.ElementTree as ET


class MPCAvoidanceDemoNode(Node):
    def __init__(self):
        super().__init__('mpc_avoidance_demo_node')

        self.callback_group = ReentrantCallbackGroup()

        # 机器人配置（全参数化，禁止型号硬编码）
        self.declare_parameter("robot_profile", "s622_gripper")
        self.declare_parameter("group_name", "robot_arm")
        self.declare_parameter("ee_link", "grasp_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("joint_names", ["j1", "j2", "j3", "j4", "j5", "j6"])
        self.declare_parameter("controller_topic", "/robot_arm_controller/joint_trajectory")
        self.declare_parameter("planning_client", "fairino")
        self.declare_parameter("move_group_namespace", "")
        self.declare_parameter("planner_id", "birrt*")

        self.robot_profile = str(self.get_parameter("robot_profile").value).strip()
        self.group_name = str(self.get_parameter("group_name").value).strip()
        self.ee_link = str(self.get_parameter("ee_link").value).strip()
        self.base_frame = str(self.get_parameter("base_frame").value).strip()
        self.joint_names = [str(x) for x in self.get_parameter("joint_names").value]
        self.controller_topic = str(self.get_parameter("controller_topic").value).strip()
        self.planner_id = str(self.get_parameter("planner_id").value).strip() or "birrt*"

        if not self.group_name:
            raise RuntimeError("参数 group_name 不能为空。")
        if not self.ee_link:
            raise RuntimeError("参数 ee_link 不能为空。")
        if not self.base_frame:
            raise RuntimeError("参数 base_frame 不能为空。")
        if not self.joint_names:
            raise RuntimeError("参数 joint_names 不能为空。")

        self._assert_group_exists_in_srdf()
        (
            self.planning_client,
            self.move_group_namespace,
            self.namespace_override_used,
        ) = self._resolve_planning_client()

        # 笛卡尔目标
        self.targets = {
            "box_above": {
                "position": [0.420, 0.33, 0.2],
                "orientation_euler": [0.0, math.pi, 0.0],
                "description": "box上方"
            },
            "case_place": {
                "position": [0.35, -0.33, 0.12],
                "orientation_euler": [0.0, math.pi, 0.0],
                "description": "放置位置"
            }
        }

        # 状态
        self.current_joints = None
        self.mpc_reached_goal = False
        self.mpc_active = False
        self.replan_requested = False
        self.current_mpc_target = None
        self.replan_count = 0
        self._controller_topic_warned = False

        # 订阅
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.mpc_status_sub = self.create_subscription(
            String, '/mpc_status', self.mpc_status_cb, status_qos,
            callback_group=self.callback_group)

        # 发布
        self.ref_traj_pub = self.create_publisher(
            JointTrajectory, '/planned_trajectory', 10)
        self.mpc_cmd_pub = self.create_publisher(
            String, '/mpc_command', 10)

        # MoveIt2 规划服务（只规划）
        self.plan_service_name = self._resolve_move_group_name("plan_kinematic_path")
        self.execute_action_name = self._resolve_move_group_name("execute_trajectory")

        self.plan_client = self.create_client(
            GetMotionPlan, self.plan_service_name,
            callback_group=self.callback_group)

        # MoveIt2 执行 Action（直接执行，用于第一段）
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, self.execute_action_name,
            callback_group=self.callback_group)

        self.create_timer(2.0, self._check_controller_topic_ready)

        self.get_logger().info('=' * 60)
        self.get_logger().info('MPC 动态避障演示节点启动')
        self.get_logger().info(f'  阶段1: {self.planner_id} + 直接执行 (无动态障碍物)')
        self.get_logger().info(f'  阶段2: {self.planner_id} + MPC 实时避障 (动态障碍物)')
        self.get_logger().info(f'  robot_profile: {self.robot_profile}')
        self.get_logger().info(
            f'  规划客户端: {self.planning_client}, 命名空间覆盖: {self.namespace_override_used}')
        self.get_logger().info(
            f'  MoveGroup namespace: "{self.move_group_namespace or "/"}"')
        self.get_logger().info(
            f'  绑定服务: {self.plan_service_name}, Action: {self.execute_action_name}')
        self.get_logger().info(
            f'  group={self.group_name}, ee_link={self.ee_link}, base_frame={self.base_frame}')
        self.get_logger().info(f'  planner_id={self.planner_id}')
        self.get_logger().info(
            f'  joints={self.joint_names}, controller_topic={self.controller_topic}')
        self.get_logger().info('=' * 60)

    def _normalize_move_group_namespace(self, namespace: str) -> str:
        ns = (namespace or "").strip()
        if ns in ("", "/"):
            return ""
        if not ns.startswith("/"):
            ns = "/" + ns
        return ns.rstrip("/")

    def _resolve_move_group_name(self, name: str) -> str:
        if not self.move_group_namespace:
            return f"/{name}"
        return f"{self.move_group_namespace}/{name}"

    def _resolve_planning_client(self):
        planning_client = str(self.get_parameter("planning_client").value).strip().lower()
        namespace_override = self._normalize_move_group_namespace(
            str(self.get_parameter("move_group_namespace").value)
        )
        client_to_namespace = {
            "fairino": "/move_group_fairino",
            "kdl": "/move_group_kdl",
        }
        if namespace_override:
            return planning_client if planning_client else "override", namespace_override, True
        if planning_client not in client_to_namespace:
            self.get_logger().warn(
                f"非法 planning_client='{planning_client}'，回退到 fairino。"
            )
            planning_client = "fairino"
        return planning_client, client_to_namespace[planning_client], False

    def _assert_group_exists_in_srdf(self):
        srdf_text = ""
        if self.has_parameter("robot_description_semantic"):
            srdf_text = str(self.get_parameter("robot_description_semantic").value or "").strip()
        if not srdf_text:
            self.get_logger().warn("未提供 robot_description_semantic，跳过 SRDF 组名校验。")
            return
        try:
            root = ET.fromstring(srdf_text)
        except ET.ParseError as exc:
            raise RuntimeError(f"robot_description_semantic 解析失败: {exc}") from exc
        groups = sorted({group.attrib.get("name", "").strip() for group in root.findall("group") if group.attrib.get("name")})
        if self.group_name not in groups:
            raise RuntimeError(
                f"group_name='{self.group_name}' 不存在于 SRDF。可选组: {groups}"
            )

    def _check_controller_topic_ready(self):
        if self._controller_topic_warned:
            return
        topic_names = {name for name, _types in self.get_topic_names_and_types()}
        if self.controller_topic not in topic_names:
            self.get_logger().warn(
                f"控制器话题暂未出现: {self.controller_topic} (可能是启动时序问题)"
            )
            self._controller_topic_warned = True

    def wait_for_future(self, future, timeout_sec=None):
        start = time.time()
        while rclpy.ok() and not future.done():
            if timeout_sec is not None and time.time() - start > timeout_sec:
                return False
            time.sleep(0.02)
        return future.done()

    # ================================================================
    #  回调
    # ================================================================
    def joint_state_cb(self, msg: JointState):
        positions = []
        for name in self.joint_names:
            if name in msg.name:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
        if len(positions) == 6:
            self.current_joints = positions

    def mpc_status_cb(self, msg: String):
        if msg.data == "REACHED":
            self.mpc_reached_goal = True
            self.mpc_active = False
            self.get_logger().info('  ✓ MPC: 到达目标')
        elif msg.data == "TRACKING":
            self.mpc_active = True
        elif msg.data == "REPLAN_REQUIRED":
            self.replan_requested = True
            self.get_logger().warn('  MPC: 请求重规划')

    # ================================================================
    #  工具函数
    # ================================================================
    def euler_to_quaternion(self, euler_zyx):
        r = R.from_euler('ZYX', euler_zyx)
        return r.as_quat()

    def create_pose_from_target(self, target_dict):
        pose = Pose()
        pos = target_dict["position"]
        euler = target_dict["orientation_euler"]
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        quat = self.euler_to_quaternion(euler)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose

    def build_plan_request(self, target_name):
        """构建规划请求"""
        target = self.targets[target_name]
        target_pose = self.create_pose_from_target(target)

        mp_request = MotionPlanRequest()
        mp_request.group_name = self.group_name
        mp_request.num_planning_attempts = 5
        mp_request.allowed_planning_time = 15.0
        mp_request.pipeline_id = "fairino"
        mp_request.planner_id = self.planner_id

        if self.current_joints is not None:
            start_state = RobotState()
            start_state.joint_state.name = self.joint_names
            start_state.joint_state.position = self.current_joints
            mp_request.start_state = start_state

        goal_constraints = Constraints()

        # 位置约束
        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.base_frame
        pos_c.link_name = self.ee_link

        bv = BoundingVolume()
        solid = SolidPrimitive()
        solid.type = SolidPrimitive.SPHERE
        solid.dimensions = [0.005]
        bv.primitives.append(solid)
        bv_pose = Pose()
        bv_pose.position = target_pose.position
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pos_c.constraint_region = bv
        pos_c.weight = 1.0
        goal_constraints.position_constraints.append(pos_c)

        # 姿态约束
        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.base_frame
        ori_c.link_name = self.ee_link
        ori_c.orientation = target_pose.orientation
        ori_c.absolute_x_axis_tolerance = 0.05
        ori_c.absolute_y_axis_tolerance = 0.05
        ori_c.absolute_z_axis_tolerance = 0.05
        ori_c.weight = 1.0
        goal_constraints.orientation_constraints.append(ori_c)

        mp_request.goal_constraints.append(goal_constraints)
        return mp_request

    # ================================================================
    #  方式 A: birrt*/rrt* 规划 + MoveIt2 直接执行（无动态避障）
    # ================================================================
    def move_direct(self, target_name, timeout=30.0):
        """
        直接规划并执行（用于无动态障碍物的路段）
        使用 MoveIt2 的 ExecuteTrajectory action
        """
        target = self.targets[target_name]
        desc = target["description"]

        self.get_logger().info(f'\n{"─" * 50}')
        self.get_logger().info(f'  [直接执行] 目标: {desc}')
        self.get_logger().info(f'  xyz={target["position"]}')
        self.get_logger().info(f'{"─" * 50}')

        # 1. 规划
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('规划服务不可用')
            return False

        request = GetMotionPlan.Request()
        request.motion_plan_request = self.build_plan_request(target_name)

        self.get_logger().info(f'  调用 {self.planner_id} 规划...')
        future = self.plan_client.call_async(request)
        if not self.wait_for_future(future):
            self.get_logger().error('  规划请求未返回结果')
            return False

        if future.result() is None:
            self.get_logger().error('  规划请求未返回结果')
            return False

        response = future.result()
        if response.motion_plan_response.error_code.val != 1:
            self.get_logger().error(
                f'  规划失败: {response.motion_plan_response.error_code.val}')
            return False

        trajectory = response.motion_plan_response.trajectory
        n_pts = len(trajectory.joint_trajectory.points)
        self.get_logger().info(f'  ✓ 规划成功: {n_pts} 点')

        # 2. 执行
        if not self.execute_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error('  执行服务不可用')
            return False

        self.get_logger().info('  执行轨迹...')
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        send_goal_future = self.execute_client.send_goal_async(goal_msg)
        if not self.wait_for_future(send_goal_future, 5.0):
            self.get_logger().error('  执行请求未返回结果')
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('  执行请求被拒绝')
            return False

        # 等待执行完成
        exec_start = time.time()
        result_future = goal_handle.get_result_async()
        if not self.wait_for_future(result_future, timeout):
            self.get_logger().error('  执行超时')
            return False

        if result_future.result() is None:
            self.get_logger().error('  执行超时')
            return False

        result = result_future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f'  ✓ [{desc}] 执行完成')
            return True

        if result.error_code.val == MoveItErrorCodes.PREEMPTED:
            joint_traj = trajectory.joint_trajectory
            if joint_traj.points:
                target_by_name = dict(zip(joint_traj.joint_names, joint_traj.points[-1].positions))
                target_joints = [target_by_name.get(name) for name in self.joint_names]
                deadline = exec_start + timeout
                while rclpy.ok() and time.time() < deadline and all(v is not None for v in target_joints):
                    if self.current_joints is not None:
                        max_err = max(abs(a - b) for a, b in zip(self.current_joints, target_joints))
                        if max_err <= math.radians(3.0):
                            self.get_logger().warn(
                                f'  [{desc}] ExecuteTrajectory 返回 PREEMPTED(-7)，'
                                f'但关节已到目标附近(max_err={math.degrees(max_err):.2f}deg)，按完成处理')
                            return True
                    time.sleep(0.05)

        self.get_logger().error(
            f'  ✗ [{desc}] 执行失败: {result.error_code.val}')
        return False

    # ================================================================
    #  方式 B: birrt*/rrt* 规划 + MPC 实时跟踪避障
    # ================================================================
    def move_with_mpc(self, target_name, timeout=300.0):
        """
        规划参考路径，交给 MPC 节点实时跟踪避障
        """
        target = self.targets[target_name]
        desc = target["description"]

        self.get_logger().info(f'\n{"─" * 50}')
        self.get_logger().info(f'  [MPC避障] 目标: {desc}')
        self.get_logger().info(f'  xyz={target["position"]}')
        self.get_logger().info(f'{"─" * 50}')

        # 1. 规划
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('规划服务不可用')
            return False

        request = GetMotionPlan.Request()
        request.motion_plan_request = self.build_plan_request(target_name)

        self.get_logger().info(f'  调用 {self.planner_id} 规划...')
        future = self.plan_client.call_async(request)
        if not self.wait_for_future(future):
            self.get_logger().error('  规划请求未返回结果')
            return False

        if future.result() is None:
            self.get_logger().error('  规划请求未返回结果')
            return False

        response = future.result()
        if response.motion_plan_response.error_code.val != 1:
            self.get_logger().error(
                f'  规划失败: {response.motion_plan_response.error_code.val}')
            return False

        trajectory = response.motion_plan_response.trajectory.joint_trajectory
        n_pts = len(trajectory.points)
        self.get_logger().info(f'  ✓ 规划成功: {n_pts} 点')

        # 2. 发布给 MPC 节点
        self.mpc_reached_goal = False
        self.mpc_active = True
        self.replan_requested = False
        self.current_mpc_target = target_name
        self.replan_count = 0
        self.ref_traj_pub.publish(trajectory)

        cmd = String()
        cmd.data = "START"
        self.mpc_cmd_pub.publish(cmd)
        self.get_logger().info(f'  参考轨迹已发布给 MPC ({n_pts} 点)')

        # 3. 等待 MPC 完成
        self.get_logger().info('  MPC 实时跟踪避障中...')
        start_time = time.time()
        last_log = start_time

        while not self.mpc_reached_goal:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.get_logger().error(
                    f'  MPC 超时: {elapsed:.1f}s, 重规划次数={self.replan_count}')
                self.stop_mpc()
                return False

            if self.replan_requested:
                self.replan_requested = False
                if not self.replan_mpc_reference(target_name):
                    self.get_logger().error('  MPC 重规划失败')
                    self.stop_mpc()
                    return False
                last_log = time.time()

            if time.time() - last_log > 5.0:
                self.get_logger().info(f'  ... 跟踪中 ({elapsed:.1f}s)')
                last_log = time.time()
            time.sleep(0.05)

        self.get_logger().info(f'  ✓ [{desc}] MPC 避障完成')
        return True

    def replan_mpc_reference(self, target_name):
        """收到 REPLAN_REQUIRED 后，从当前关节状态到同一目标重新规划并发给 MPC。"""
        if self.current_joints is None:
            self.get_logger().error('  无当前关节状态，无法重规划')
            return False

        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('  规划服务不可用，无法重规划')
            return False

        self.replan_count += 1
        target = self.targets[target_name]
        self.get_logger().warn(
            f'  [MPC重规划 #{self.replan_count}] 目标: {target["description"]}')

        request = GetMotionPlan.Request()
        request.motion_plan_request = self.build_plan_request(target_name)

        future = self.plan_client.call_async(request)
        if not self.wait_for_future(future):
            self.get_logger().error('  重规划请求未返回结果')
            return False

        if future.result() is None:
            self.get_logger().error('  重规划请求未返回结果')
            return False

        response = future.result()
        if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'  重规划失败: {response.motion_plan_response.error_code.val}')
            return False

        trajectory = response.motion_plan_response.trajectory.joint_trajectory
        n_pts = len(trajectory.points)
        if n_pts < 2:
            self.get_logger().error('  重规划轨迹点数不足')
            return False

        self.ref_traj_pub.publish(trajectory)
        cmd = String()
        cmd.data = "START"
        self.mpc_cmd_pub.publish(cmd)
        self.mpc_active = True
        self.get_logger().info(f'  ✓ 新参考轨迹已发布给 MPC ({n_pts} 点)')
        return True

    def stop_mpc(self):
        if not rclpy.ok():
            return
        cmd = String()
        cmd.data = "STOP"
        self.mpc_cmd_pub.publish(cmd)
        self.mpc_active = False

    # ================================================================
    #  演示主流程
    # ================================================================
    def run_demo(self):
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('   MPC 动态避障演示')
        self.get_logger().info('   阶段1: 直接执行 | 阶段2: MPC 避障')
        self.get_logger().info('=' * 60)

        # 等待关节状态
        self.get_logger().info('等待关节状态...')
        t0 = time.time()
        while self.current_joints is None:
            if time.time() - t0 > 30.0:
                self.get_logger().error('超时')
                return
            time.sleep(0.5)

        self.get_logger().info(
            f'当前关节: [{", ".join(f"{j:.3f}" for j in self.current_joints)}]')
        time.sleep(2.0)

        # ═══════════════════════════════════════════════════
        # 阶段 1: 初始位置 → box_above（直接执行，无动态障碍物）
        # ═══════════════════════════════════════════════════
        self.get_logger().info('\n' + '═' * 60)
        self.get_logger().info('  阶段 1: 直接执行到 box 上方')
        self.get_logger().info('═' * 60)

        if not self.move_direct("box_above", timeout=30.0):
            self.get_logger().error('阶段 1 失败')
            return

        time.sleep(2.0)

        # ═══════════════════════════════════════════════════
        # 阶段 2: box_above → case_place（MPC 动态避障）
        # ═══════════════════════════════════════════════════
        self.get_logger().info('\n' + '═' * 60)
        self.get_logger().info('  阶段 2: MPC 动态避障到放置位置')
        self.get_logger().info('═' * 60)

        if not self.move_with_mpc("case_place", timeout=300.0):
            self.get_logger().error('阶段 2 失败')
            return

        self.get_logger().info('\n' + '═' * 60)
        self.get_logger().info('   ✓ 演示完成！')
        self.get_logger().info('═' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = MPCAvoidanceDemoNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def run_task():
        time.sleep(3.0)
        node.run_demo()

    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_mpc()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
