#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from pymoveit2 import MoveIt2  



ARM_JOINT_NAMES = [
    "j1", "j2", "j3", "j4", "j5", "j6"
]

HAND_JOINT_NAMES = [
    "finger1_joint", "finger2_joint"
]

# 机器人基座与末端执行器链接名
BASE_LINK_NAME = "base_link"         
END_EFFECTOR_LINK = "grasp_frame"          

HOME_ARM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]          
OPEN = [0.028, -0.028]
CLOSE = [0.0, 0.0]                               

# 你原脚本中执行的几组关节配置（已按顺序整理）
ARM_CONF_0 = [ 1.536,0.0,0.0,0.0,0.0,0.0]

ARM_CONF_1 = [ 1.536,-0.262,0.524,0.0,0.803,0.0 ]


ARM_CONF_2 = [0.524,-0.262,-0.367,0.0,0.785,0.0]

ARM_CONF_3 = [0.0,-0.785,-0.367,0.017,-0.454,0.0]
###############################################################################

def main():
    rclpy.init()
    logger = get_logger("pymoveit2.pose_goal")
    node = Node("pick_drop")
    logger.info("Node 创建完成")

    arm = MoveIt2(
        node=node,
        joint_names=ARM_JOINT_NAMES,
        base_link_name=BASE_LINK_NAME,
        end_effector_name=END_EFFECTOR_LINK,
        group_name="robot_arm",
        use_move_group_action=True,
    )
    hand = MoveIt2(
        node=node,
        joint_names=HAND_JOINT_NAMES,
        base_link_name=BASE_LINK_NAME,
        end_effector_name=END_EFFECTOR_LINK,   
        group_name="hand",
        use_move_group_action=True,
    )

    logger.info("MoveIt2 接口创建完成")

    # 可选：设置速度/加速度比例（0~1，0 表示使用默认）
    arm.max_velocity = 0.8
    arm.max_acceleration = 0.8
    hand.max_velocity = 0.8
    hand.max_acceleration = 0.8

    # 等待 Gazebo/控制器/状态反馈起来（类似你原先的 sleep(10)）
    time.sleep(7.0)

    # # ========== 1) 机械臂回 home ==========
    # logger.info("机械臂 -> HOME")
    # arm.move_to_configuration(joint_positions=HOME_ARM, joint_names=ARM_JOINT_NAMES)
    # # arm.wait_until_executed()
    # time.sleep(7.0)

    # ========== 2) 夹爪 open ==========
    logger.info("夹爪 -> OPEN")
    hand.move_to_configuration(joint_positions=OPEN, joint_names=HAND_JOINT_NAMES)
    hand.wait_until_executed()
    time.sleep(7.0)

    # # ========== 3) 机械臂到配置 0 ==========
    logger.info("机械臂 -> 配置 0")
    arm.move_to_configuration(joint_positions=ARM_CONF_0, joint_names=ARM_JOINT_NAMES)
    arm.wait_until_executed()
    time.sleep(7.0)

    # ========== 4) 机械臂到配置 1 ==========
    logger.info("机械臂 -> 配置 1")
    arm.move_to_configuration(joint_positions=ARM_CONF_1, joint_names=ARM_JOINT_NAMES)
    arm.wait_until_executed()
    time.sleep(7.0)

    # ========== 5) 夹爪到轻微合拢 ==========
    logger.info("夹爪 -> CLOSE")
    hand.move_to_configuration(joint_positions=CLOSE, joint_names=HAND_JOINT_NAMES)
    hand.wait_until_executed()
    time.sleep(7.0)

    # ========== 6) 机械臂到配置 2 ==========
    logger.info("机械臂 -> 配置 2")
    arm.move_to_configuration(joint_positions=ARM_CONF_2, joint_names=ARM_JOINT_NAMES)
    arm.wait_until_executed()
    time.sleep(7.0)

    # ========== 7) 机械臂到配置 3 ==========
    logger.info("机械臂 -> 配置 3")
    arm.move_to_configuration(joint_positions=ARM_CONF_3, joint_names=ARM_JOINT_NAMES)
    arm.wait_until_executed()
    time.sleep(7.0)

    # ========== 8) 夹爪再次 open ==========
    logger.info("夹爪 -> 再次 OPEN")
    hand.move_to_configuration(joint_positions=OPEN, joint_names=HAND_JOINT_NAMES)
    hand.wait_until_executed()
    time.sleep(7.0)

    # ========== 9) 机械臂回 HOME ==========
    logger.info("机械臂 -> 回 HOME")
    arm.move_to_configuration(joint_positions=HOME_ARM, joint_names=ARM_JOINT_NAMES)
    arm.wait_until_executed()
    time.sleep(7.0)

    logger.info("运动完成")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
