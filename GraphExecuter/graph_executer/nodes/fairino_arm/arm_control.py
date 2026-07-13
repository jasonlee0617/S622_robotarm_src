#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re

from NodeGraphQt import BaseNode, NodeBaseWidget
from Qt import QtWidgets
from geometry_msgs.msg import PoseStamped
from llm_arm_control.srv import ControlPose
from openai import OpenAI
from utils import deepseek_credentials
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from utils.general import find_nodes_folder

__all__ = ["FairinoArmControlNode", "FairinoArmDeepSeekControlNode"]


def _gripper_width_from_state(value):
    return 0.04 if str(value).strip().lower() == "open" else 0.0


def _pose_from_values(values, frame_id="base_link"):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(values[0])
    pose.pose.position.y = float(values[1])
    pose.pose.position.z = float(values[2])
    pose.pose.orientation.x = float(values[3])
    pose.pose.orientation.y = float(values[4])
    pose.pose.orientation.z = float(values[5])
    pose.pose.orientation.w = float(values[6])
    return pose


def _pose_values(pose):
    return [
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z,
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    ]


class _FairinoRosMixin:
    def create_ros2_node(self):
        self.srv_node = Node(self.NODE_NAME.replace(" ", "_"))
        self.cli = self.srv_node.create_client(ControlPose, "/llm_arm/control_pose")
        self.sub_end_pose = self.srv_node.create_subscription(
            PoseStamped,
            "/llm_arm/current_pose",
            self.end_pose_listener_callback,
            10,
        )
        self.end_pose = None
        self.is_get_end_pose = False
        self.is_created_node = True

    def delete_ros2_node(self):
        self.srv_node.destroy_node()
        self.is_created_node = False

    def end_pose_listener_callback(self, msg):
        self.end_pose = msg
        self.is_get_end_pose = True

    def wait_current_pose(self):
        while not self.is_get_end_pose:
            rclpy.spin_once(self.srv_node)
        self.is_get_end_pose = False
        return self.end_pose

    def call_control_pose(self, pose, gripper_width):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.srv_node.get_logger().info("/llm_arm/control_pose not available, waiting again...")
        request = ControlPose.Request()
        request.target_pose = pose
        request.target_pose.header.stamp = self.srv_node.get_clock().now().to_msg()
        request.gripper_width = float(gripper_width)
        request.execute = True
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self.srv_node, future)
        return future.result()


class FairinoArmDeepSeekControlNode(_FairinoRosMixin, BaseNode):
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = "FairinoArmDeepSeekControlNode"

    def __init__(self):
        super().__init__()
        self.add_input("text_in")
        self.add_output("next_step")
        self.add_text_input("max_mem_len", label="Max Memory Length")
        self.set_property("max_mem_len", "20")
        self.client = None
        self._client_api_key = None
        self.messages = [{
            "role": "system",
            "content": """
你是一个六自由度机械臂控制助手。请根据用户意图输出机械臂末端目标位姿，必须只返回 JSON：
{
  "position.x": 0.3,
  "position.y": 0.3,
  "position.z": 0.2,
  "orientation.x": 0.0,
  "orientation.y": 0.0,
  "orientation.z": 0.0,
  "orientation.w": 1.0,
  "gripper_state": "open",
  "answer": "已给出目标位置和姿态。"
}
gripper_state 只能是 open 或 close。
""",
        }]
        self.is_created_node = False

    def _get_client(self):
        deepseek_api = deepseek_credentials.get_deepseek_api_key()
        if self.client is None or self._client_api_key != deepseek_api:
            self.client = OpenAI(api_key=deepseek_api, base_url="https://api.deepseek.com")
            self._client_api_key = deepseek_api
        return self.client

    def execute(self):
        try:
            client = self._get_client()
        except RuntimeError as exc:
            self.messageSignal.emit(str(exc))
            return
        if not self.is_created_node:
            self.create_ros2_node()

        text_in = self.input(0).connected_ports()[0].node().text_out
        end_pose = _pose_values(self.wait_current_pose())
        user_prompt = (
            "机械臂末端当前位置和姿态是："
            f'"position.x": {end_pose[0]}, "position.y": {end_pose[1]}, "position.z": {end_pose[2]}, '
            f'"orientation.x": {end_pose[3]}, "orientation.y": {end_pose[4]}, '
            f'"orientation.z": {end_pose[5]}, "orientation.w": {end_pose[6]}; '
            f"用户意图是：{text_in}"
        )
        self.messages.append({"role": "user", "content": user_prompt})
        self.messageSignal.emit(f"{self.NODE_NAME}的LLM输入：{user_prompt}")

        completion = client.chat.completions.create(messages=self.messages, model="deepseek-chat")
        assistant_message = completion.choices[0].message.content
        matches = re.findall(r"\{([^{}]*)\}", assistant_message)
        data_json = json.loads("{" + matches[0] + "}")
        self.messages.append(completion.choices[0].message)
        if len(self.messages) > int(self.get_property("max_mem_len")):
            del self.messages[1:3]

        pose = _pose_from_values([
            data_json["position.x"],
            data_json["position.y"],
            data_json["position.z"],
            data_json["orientation.x"],
            data_json["orientation.y"],
            data_json["orientation.z"],
            data_json["orientation.w"],
        ])
        response = self.call_control_pose(pose, _gripper_width_from_state(data_json["gripper_state"]))
        self.text_out = data_json["answer"] + ("执行成功！" if response.success else "执行失败！")
        self.messageSignal.emit(response.message)
        self.delete_ros2_node()
        self.messageSignal.emit(f"{self.NODE_NAME} executed.")

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal


class MyCustomWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.btn = QtWidgets.QPushButton("get_current_pose")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.btn)
        self.btn.clicked.connect(self.get_end_pose)

    def set_node_obj(self, obj):
        self.node_obj = obj

    def get_end_pose(self):
        if not self.node_obj.is_created_node:
            self.node_obj.create_ros2_node()
        self.node_obj.get_end_pose()
        self.node_obj.get_joint_states()


class NodeWidgetWrapper(NodeBaseWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_name("my_widget")
        self.set_custom_widget(MyCustomWidget())
        self.wire_signals()

    def wire_signals(self):
        pass

    def get_value(self):
        pass

    def set_value(self, value):
        pass


class FairinoArmControlNode(_FairinoRosMixin, BaseNode):
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = "fairino arm control"

    def __init__(self):
        super().__init__()
        self.add_input("in")
        self.add_output("next_step")
        node_widget = NodeWidgetWrapper(self.view)
        node_widget.get_custom_widget().set_node_obj(self)
        self.add_custom_widget(node_widget, tab="Custom")
        self.add_text_input("position.x", "position.x", text="0.3")
        self.add_text_input("position.y", "position.y", text="0.3")
        self.add_text_input("position.z", "position.z", text="0.3")
        self.add_text_input("orientation.x", "orientation.x", text="0.0")
        self.add_text_input("orientation.y", "orientation.y", text="0.0")
        self.add_text_input("orientation.z", "orientation.z", text="0.0")
        self.add_text_input("orientation.w", "orientation.w", text="1.0")
        self.add_combo_menu("gripper_state", "gripper_state", items=["open", "close"])
        self.is_created_node = False

    def create_ros2_node(self):
        super().create_ros2_node()
        self.sub_joint_states = self.srv_node.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_listener_callback,
            10,
        )
        self.joint_states = None
        self.is_get_joint_states = False

    def joint_states_listener_callback(self, msg):
        self.joint_states = msg
        self.is_get_joint_states = True

    def get_joint_states(self):
        while not self.is_get_joint_states:
            rclpy.spin_once(self.srv_node)
        self.is_get_joint_states = False
        if self.joint_states.position and self.joint_states.position[-1] < 0.01:
            self.set_property("gripper_state", "close")
        else:
            self.set_property("gripper_state", "open")

    def get_end_pose(self):
        values = _pose_values(self.wait_current_pose())
        for key, value in zip(
            ["position.x", "position.y", "position.z", "orientation.x", "orientation.y", "orientation.z", "orientation.w"],
            values,
        ):
            self.set_property(key, str(value))
        return values

    def execute(self):
        if not self.is_created_node:
            self.create_ros2_node()
        pose = _pose_from_values([
            self.get_property("position.x"),
            self.get_property("position.y"),
            self.get_property("position.z"),
            self.get_property("orientation.x"),
            self.get_property("orientation.y"),
            self.get_property("orientation.z"),
            self.get_property("orientation.w"),
        ])
        response = self.call_control_pose(pose, _gripper_width_from_state(self.get_property("gripper_state")))
        self.text_out = "执行成功！" if response.success else "执行失败！"
        self.messageSignal.emit(response.message)
        self.delete_ros2_node()
        self.messageSignal.emit(f"{self.NODE_NAME} executed.")

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal
