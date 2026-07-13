#!/usr/bin/env python3
"""GraphExecuter client for the shared Fairino LLM-YOLO task server."""

import json
import uuid

from NodeGraphQt import BaseNode
from PySide6.QtCore import QObject, Qt, Signal, Slot

from utils.general import find_nodes_folder


__all__ = ["LLMYoloPickPreviewNode"]


class _PreviewUiBridge(QObject):
    reset_confirm_requested = Signal()

    def __init__(self, node):
        super().__init__()
        self._node = node
        self.reset_confirm_requested.connect(self._reset_confirm, Qt.QueuedConnection)

    @Slot()
    def _reset_confirm(self):
        self._node.set_property("confirm_pick", False)


class LLMYoloPickPreviewNode(BaseNode):
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = "LLM YOLO pick preview"

    def __init__(self):
        super().__init__()
        self.add_input("text_in")
        self.add_input("yolo_obb")  # Saved-graph compatibility; server owns perception now.
        self.add_output("preview")
        self.add_checkbox("confirm_pick", text="Confirm task")
        # Kept so existing saved graphs still load. The central server owns and
        # enforces the preview lifetime; this value is display-only.
        self.add_text_input("preview_max_age_sec", label="Server preview age (s)")
        self.set_property("preview_max_age_sec", "15.0")
        self.text_out = ""
        self._session_id = uuid.uuid4().hex
        self._preview_id = ""
        self._ros_node = None
        self._preview_client = None
        self._action_client = None
        self._ui_bridge = _PreviewUiBridge(self)

    def _emit(self, message):
        self.text_out = str(message)
        self.messageSignal.emit(self.text_out)

    def _upstream_node(self, input_index):
        ports = self.input(input_index).connected_ports()
        return ports[0].node() if ports else None

    def _request_confirm_reset(self):
        bridge = getattr(self, "_ui_bridge", None)
        if bridge is None:
            self.set_property("confirm_pick", False)
        else:
            bridge.reset_confirm_requested.emit()

    def _ensure_ros(self):
        if self._ros_node is not None:
            return True
        try:
            from llm_arm_control.action import ExecutePreview
            from llm_arm_control.srv import PreviewCommand
            from rclpy.action import ActionClient
            from rclpy.node import Node
            self._preview_type = PreviewCommand
            self._execute_type = ExecutePreview
            self._ros_node = Node(
                f"graph_executer_llm_task_client_{self._session_id[:8]}"
            )
            self._preview_client = self._ros_node.create_client(
                PreviewCommand, "/llm_arm/preview_command"
            )
            self._action_client = ActionClient(
                self._ros_node, ExecutePreview, "/llm_arm/execute_preview"
            )
        except Exception as exc:
            if self._ros_node is not None:
                try:
                    self._ros_node.destroy_node()
                except Exception:
                    pass
            self._ros_node = None
            self._preview_client = None
            self._action_client = None
            self._emit(
                "LLM task interfaces unavailable; source the rebuilt ROS overlay: "
                f"{exc}"
            )
            return False
        return True

    def _spin_future(self, future, timeout_sec):
        import rclpy

        rclpy.spin_until_future_complete(
            self._ros_node,
            future,
            timeout_sec=None if timeout_sec is None else float(timeout_sec),
        )
        return future.result() if future.done() else None

    def _feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self._emit(
            f"[{feedback.step_index}/{feedback.step_count}] "
            f"{feedback.phase}: {feedback.message}"
        )

    def _confirm(self):
        self._request_confirm_reset()
        if not self._preview_id:
            self._emit("No pending preview; generate one before confirming.")
            return
        try:
            action_available = self._action_client.wait_for_server(timeout_sec=2.0)
        except Exception as exc:
            self._emit(f"LLM execute action lookup failed: {exc}")
            return
        if not action_available:
            self._emit("LLM execute action is unavailable.")
            return
        goal = self._execute_type.Goal()
        goal.session_id = self._session_id
        goal.preview_id = self._preview_id
        try:
            goal_future = self._action_client.send_goal_async(
                goal,
                feedback_callback=self._feedback,
            )
        except Exception as exc:
            self._emit(f"Task confirmation could not be submitted: {exc}")
            return
        # From this point submission is uncertain or complete. Consume the ID
        # locally so a retry cannot execute the same server preview twice.
        self._preview_id = ""
        try:
            goal_handle = self._spin_future(goal_future, 5.0)
        except Exception as exc:
            self._emit(f"Task confirmation status is unknown: {exc}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self._emit("Task confirmation was rejected or expired.")
            return
        try:
            wrapped = self._spin_future(goal_handle.get_result_async(), None)
        except Exception as exc:
            self._emit(f"Task execution result is unavailable: {exc}")
            return
        if wrapped is None:
            self._emit("Task execution ended without a result.")
            return
        result = wrapped.result
        self._emit(f"{result.terminal_state}: {result.message}")

    def execute(self):
        text_node = self._upstream_node(0)
        if text_node is None:
            self._emit("Connect Text input before running this node.")
            return
        if not self._ensure_ros():
            return
        if self.get_property("confirm_pick"):
            self._confirm()
            return

        instruction = str(getattr(text_node, "text_out", "")).strip()
        if not instruction:
            self._emit("Enter a task instruction before generating a preview.")
            return
        # A new instruction invalidates any locally cached confirmation even
        # if the replacement request later fails.
        self._preview_id = ""
        try:
            preview_available = self._preview_client.wait_for_service(timeout_sec=2.0)
        except Exception as exc:
            self._emit(f"LLM preview service lookup failed: {exc}")
            return
        if not preview_available:
            self._emit("LLM preview service is unavailable; start llm_yolo_control.launch.py.")
            return
        request = self._preview_type.Request()
        request.session_id = self._session_id
        request.instruction = instruction
        try:
            response = self._spin_future(
                self._preview_client.call_async(request),
                45.0,
            )
        except Exception as exc:
            self._emit(f"LLM task preview request failed: {exc}")
            return
        if response is None:
            self._emit("LLM task preview timed out.")
            return
        self._preview_id = response.preview_id if response.accepted else ""
        if response.accepted and not self._preview_id:
            self._emit(
                "invalid_response: preview service accepted the task without a preview ID"
            )
        elif response.accepted:
            try:
                preview = json.dumps(
                    json.loads(response.preview_json),
                    ensure_ascii=False,
                    indent=2,
                )
            except (TypeError, json.JSONDecodeError):
                preview = response.preview_json
            self._emit(f"{preview}\n{response.message}")
        else:
            self._emit(f"{response.status}: {response.message}")

    def close_node(self):
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        self._preview_client = None
        self._action_client = None
        self._preview_id = ""

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal
