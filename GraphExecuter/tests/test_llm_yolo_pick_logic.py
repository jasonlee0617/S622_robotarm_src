from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_executer"))

from nodes.moveit2_yolobb_ws.llm_yolo_pick_preview import (  # noqa: E402
    LLMYoloPickPreviewNode,
)
from nodes.moveit2_yolobb_ws import moveit2_yolobb_ws as yolo_module  # noqa: E402
from nodes.moveit2_yolobb_ws.moveit2_yolobb_ws import YoloObbNode  # noqa: E402


class _PreviewType:
    class Request:
        session_id = ""
        instruction = ""


class _ExecuteType:
    class Goal:
        session_id = ""
        preview_id = ""


class _FakePreviewClient:
    def __init__(self, response, available=True):
        self.response = response
        self.available = available
        self.requests = []

    def wait_for_service(self, timeout_sec):
        assert timeout_sec == 2.0
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        return self.response


class _FakeGoalHandle:
    def __init__(self, accepted=True, result=None):
        self.accepted = accepted
        self._result = result

    def get_result_async(self):
        return SimpleNamespace(result=self._result)


class _FakeActionClient:
    def __init__(self, goal_handle, available=True):
        self.goal_handle = goal_handle
        self.available = available
        self.goals = []
        self.feedback_callbacks = []

    def wait_for_server(self, timeout_sec):
        assert timeout_sec == 2.0
        return self.available

    def send_goal_async(self, goal, feedback_callback):
        self.goals.append(goal)
        self.feedback_callbacks.append(feedback_callback)
        return self.goal_handle


def _client_node(response=None):
    node = object.__new__(LLMYoloPickPreviewNode)
    node._session_id = "session-1"
    node._preview_id = ""
    node._preview_type = _PreviewType
    node._execute_type = _ExecuteType
    node._ros_node = object()
    node._preview_client = _FakePreviewClient(response)
    node._action_client = None
    node._ui_bridge = None
    node.text_out = ""
    node.messages = []
    node.properties = {"confirm_pick": False}
    node._emit = lambda message: (
        setattr(node, "text_out", str(message)),
        node.messages.append(str(message)),
    )
    node.get_property = node.properties.get
    node.set_property = node.properties.__setitem__
    node._ensure_ros = lambda: True
    node._spin_future = lambda future, _timeout: future
    return node


def test_preview_calls_central_service_without_using_yolo_input():
    response = SimpleNamespace(
        accepted=True,
        status="preview_ready",
        preview_id="preview-1",
        preview_json='{"actions":[{"type":"pick_place"}]}',
        message="Confirm once to execute the complete task.",
    )
    node = _client_node(response)
    text_node = SimpleNamespace(text_out="  抓取 bolt，然后放到 box  ")

    def upstream(index):
        assert index == 0, "the Qt client must not read the legacy YOLO input"
        return text_node

    node._upstream_node = upstream
    node.execute()

    assert node._preview_id == "preview-1"
    assert len(node._preview_client.requests) == 1
    request = node._preview_client.requests[0]
    assert request.session_id == "session-1"
    assert request.instruction == "抓取 bolt，然后放到 box"
    assert '"actions": [' in node.text_out
    assert node.text_out.endswith("Confirm once to execute the complete task.")


def test_rejected_preview_clears_previous_preview():
    response = SimpleNamespace(
        accepted=False,
        status="clarification_required",
        preview_id="",
        preview_json="",
        message="Which bolt should be used?",
    )
    node = _client_node(response)
    node._preview_id = "stale-preview"
    node._upstream_node = lambda _index: SimpleNamespace(text_out="抓取 bolt")

    node.execute()

    assert node._preview_id == ""
    assert node.text_out == "clarification_required: Which bolt should be used?"


def test_accepted_preview_requires_nonempty_server_id():
    response = SimpleNamespace(
        accepted=True,
        status="preview_ready",
        preview_id="",
        preview_json="{}",
        message="",
    )
    node = _client_node(response)
    node._upstream_node = lambda _index: SimpleNamespace(text_out="抓取 cube")

    node.execute()

    assert node._preview_id == ""
    assert node.text_out.startswith("invalid_response:")


def test_preview_service_unavailable_is_actionable():
    node = _client_node(None)
    node._preview_id = "stale-preview"
    node._preview_client.available = False
    node._upstream_node = lambda _index: SimpleNamespace(text_out="抓取 cube")

    node.execute()

    assert node._preview_id == ""
    assert node._preview_client.requests == []
    assert "start llm_yolo_control.launch.py" in node.text_out


def test_confirmation_executes_central_action_exactly_once():
    result = SimpleNamespace(terminal_state="SUCCEEDED", message="task complete")
    action_client = _FakeActionClient(_FakeGoalHandle(result=result))
    node = _client_node()
    node._preview_id = "preview-1"
    node._action_client = action_client

    node._confirm()
    node._confirm()

    assert len(action_client.goals) == 1
    assert action_client.goals[0].session_id == "session-1"
    assert action_client.goals[0].preview_id == "preview-1"
    assert node._preview_id == ""
    assert node.properties["confirm_pick"] is False
    assert node.text_out == "No pending preview; generate one before confirming."


def test_unavailable_action_keeps_preview_for_safe_retry():
    action_client = _FakeActionClient(None, available=False)
    node = _client_node()
    node._preview_id = "preview-1"
    node._action_client = action_client

    node._confirm()

    assert node._preview_id == "preview-1"
    assert action_client.goals == []
    assert node.text_out == "LLM execute action is unavailable."


def test_action_rejection_consumes_preview():
    action_client = _FakeActionClient(_FakeGoalHandle(accepted=False))
    node = _client_node()
    node._preview_id = "preview-1"
    node._action_client = action_client

    node._confirm()

    assert node._preview_id == ""
    assert node.text_out == "Task confirmation was rejected or expired."


def test_uncertain_action_submission_consumes_preview_to_prevent_replay():
    action_client = _FakeActionClient(_FakeGoalHandle())
    node = _client_node()
    node._preview_id = "preview-1"
    node._action_client = action_client
    node._spin_future = lambda _future, _timeout: (_ for _ in ()).throw(
        RuntimeError("transport interrupted")
    )

    node._confirm()

    assert len(action_client.goals) == 1
    assert node._preview_id == ""
    assert node.text_out == "Task confirmation status is unknown: transport interrupted"


def test_action_feedback_is_forwarded_to_graph_log():
    node = _client_node()
    feedback = SimpleNamespace(
        step_index=2,
        step_count=5,
        phase="REVALIDATE",
        message="box target updated",
    )

    node._feedback(SimpleNamespace(feedback=feedback))

    assert node.text_out == "[2/5] REVALIDATE: box target updated"


class _FakeHeader:
    class _Stamp:
        sec = 1
        nanosec = 2

    stamp = _Stamp()


class _FakeDetection:
    class_name = "bolt"
    coordinates = [10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]


class _FakeYoloMessage:
    header = _FakeHeader()
    yolov8_inference = [_FakeDetection()]


class _FakeLogger:
    def warning(self, _message):
        pass


class _FakeCameraNode:
    def get_logger(self):
        return _FakeLogger()


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_yolo_preview_rejects_missing_depth_tf_target():
    node = YoloObbNode.__new__(YoloObbNode)
    node.last_yolo = _FakeYoloMessage()
    node.active_yolo = node.last_yolo
    node.active_frame = {"header": node.active_yolo.header}
    node.camera_subscriber = _FakeCameraNode()
    node.project_obb_to_base = lambda _points: None

    assert node.get_pick_candidates() == [
        {"index": 0, "class_name": "bolt", "center_uv": [15.0, 15.0]}
    ]
    assert node.preview_pick_candidate(0) is None


def test_yolo_candidates_require_the_current_synced_frame():
    node = YoloObbNode.__new__(YoloObbNode)
    node.last_yolo = _FakeYoloMessage()
    node.active_yolo = None
    node.active_frame = None

    assert node.get_pick_candidates() == []


def test_candidates_use_the_last_synchronized_yolo_frame():
    node = YoloObbNode.__new__(YoloObbNode)
    node.active_yolo = _FakeYoloMessage()
    node.active_frame = {"header": node.active_yolo.header}
    node.last_yolo = type("NewerUnsynchronizedYolo", (), {
        "header": type("Header", (), {
            "stamp": type("Stamp", (), {"sec": 9, "nanosec": 9})(),
        })(),
        "yolov8_inference": [],
    })()

    assert node.get_pick_candidates() == [
        {"index": 0, "class_name": "bolt", "center_uv": [15.0, 15.0]}
    ]


def test_yolo_frozen_frame_does_not_poll_ros(monkeypatch):
    node = YoloObbNode.__new__(YoloObbNode)
    node.get_property = lambda name: name == "freeze_frame"
    node.create_ros2_node = lambda: pytest.fail("frozen frame must not create or poll ROS nodes")
    sleeps = []
    monkeypatch.setattr(yolo_module.time, "sleep", sleeps.append)

    node.execute()

    assert sleeps == [0.02]


def test_yolo_target_publish_sends_target_and_trigger_once():
    node = YoloObbNode.__new__(YoloObbNode)
    node.pub = _FakePublisher()
    node.trigger_pub = _FakePublisher()

    assert node.publish_pick_target([0.1, 0.2, 0.3])
    assert len(node.pub.messages) == 1
    assert list(node.pub.messages[0].data) == [0.1, 0.2, 0.3]
    assert len(node.trigger_pub.messages) == 1
