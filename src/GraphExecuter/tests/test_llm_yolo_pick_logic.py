from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_executer"))

from nodes.moveit2_yolobb_ws.llm_yolo_pick_logic import (  # noqa: E402
    make_preview,
    parse_selected_index,
    preview_is_confirmable,
)
from nodes.moveit2_yolobb_ws.llm_yolo_pick_preview import LLMYoloPickPreviewNode  # noqa: E402
from nodes.moveit2_yolobb_ws import moveit2_yolobb_ws as yolo_module  # noqa: E402
from nodes.moveit2_yolobb_ws.moveit2_yolobb_ws import YoloObbNode  # noqa: E402
from utils import deepseek_credentials  # noqa: E402


CANDIDATES = [
    {"index": 0, "class_name": "bolt", "center_uv": [20.0, 10.0]},
    {"index": 1, "class_name": "case", "center_uv": [50.0, 30.0]},
]


def test_parse_selected_index_accepts_known_candidate():
    assert parse_selected_index('{"selected_index": 1}', CANDIDATES) == 1


@pytest.mark.parametrize("response", ['{}', '{"selected_index": 9}', '{"selected_index": true}', 'not-json'])
def test_parse_selected_index_rejects_invalid_response(response):
    with pytest.raises(ValueError):
        parse_selected_index(response, CANDIDATES)


def test_preview_confirmation_requires_matching_fresh_frame():
    preview = make_preview(
        {"index": 1, "class_name": "case", "target": [0.2, 0.25, 0.0], "frame_stamp_ns": 42},
        now_monotonic=10.0,
    )

    assert preview_is_confirmable(preview, 42, 2.0, now_monotonic=11.9)
    assert not preview_is_confirmable(preview, 41, 2.0, now_monotonic=11.0)
    assert not preview_is_confirmable(preview, 42, 2.0, now_monotonic=12.1)


class _FakeYoloNode:
    def __init__(self, stamp_ns):
        self.stamp_ns = stamp_ns
        self.published = []

    def current_pick_frame_stamp_ns(self):
        return self.stamp_ns

    def publish_pick_target(self, target):
        self.published.append(tuple(target))
        return True


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


class _FakeCompletionClient:
    class _Completions:
        @staticmethod
        def create(**_kwargs):
            message = type("Message", (), {"content": '{"selected_index": 0}'})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    chat = type("Chat", (), {"completions": _Completions()})()


class _FakePreviewYolo(_FakeYoloNode):
    def get_pick_candidates(self):
        return CANDIDATES

    def preview_pick_candidate(self, index):
        return {
            "index": index,
            "class_name": "bolt",
            "target": [0.1, 0.2, 0.3],
            "frame_stamp_ns": self.stamp_ns,
        }


def _preview_node(preview):
    node = object.__new__(LLMYoloPickPreviewNode)
    node._preview = preview
    node._preview_max_age_sec = lambda: 2.0
    node.set_property = lambda *_args: None
    node._emit = lambda _message: None
    return node


def test_confirmation_publishes_cached_target_once(monkeypatch):
    preview = make_preview(
        {"index": 0, "class_name": "bolt", "target": [0.1, 0.2, 0.3], "frame_stamp_ns": 42},
        now_monotonic=10.0,
    )
    monkeypatch.setattr(
        "nodes.moveit2_yolobb_ws.llm_yolo_pick_preview.preview_is_confirmable",
        lambda *_args: True,
    )
    node = _preview_node(preview)
    yolo = _FakeYoloNode(42)

    node._confirm(yolo)
    node._confirm(yolo)

    assert yolo.published == [(0.1, 0.2, 0.3)]


def test_preview_then_confirm_executes_exactly_one_pick(monkeypatch):
    text_node = type("TextNode", (), {"text_out": "抓取 bolt"})()
    yolo = _FakePreviewYolo(42)
    messages = []
    properties = {"confirm_pick": False, "preview_max_age_sec": "2.0"}
    node = object.__new__(LLMYoloPickPreviewNode)
    node.client = _FakeCompletionClient()
    node._client_api_key = "test-key"
    node._preview = None
    node.messageSignal = type("Signal", (), {"emit": messages.append})()
    node.get_property = properties.get
    node.set_property = properties.__setitem__
    node._upstream_node = lambda index: (text_node, yolo)[index]
    monkeypatch.setattr(deepseek_credentials, "get_deepseek_api_key", lambda: "test-key")

    node.execute()
    assert yolo.published == []
    assert messages[-1].startswith("Preview: bolt[0]")

    properties["confirm_pick"] = True
    node.execute()
    node.execute()

    assert yolo.published == [(0.1, 0.2, 0.3)]


def test_expired_confirmation_does_not_publish(monkeypatch):
    preview = make_preview(
        {"index": 0, "class_name": "bolt", "target": [0.1, 0.2, 0.3], "frame_stamp_ns": 42},
        now_monotonic=10.0,
    )
    monkeypatch.setattr(
        "nodes.moveit2_yolobb_ws.llm_yolo_pick_preview.preview_is_confirmable",
        lambda *_args: False,
    )
    node = _preview_node(preview)
    yolo = _FakeYoloNode(42)

    node._confirm(yolo)

    assert not yolo.published


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
