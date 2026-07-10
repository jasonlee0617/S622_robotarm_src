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
from nodes.moveit2_yolobb_ws.moveit2_yolobb_ws import YoloObbNode  # noqa: E402


CANDIDATES = [
    {"index": 0, "class_name": "bolt", "center_uv": [20.0, 10.0]},
    {"index": 1, "class_name": "case", "center_uv": [50.0, 30.0]},
]


def test_deepseek_nodes_read_only_the_environment_key():
    root = Path(__file__).resolve().parents[1] / "graph_executer"
    for relative_path in (
        "nodes/llm/deepseek.py",
        "nodes/fairino_arm/arm_control.py",
        "nodes/moveit2_yolobb_ws/llm_yolo_pick_preview.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "DEEPSEEK_API_KEY" in source
        assert "llm.json" not in source


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
    node.active_frame = {"header": node.last_yolo.header}
    node.camera_subscriber = _FakeCameraNode()
    node.project_obb_to_base = lambda _points: None

    assert node.get_pick_candidates() == [
        {"index": 0, "class_name": "bolt", "center_uv": [15.0, 15.0]}
    ]
    assert node.preview_pick_candidate(0) is None


def test_yolo_candidates_require_the_current_synced_frame():
    node = YoloObbNode.__new__(YoloObbNode)
    node.last_yolo = _FakeYoloMessage()
    node.active_frame = None

    assert node.get_pick_candidates() == []


def test_yolo_target_publish_sends_target_and_trigger_once():
    node = YoloObbNode.__new__(YoloObbNode)
    node.pub = _FakePublisher()
    node.trigger_pub = _FakePublisher()

    assert node.publish_pick_target([0.1, 0.2, 0.3])
    assert len(node.pub.messages) == 1
    assert list(node.pub.messages[0].data) == [0.1, 0.2, 0.3]
    assert len(node.trigger_pub.messages) == 1
