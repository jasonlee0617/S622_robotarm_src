import os
from pathlib import Path
import sys
import threading
import types


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_executer"))

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit

from nodes.fairino_arm.arm_control import FairinoArmDeepSeekControlNode
from nodes.llm.deepseek import DeepSeekLLMNode
from nodes.moveit2_yolobb_ws.llm_yolo_pick_preview import LLMYoloPickPreviewNode
from nodes.moveit2_yolobb_ws.moveit2_yolobb_ws import ImageDisplayWidget
from nodes.ocr.formula_recognition import Pix2TextNode
from src.mainwindow import MainWindow
from utils import deepseek_credentials


@pytest.fixture(scope="module", autouse=True)
def qapplication():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_real_deepseek_credentials(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(deepseek_credentials.keyring, "get_password", lambda *_args: None)


@pytest.mark.parametrize(
    "node_class",
    (DeepSeekLLMNode, FairinoArmDeepSeekControlNode),
)
def test_deepseek_nodes_can_be_created_without_an_api_key(monkeypatch, node_class):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    node = node_class()

    assert node.client is None
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        node._get_client()


@pytest.mark.parametrize(
    "node_class",
    (DeepSeekLLMNode, FairinoArmDeepSeekControlNode),
)
def test_deepseek_nodes_refresh_client_after_key_change(monkeypatch, node_class):
    module = sys.modules[node_class.__module__]
    current = {"key": "first-key"}
    created = []
    monkeypatch.setattr(deepseek_credentials, "get_deepseek_api_key", lambda: current["key"])
    monkeypatch.setattr(
        module,
        "OpenAI",
        lambda **kwargs: created.append(types.SimpleNamespace(**kwargs)) or created[-1],
    )
    node = node_class()

    first = node._get_client()
    assert node._get_client() is first
    current["key"] = "second-key"
    second = node._get_client()

    assert second is not first
    assert [client.api_key for client in created] == ["first-key", "second-key"]


def test_llm_yolo_qt_client_is_lazy_and_has_no_deepseek_client():
    node = LLMYoloPickPreviewNode()

    assert node._ros_node is None
    assert node._preview_client is None
    assert node._action_client is None
    assert not hasattr(node, "client")
    assert not hasattr(node, "_get_client")


def test_missing_api_key_is_reported_without_execution_failure(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    messages = []
    node = FairinoArmDeepSeekControlNode()
    node.set_messageSignal(types.SimpleNamespace(emit=messages.append))

    node.execute()

    assert messages == [deepseek_credentials.MISSING_MESSAGE]


def test_deepseek_node_reports_a_missing_text_connection(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    messages = []
    node = DeepSeekLLMNode()
    node.set_messageSignal(types.SimpleNamespace(emit=messages.append))

    node.execute()

    assert messages == ["DeepSeek LLM: connect Text input before execution"]


def test_pix2text_node_creation_does_not_load_the_model():
    node = Pix2TextNode()

    assert node.p2t is None


def test_pix2text_model_is_loaded_once_on_demand(monkeypatch):
    created = []

    class FakeLatexOCR:
        def __init__(self):
            created.append(self)

    package = types.ModuleType("pix2tex")
    package.__path__ = []
    cli = types.ModuleType("pix2tex.cli")
    cli.LatexOCR = FakeLatexOCR
    monkeypatch.setitem(sys.modules, "pix2tex", package)
    monkeypatch.setitem(sys.modules, "pix2tex.cli", cli)

    messages = []
    node = Pix2TextNode()
    node.set_messageSignal(types.SimpleNamespace(emit=messages.append))

    first = node._get_model()
    second = node._get_model()

    assert first is second
    assert len(created) == 1
    assert messages == ["pix2text: loading model (the first run may download weights)..."]


def test_yolo_image_update_from_worker_is_queued_to_the_qt_thread():
    widget = ImageDisplayWidget()
    image = np.ones((8, 8, 3), dtype=np.uint8)

    worker = threading.Thread(target=lambda: widget.live_image_signal.emit(image))
    worker.start()
    worker.join()

    assert widget.img.shape == (480, 640, 3)
    QApplication.processEvents()
    assert widget.img.shape == image.shape


def test_worker_log_message_is_queued_to_the_qt_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "initGui", lambda _self: None)
    window = MainWindow()
    window.messageconsole.logPath = str(tmp_path / "graph.log")

    worker = threading.Thread(target=lambda: window.messageSignal.emit("worker message"))
    worker.start()
    worker.join()

    assert "worker message" not in window.messageconsole.ui.textBrowser.toPlainText()
    QApplication.processEvents()
    assert "worker message" in window.messageconsole.ui.textBrowser.toPlainText()


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("missing", "未配置"),
        ("keyring", "系统密钥环"),
        ("environment", "环境变量"),
    ),
)
def test_deepseek_key_status_labels(monkeypatch, source, expected):
    monkeypatch.setattr(deepseek_credentials, "credential_source", lambda: source)

    assert expected in MainWindow._deepseek_key_status(None)


def test_preview_checkbox_reset_from_worker_is_queued_to_qt_thread():
    node = LLMYoloPickPreviewNode()
    updates = []
    node.set_property = lambda name, value: updates.append((name, value))

    worker = threading.Thread(target=node._request_confirm_reset)
    worker.start()
    worker.join()

    assert updates == []
    QApplication.processEvents()
    assert updates == [("confirm_pick", False)]


def test_step6_preview_runs_in_graph_worker_without_touching_qt(capsys, monkeypatch, tmp_path):
    window = MainWindow()
    window.messageconsole.logPath = str(tmp_path / "graph.log")
    graph = window.graph.graph
    text_node = graph.get_node_by_name("Text input")
    yolo_node = graph.get_node_by_name("yolo_obb")
    preview_node = graph.get_node_by_name("LLM YOLO pick preview")

    text_node.set_property("text_in", "抓取 bolt")
    yolo_node.set_property("freeze_frame", True)
    preview_node.set_property("confirm_pick", False)
    requests = []

    class PreviewType:
        class Request:
            session_id = ""
            instruction = ""

    class PreviewClient:
        @staticmethod
        def wait_for_service(timeout_sec):
            return timeout_sec == 2.0

        @staticmethod
        def call_async(request):
            requests.append(request)
            return types.SimpleNamespace(
                accepted=True,
                status="preview_ready",
                preview_id="preview-worker-test",
                preview_json=(
                    '{"actions":[{"type":"pick_place",'
                    '"source":"bolt","destination":"box"}]}'
                ),
                message="Confirm once to execute the complete task.",
            )

    preview_node._preview_type = PreviewType
    preview_node._preview_client = PreviewClient()
    preview_node._ensure_ros = lambda: True
    preview_node._spin_future = lambda future, _timeout: future
    graph.clear_selection()
    preview_node.set_selected(True)

    window.graph.execute_selected_nodes()
    worker = window.graph.threads[preview_node.NODE_NAME]
    worker.join(timeout=2.0)
    QApplication.processEvents()

    assert not worker.is_alive()
    assert preview_node._preview_id == "preview-worker-test"
    assert len(requests) == 1
    assert requests[0].session_id == preview_node._session_id
    assert requests[0].instruction == "抓取 bolt"
    assert "pick_place" in window.messageconsole.ui.textBrowser.toPlainText()
    assert "Confirm once" in window.messageconsole.ui.textBrowser.toPlainText()
    assert "QBasicTimer" not in capsys.readouterr().err
    assert window.deepseek_key_action in window.ui.menuTools.actions()

    echo_modes = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda _parent, _title, _label, mode: echo_modes.append(mode) or ("", False),
    )
    assert window._prompt_deepseek_api_key() == ("", False)
    assert echo_modes == [QLineEdit.Password]
    window.graph.timeer_check_thread.stop()
    window.hide()
