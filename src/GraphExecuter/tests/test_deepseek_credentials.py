from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_executer"))

from utils import deepseek_credentials  # noqa: E402


@pytest.fixture
def fake_keyring(monkeypatch):
    store = {}
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        deepseek_credentials.keyring,
        "get_password",
        lambda service, account: store.get((service, account)),
    )
    monkeypatch.setattr(
        deepseek_credentials.keyring,
        "set_password",
        lambda service, account, value: store.__setitem__((service, account), value),
    )

    def delete_password(service, account):
        try:
            del store[(service, account)]
        except KeyError as exc:
            raise deepseek_credentials.PasswordDeleteError("missing") from exc

    monkeypatch.setattr(deepseek_credentials.keyring, "delete_password", delete_password)
    return store


def test_keyring_save_read_delete_lifecycle(fake_keyring):
    assert deepseek_credentials.credential_source() == "missing"
    with pytest.raises(deepseek_credentials.DeepSeekCredentialError, match="Tools"):
        deepseek_credentials.get_deepseek_api_key()

    deepseek_credentials.save_deepseek_api_key("  test-key  ")
    assert deepseek_credentials.credential_source() == "keyring"
    assert deepseek_credentials.get_deepseek_api_key() == "test-key"
    assert deepseek_credentials.delete_deepseek_api_key()
    assert not deepseek_credentials.delete_deepseek_api_key()


def test_environment_key_takes_precedence(monkeypatch, fake_keyring):
    deepseek_credentials.save_deepseek_api_key("keyring-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")

    assert deepseek_credentials.credential_source() == "environment"
    assert deepseek_credentials.get_deepseek_api_key() == "environment-key"


def test_empty_key_is_rejected(fake_keyring):
    with pytest.raises(ValueError, match="cannot be empty"):
        deepseek_credentials.save_deepseek_api_key("  ")


def test_keyring_backend_failure_is_actionable(monkeypatch, fake_keyring):
    monkeypatch.setattr(
        deepseek_credentials.keyring,
        "get_password",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("backend offline")),
    )

    with pytest.raises(deepseek_credentials.DeepSeekCredentialError, match="backend offline"):
        deepseek_credentials.get_deepseek_api_key()


def test_nodes_that_call_deepseek_use_the_shared_credential_provider():
    root = Path(__file__).resolve().parents[1] / "graph_executer"
    for relative_path in (
        "nodes/llm/deepseek.py",
        "nodes/fairino_arm/arm_control.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "deepseek_credentials.get_deepseek_api_key()" in source
        assert "os.environ" not in source
        assert "llm.json" not in source


def test_yolo_task_preview_is_only_a_central_ros_client():
    root = Path(__file__).resolve().parents[1] / "graph_executer"
    source = (
        root / "nodes/moveit2_yolobb_ws/llm_yolo_pick_preview.py"
    ).read_text(encoding="utf-8")

    assert "/llm_arm/preview_command" in source
    assert "/llm_arm/execute_preview" in source
    assert "OpenAI" not in source
    assert "get_deepseek_api_key" not in source
    assert "os.environ" not in source
