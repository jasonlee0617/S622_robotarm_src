import pytest

from llm_arm_control import deepseek_credentials as credentials


def test_environment_short_circuits_keyring(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.setattr(credentials.keyring, "get_password", lambda *_: pytest.fail("keyring read"))
    assert credentials.credential_status() == "environment"
    assert credentials.get_deepseek_api_key() == "env-key"


def test_keyring_fallback_and_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(credentials.keyring, "get_password", lambda *_: "ring-key")
    assert credentials.get_deepseek_api_key() == "ring-key"
    monkeypatch.setattr(credentials.keyring, "get_password", lambda *_: None)
    with pytest.raises(credentials.DeepSeekCredentialError):
        credentials.get_deepseek_api_key()


def test_set_and_delete_use_stable_keyring_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(credentials.keyring, "set_password", lambda *args: calls.append(("set", args)))
    monkeypatch.setattr(credentials.keyring, "delete_password", lambda *args: calls.append(("delete", args)))
    credentials.set_deepseek_api_key("new-key")
    assert credentials.delete_deepseek_api_key()
    assert calls == [
        ("set", ("GraphExecuter", "deepseek_api_key", "new-key")),
        ("delete", ("GraphExecuter", "deepseek_api_key")),
    ]
