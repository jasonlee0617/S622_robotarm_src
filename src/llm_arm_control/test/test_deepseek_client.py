import io
import json
from urllib import error

import pytest

from llm_arm_control import deepseek_client


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_chat_uses_json_mode_without_leaking_key(monkeypatch):
    captured = {}

    def urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({"choices": [{"message": {"content": '{"actions":[]}'}}]})

    monkeypatch.setattr(deepseek_client.request, "urlopen", urlopen)
    client = deepseek_client.DeepSeekClient("secret-key", timeout_sec=7.0)

    result = client.chat([{"role": "user", "content": "hello"}])

    assert result == '{"actions":[]}'
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["stream"] is False
    assert captured["timeout"] == 7.0


def test_http_error_is_bounded_and_does_not_include_key(monkeypatch):
    http_error = error.HTTPError(
        "https://api.deepseek.com/chat/completions",
        401,
        "unauthorized",
        {},
        io.BytesIO(b"x" * 500),
    )
    monkeypatch.setattr(
        deepseek_client.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error),
    )

    with pytest.raises(deepseek_client.DeepSeekClientError) as caught:
        deepseek_client.DeepSeekClient("secret-key").chat([])

    assert "HTTP 401" in str(caught.value)
    assert "secret-key" not in str(caught.value)
    assert len(str(caught.value)) < 350


def test_missing_assistant_content_is_rejected(monkeypatch):
    monkeypatch.setattr(
        deepseek_client.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"choices": []}),
    )

    with pytest.raises(deepseek_client.DeepSeekClientError, match="assistant content"):
        deepseek_client.DeepSeekClient("key").chat([])
