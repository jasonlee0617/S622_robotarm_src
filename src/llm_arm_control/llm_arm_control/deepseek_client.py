"""Small DeepSeek Chat Completions client using only the Python standard library."""

from __future__ import annotations

import json
from urllib import error, request


class DeepSeekClientError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", timeout_sec: float = 30.0):
        self.api_key = str(api_key).strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        if not self.api_key:
            raise ValueError("DeepSeek API key cannot be empty")

    def chat(self, messages, model: str = "deepseek-chat") -> str:
        payload = json.dumps(
            {
                "model": str(model),
                "messages": list(messages),
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekClientError(f"DeepSeek HTTP {exc.code}: {detail[:300]}") from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DeepSeekClientError(f"DeepSeek request failed: {exc}") from exc

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekClientError("DeepSeek response has no assistant content") from exc
