from types import SimpleNamespace

import pytest


pytest.importorskip("rclpy")

from llm_arm_control.llm_yolo_cli import (  # noqa: E402
    LlmYoloCli,
    deepseek_credentials,
    execution_key_effect,
    should_offer_cached_box_fallback,
)


def test_space_and_ctrl_c_stop_and_cancel_action():
    assert execution_key_effect(" ") == ("stop", True)
    assert execution_key_effect("\x03") == ("stop", True)


def test_home_reset_does_not_race_with_action_cancel():
    assert execution_key_effect("h") == ("reset", False)
    assert execution_key_effect("H") == ("reset", False)


def test_other_execution_keys_are_ignored():
    assert execution_key_effect("y") == ("", False)


def test_cached_box_fallback_is_offered_only_for_zero_fresh_frames():
    message = (
        "box was not stable within 5 seconds: 0 fresh frames, "
        "0 unstable windows; requires 5 stable frames"
    )
    assert should_offer_cached_box_fallback("HOLDING_RECOVERY", message)
    assert not should_offer_cached_box_fallback("FAILED", message)
    assert not should_offer_cached_box_fallback(
        "HOLDING_RECOVERY",
        "box was not stable within 5 seconds: 3 fresh frames, 1 unstable windows",
    )


class _PreviewClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def wait_for_service(self, timeout_sec):
        assert timeout_sec == 2.0
        return True

    def call_async(self, _request):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(done=lambda: True, result=lambda: self.response)


def _cli_for_preview(response=None, error=None):
    cli = object.__new__(LlmYoloCli)
    cli.preview_client = _PreviewClient(response=response, error=error)
    cli.session_id = "session"
    cli.preview_id = ""
    return cli


def test_missing_key_rejection_explains_where_key_set_must_run(capsys):
    response = SimpleNamespace(
        accepted=False,
        preview_id="",
        preview_json="",
        message=f"Preview rejected: {deepseek_credentials.MISSING_MESSAGE}",
    )

    _cli_for_preview(response=response).preview("抓取最近的 bolt")
    output = capsys.readouterr().out

    assert "Enter `key set` at this llm-arm> prompt" in output
    assert "Do not run `key set` at the Linux" in output


def test_preview_transport_error_does_not_terminate_cli(capsys):
    _cli_for_preview(error=RuntimeError("transport unavailable")).preview("抓取 cube")

    output = capsys.readouterr().out
    assert "Preview request failed: transport unavailable" in output
    assert "The CLI is still active" in output


def test_key_set_saves_from_cli_prompt(monkeypatch, capsys):
    cli = object.__new__(LlmYoloCli)
    saved = []
    monkeypatch.setattr("llm_arm_control.llm_yolo_cli.getpass.getpass", lambda _prompt: "new-key")
    monkeypatch.setattr(deepseek_credentials, "set_deepseek_api_key", saved.append)

    cli.key_command(["key", "set"])

    assert saved == ["new-key"]
    assert "saved to GNOME Keyring" in capsys.readouterr().out


def test_cancelled_key_entry_keeps_cli_alive(monkeypatch, capsys):
    cli = object.__new__(LlmYoloCli)

    def cancel(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("llm_arm_control.llm_yolo_cli.getpass.getpass", cancel)

    cli.key_command(["key", "set"])

    assert "Key entry cancelled; the CLI is still active" in capsys.readouterr().out
