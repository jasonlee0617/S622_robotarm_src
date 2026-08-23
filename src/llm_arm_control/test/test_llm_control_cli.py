import errno
import json
import os
import pty
import select
import subprocess
import sys
from types import SimpleNamespace

import pytest
from std_msgs.msg import String


pytest.importorskip("rclpy")

from llm_arm_control_nodes.llm_control_cli import (  # noqa: E402
    LlmControlCli,
    compact_preview,
    deepseek_credentials,
    execution_key_effect,
)


def test_space_and_ctrl_c_stop_and_cancel_action():
    assert execution_key_effect(" ") == ("stop", True)
    assert execution_key_effect("\x03") == ("stop", True)


def test_pregrasp_reset_does_not_race_with_action_cancel():
    assert execution_key_effect("h") == ("reset", False)
    assert execution_key_effect("H") == ("reset", False)


def test_other_execution_keys_are_ignored():
    assert execution_key_effect("y") == ("", False)
    assert execution_key_effect("g") == ("", False)


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_cli_relays_stop_and_reset_without_processing_g():
    cli = object.__new__(LlmControlCli)
    cli.event_pub = _Publisher()
    cli.abort_pub = _Publisher()

    cli._relay_command(String(data="stop"))
    cli._relay_command(String(data="reset"))
    cli._relay_command(String(data="g"))

    assert [message.data for message in cli.event_pub.messages] == ["stop", "stop"]
    assert [message.data for message in cli.abort_pub.messages] == [True]


def test_cli_mode_switch_requires_idle_or_holding_and_wait_g(capsys):
    cli = object.__new__(LlmControlCli)
    cli.active_mode = "yolo"
    cli.preview_id = "pending"
    cli.mode_pub = _Publisher()
    cli._server_status = lambda: {"state": "PREGRASP_POSE", "graspnet_state": "WAIT_G", "active_mode": "yolo"}

    cli.set_mode(["mode", "graspnet"])

    assert cli.active_mode == "yolo"
    assert not cli.mode_pub.messages
    assert "only when /llm_control/status reports IDLE or HOLDING" in capsys.readouterr().out

    cli._server_status = lambda: {"state": "HOLDING", "graspnet_state": "WAIT_G", "active_mode": "graspnet"}
    cli.set_mode(["mode", "graspnet"])
    assert cli.active_mode == "graspnet"
    assert cli.mode_pub.messages[-1].data == "graspnet"

    cli._server_status = lambda: {"state": "IDLE", "graspnet_state": "COMPUTE", "active_mode": "graspnet"}
    cli.set_mode(["mode", "yolo"])
    assert cli.active_mode == "graspnet"


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (
            [{"type": "pick", "source_index": 0}],
            ["pick cube: base_link x=0.200, y=0.446, z=0.025, yaw=1.493 rad"],
        ),
        (
            [{"type": "place", "destination_index": 1}],
            [
                "holding cube: base_link x=0.200, y=0.446, z=0.025, yaw=1.493 rad",
                "box box: base_link x=0.300, y=0.400, z=0.040, yaw=0.000 rad",
            ],
        ),
        (
            [{"type": "pick_place", "source_index": 0, "destination_index": 1}],
            [
                "pick cube: base_link x=0.200, y=0.446, z=0.025, yaw=1.493 rad",
                "box box: base_link x=0.300, y=0.400, z=0.040, yaw=0.000 rad",
            ],
        ),
    ],
)
def test_compact_preview_shows_only_involved_base_link_poses(actions, expected):
    preview = {
        "actions": actions,
        "detections": [
            {"index": 0, "class_name": "cube", "base_xyz": [0.2, 0.446, 0.025], "yaw": 1.493, "confidence": 0.9},
            {"index": 1, "class_name": "box", "base_xyz": [0.3, 0.4, 0.04], "yaw": 0.0, "confidence": 0.8},
        ],
        "steps": [{"type": "grasp"}],
    }

    output = compact_preview(json.dumps(preview))

    assert output.splitlines() == expected
    assert "confidence" not in output
    assert "steps" not in output


def test_compact_preview_for_empty_place_shows_only_the_box():
    preview = {
        "actions": [{"type": "place", "destination_index": 1}],
        "detections": [
            {"index": 1, "class_name": "box", "base_xyz": [0.3, 0.4, 0.04], "yaw": 0.0},
        ],
    }

    assert compact_preview(json.dumps(preview)) == (
        "box box: base_link x=0.300, y=0.400, z=0.040, yaw=0.000 rad"
    )


def test_cli_preview_does_not_print_the_full_service_json(capsys):
    response = SimpleNamespace(
        accepted=True,
        preview_id="preview",
        preview_json=json.dumps({
            "actions": [{"type": "pick", "source_index": 0}],
            "detections": [{
                "index": 0, "class_name": "cube", "base_xyz": [0.2, 0.4, 0.03],
                "yaw": 0.0, "confidence": 0.9,
            }],
            "steps": [{"type": "grasp"}],
        }),
        message="Preview ready.",
    )

    _cli_for_preview(response=response).preview("抓取cube")

    output = capsys.readouterr().out
    assert "pick cube: base_link x=0.200, y=0.400, z=0.030, yaw=0.000 rad" in output
    assert "confidence" not in output
    assert '"steps"' not in output


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
    cli = object.__new__(LlmControlCli)
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

    assert "Enter `key set` at this llm-control> prompt" in output
    assert "Do not run `key set` at the Linux" in output


def test_preview_transport_error_does_not_terminate_cli(capsys):
    _cli_for_preview(error=RuntimeError("transport unavailable")).preview("抓取 cube")

    output = capsys.readouterr().out
    assert "Preview request failed: transport unavailable" in output
    assert "The CLI is still active" in output


def test_key_set_saves_from_cli_prompt(monkeypatch, capsys):
    cli = object.__new__(LlmControlCli)
    saved = []
    monkeypatch.setattr("llm_arm_control_nodes.llm_control_cli.getpass.getpass", lambda _prompt: "new-key")
    monkeypatch.setattr(deepseek_credentials, "set_deepseek_api_key", saved.append)

    cli.key_command(["key", "set"])

    assert saved == ["new-key"]
    assert "saved to GNOME Keyring" in capsys.readouterr().out


def test_cancelled_key_entry_keeps_cli_alive(monkeypatch, capsys):
    cli = object.__new__(LlmControlCli)

    def cancel(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("llm_arm_control_nodes.llm_control_cli.getpass.getpass", cancel)

    cli.key_command(["key", "set"])

    assert "Key entry cancelled; the CLI is still active" in capsys.readouterr().out


def test_terminal_readline_erases_unicode_before_submitting():
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import readline; line = input('llm-control> '); print('RESULT=' + line)",
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    try:
        assert select.select([master], [], [], 3.0)[0]
        os.read(master, 1024)
        typed = "抓取石头"
        os.write(master, typed.encode("utf-8") + b"\x7f" * len(typed) + b"cube\n")
        output = bytearray()
        while process.poll() is None:
            assert select.select([master], [], [], 3.0)[0]
            try:
                output.extend(os.read(master, 4096))
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
        process.wait(timeout=3.0)
        assert b"RESULT=cube" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
