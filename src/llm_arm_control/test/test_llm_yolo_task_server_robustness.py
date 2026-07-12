import threading
from types import SimpleNamespace
import json

import pytest
from rclpy.action import GoalResponse

from llm_arm_control.llm_yolo_task_server import LlmYoloTaskServer, ResolvedCandidate
from llm_arm_control.task_logic import SafetyState


class _Abort:
    def __init__(self, blocked=False):
        self.blocked = blocked

    def is_set(self):
        return self.blocked


class _RecoveryAbort(_Abort):
    def __init__(self, released=False, stopped=False, active=False, message=""):
        super().__init__(released or stopped)
        self.released = released
        self.stopped = stopped
        self.active = active
        self.message = message

    def recovery_released(self):
        return self.released

    def is_stop_requested(self):
        return self.stopped

    def recovery_active(self):
        return self.active

    def recovery_message(self):
        return self.message


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def _server(**values):
    server = object.__new__(LlmYoloTaskServer)
    server._lock = threading.RLock()
    for name, value in values.items():
        setattr(server, name, value)
    return server


def _candidate(x, class_name="elongated_object", stamp=1):
    return ResolvedCandidate(0, class_name, 0.9, (10.0, 20.0), (x, 0.0, 0.1), 0.0, stamp, 1.0)


def test_preview_reports_both_safety_sources_without_calling_llm():
    server = _server(
        _state="IDLE",
        _safety=SafetyState(blocked=True),
        abort=_Abort(True),
    )
    response = SimpleNamespace()
    result = server._preview_command(
        SimpleNamespace(instruction="home", session_id="test"), response
    )

    assert result.accepted is False
    assert "safety state is blocked" in result.message
    assert "abort manager is set" in result.message


def test_goal_rejection_keeps_detailed_warning():
    logger = _Logger()
    server = _server(
        _state="IDLE",
        _safety=SafetyState(blocked=True),
        abort=_Abort(True),
        _previews={},
        _pending_place=None,
        _held_source=None,
        _consumed_previews=frozenset(),
        get_logger=lambda: logger,
    )

    result = server._goal_callback(SimpleNamespace(preview_id="missing", session_id="test"))

    assert result == GoalResponse.REJECT
    assert "safety state is blocked; abort manager is set" in logger.warnings[0]


@pytest.mark.parametrize(
    ("label", "limit", "class_name", "accepted", "rejected"),
    [
        ("pick target", 0.02, "elongated_object", 0.019, 0.021),
        ("box", 0.05, "box", 0.049, 0.051),
    ],
)
def test_preview_revalidation_uses_role_specific_shift_limit(
    label, limit, class_name, accepted, rejected
):
    previous = _candidate(0.0, class_name)
    server = _server(_fresh_match=lambda _old: _candidate(accepted, class_name, 2))
    assert server._revalidate_candidate(previous, label, limit).xyz[0] == accepted

    server._fresh_match = lambda _old: _candidate(rejected, class_name, 3)
    with pytest.raises(ValueError, match=f"{limit * 1000:g} mm"):
        server._revalidate_candidate(previous, label, limit)


def test_box_timeout_reports_observation_counts():
    server = _server(
        box_retarget_timeout_sec=0.0,
        box_sample_count=5,
        _fresh_match=lambda _old: None,
    )

    candidate, message = server._collect_box_samples(_candidate(0.0, "box"))

    assert candidate is None
    assert "0 fresh frames" in message
    assert "0 unstable windows" in message


def test_zero_frame_failure_enables_cached_box_confirmation():
    source = _candidate(0.1)
    destination = _candidate(0.25, "box")
    server = _server(
        _cached_box_fallback_available=False,
        _held_source=source,
        _pending_place=None,
        _state="EXECUTING",
        box_retarget_timeout_sec=0.0,
        box_sample_count=5,
        _fresh_match=lambda _old: None,
    )

    ok, _message = server._execute_place_tail(source, destination)

    assert not ok
    assert server._state == "HOLDING_RECOVERY"
    assert server._cached_box_fallback_available


def test_unstable_box_frames_do_not_enable_cached_confirmation():
    source = _candidate(0.1)
    destination = _candidate(0.25, "box")
    server = _server(
        _cached_box_fallback_available=False,
        _held_source=source,
        _pending_place=None,
        _state="EXECUTING",
        _collect_box_samples=lambda *_args, **_kwargs: (
            None,
            "box was not stable within 5 seconds: 3 fresh frames, "
            "1 unstable windows; requires 5 stable frames",
        ),
    )

    ok, _message = server._execute_place_tail(source, destination)

    assert not ok
    assert not server._cached_box_fallback_available


def test_retry_preview_exposes_cached_box_pose_for_manual_confirmation():
    source = _candidate(0.1)
    destination = _candidate(0.25, "box", stamp=1_000_000_000)
    server = _server(
        _pending_place=(source, destination),
        _held_source=source,
        _state="HOLDING_RECOVERY",
        _safety=SafetyState(),
        _cached_box_fallback_available=True,
        _fresh_match=lambda _old: None,
        box_max_shift_m=0.05,
        _place_preview_poses=lambda *_args: {"approach_box": object(), "release": object()},
        _check_pose=lambda _pose: None,
        preview_max_age_sec=15.0,
        base_frame="base_link",
        _place_public_steps=lambda *_args: [{"type": "re_detect_box"}],
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=6_000_000_000)
        ),
        _previews={},
        _motion_block_reason_locked=lambda: "",
    )
    response = SimpleNamespace(
        accepted=False, status="rejected", preview_id="", preview_json="", message=""
    )

    server._retry_preview("session", response)
    payload = json.loads(response.preview_json)

    assert response.accepted
    assert payload["actions"] == [{"type": "retry_place", "use_cached_box_pose": True}]
    assert payload["steps"][0]["type"] == "manual_cached_box_pose"
    assert payload["steps"][0]["detection_age_sec"] == pytest.approx(5.0)
    record = server._previews[response.preview_id]
    assert record.enriched_actions[0]["use_cached_box_pose"]


def test_confirmed_cached_box_pose_skips_redetection_and_runs_place_tail():
    source = _candidate(0.1)
    destination = _candidate(0.25, "box")
    calls = []
    server = _server(
        _cached_box_fallback_available=True,
        _pending_place=(source, destination),
        _held_source=source,
        _fresh_match=lambda _old: None,
        box_retarget_threshold_m=0.01,
        _place_preview_poses=lambda *_args: {"approach_box": "approach", "release": "release"},
        _move_pose=lambda _pose, name, *_args: calls.append(name) or True,
        _apply_gripper=lambda width: calls.append(f"gripper:{width}") or True,
        _go_home=lambda: calls.append("home") or True,
        open_finger_position=0.0305,
    )

    ok, message = server._execute_place_tail(
        source, destination, use_cached_destination=True
    )

    assert ok
    assert "manual_cached_pose" in message
    assert calls[:2] == ["approach_box", "release"]
    assert "home" in calls
    assert not server._cached_box_fallback_available


def test_cached_confirmation_is_rejected_if_box_reappears_shifted():
    source = _candidate(0.1)
    destination = _candidate(0.25, "box")
    server = _server(
        _cached_box_fallback_available=True,
        _pending_place=(source, destination),
        _held_source=source,
        _fresh_match=lambda _old: _candidate(0.27, "box", stamp=2),
        box_retarget_threshold_m=0.01,
    )

    ok, message = server._execute_place_tail(
        source, destination, use_cached_destination=True
    )

    assert not ok
    assert "moved beyond" in message


def test_deepseek_network_call_does_not_hold_session_lock():
    class TrackingLock:
        depth = 0

        def __enter__(self):
            self.depth += 1

        def __exit__(self, *_args):
            self.depth -= 1

    lock = TrackingLock()

    class Client:
        def chat(self, _messages, _model):
            assert lock.depth == 0
            return '{"actions":[{"type":"home"}]}'

    pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    server = _server(
        _sessions={},
        _held_source=None,
        _current_pose=lambda: SimpleNamespace(pose=pose),
        _deepseek=lambda: Client(),
        deepseek_model="deepseek-chat",
        base_frame="base_link",
        pick_classes=frozenset({"elongated_object", "cube"}),
        place_classes=frozenset({"box"}),
    )
    server._lock = lock

    server._llm_plan("session", "home", [])

    assert lock.depth == 0
    assert len(server._sessions["session"]) == 2


def test_successful_recovery_clears_holding_and_safety_state():
    server = _server(
        abort=_RecoveryAbort(released=True),
        _held_source=object(),
        _pending_place=(object(), object()),
        _cached_box_fallback_available=True,
        _previews={"preview": object()},
        _reset_failed=True,
        _safety=SafetyState(blocked=True, command="reset"),
        _execution_active=False,
        _state="RESETTING",
    )

    server._recovery_complete(True)

    assert server._state == "IDLE"
    assert not server._safety.blocked
    assert server._held_source is None
    assert server._pending_place is None
    assert not server._cached_box_fallback_available


def test_space_interrupted_recovery_stays_stopped_and_preserves_held_object():
    held = object()
    pending = (held, object())
    server = _server(
        abort=_RecoveryAbort(released=False, stopped=True),
        _held_source=held,
        _pending_place=pending,
        _cached_box_fallback_available=True,
        _previews={"preview": object()},
        _reset_failed=False,
        _safety=SafetyState(blocked=True, command="stop"),
        _execution_active=False,
        _state="RESETTING",
    )

    server._recovery_complete(False)

    assert server._state == "STOPPED"
    assert server._held_source is held
    assert server._pending_place == pending
    assert not server._reset_failed


def test_home_failure_after_open_clears_holding_and_reports_reset_failed():
    server = _server(
        abort=_RecoveryAbort(released=True),
        _held_source=object(),
        _pending_place=(object(), object()),
        _cached_box_fallback_available=True,
        _previews={},
        _reset_failed=False,
        _safety=SafetyState(blocked=True, command="reset"),
        _execution_active=False,
        _state="RESETTING",
    )

    server._recovery_complete(False)

    assert server._state == "RESET_FAILED"
    assert server._reset_failed
    assert server._held_source is None
    assert server._pending_place is None


def test_status_exposes_recovery_progress_without_changing_existing_fields():
    server = _server(
        abort=_RecoveryAbort(active=True, message="returning Home"),
        _state="RESETTING",
        _pending_place=None,
        _held_source=None,
        _cached_box_fallback_available=False,
        _client_key=None,
        _current_frame=lambda: None,
        _metadata=lambda _frame: [],
    )
    response = SimpleNamespace(success=False, message="")

    server._status(None, response)
    payload = json.loads(response.message)

    assert payload["state"] == "RESETTING"
    assert payload["recovery_active"] is True
    assert payload["reset_message"] == "returning Home"
