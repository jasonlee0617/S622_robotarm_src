import threading
import time
from collections import deque
from types import SimpleNamespace
import json

import numpy as np
import pytest
from rclpy.action import GoalResponse

from llm_arm_control_nodes.llm_yolo_task_server import LlmYoloTaskServer, PreviewRecord
from llm_arm_control_nodes.perception import ResolvedCandidate, RgbdPerception
from llm_arm_control_nodes.task_logic import (
    ClarificationRequired,
    SafetyState,
    TaskPlan,
    TaskPreview,
)


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


def _perception(**values):
    perception = object.__new__(RgbdPerception)
    perception._lock = threading.RLock()
    for name, value in values.items():
        setattr(perception, name, value)
    return perception


def _candidate(x, class_name="elongated_object", stamp=1):
    return ResolvedCandidate(0, class_name, 0.9, (10.0, 20.0), (x, 0.0, 0.1), 0.0, stamp, 1.0)


def _header(sec, nanosec=0):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def test_rgbd_matching_keeps_yolo_history_when_depth_processing_lags():
    older_yolo = SimpleNamespace(header=_header(1, 0))
    newer_yolo = SimpleNamespace(header=_header(1, 100_000_000))
    matching_depth = (_header(1, 16_000_000), object())
    perception = _perception(
        _yolo_frames=deque([older_yolo, newer_yolo], maxlen=20),
        _depth_frames=deque([matching_depth], maxlen=20),
        _active_frame=None,
        rgb_depth_tolerance_sec=0.05,
    )

    perception._activate_frame_locked()

    assert perception._active_frame["yolo"] is older_yolo
    assert perception._active_frame["sync_delta_sec"] == pytest.approx(0.016)


def test_same_rgbd_pair_does_not_refresh_freshness(monkeypatch):
    yolo = SimpleNamespace(header=_header(2, 0))
    depth = (_header(2, 10_000_000), object())
    perception = _perception(
        _yolo_frames=deque([yolo], maxlen=20),
        _depth_frames=deque([depth], maxlen=20),
        _active_frame=None,
        rgb_depth_tolerance_sec=0.05,
    )
    ticks = iter((10.0, 20.0))
    monkeypatch.setattr("llm_arm_control_nodes.perception.time.monotonic", lambda: next(ticks))

    perception._activate_frame_locked()
    perception._activate_frame_locked()

    assert perception._active_frame["received_monotonic"] == 10.0


def test_visual_preview_rejects_missing_rgbd_before_calling_llm():
    def unavailable():
        raise ClarificationRequired("Vision input unavailable: no synchronized RGB-D frame.")

    server = _server(
        _state="IDLE",
        _safety=SafetyState(),
        _previews={},
        _lock=threading.RLock(),
        _motion_block_reason_locked=lambda: "",
        perception=SimpleNamespace(
            current_frame=lambda: None,
            wait_for_planning_metadata=unavailable,
            metadata=lambda _frame: [],
        ),
        _llm_plan=lambda *_args: pytest.fail("LLM must not be called without RGB-D"),
    )
    response = SimpleNamespace(
        accepted=False, status="", preview_id="", preview_json="", message=""
    )

    server._preview_command(
        SimpleNamespace(instruction="抓取 cube", session_id="test"), response
    )

    assert response.status == "clarification_required"
    assert "Vision input unavailable" in response.message


def test_planning_metadata_includes_base_coordinates_for_spatial_selection():
    resolved = _candidate(0.3)
    item = SimpleNamespace(
        class_name="elongated_object",
        confidence=0.9,
        coordinates=[0.0, 10.0, 20.0, 10.0, 20.0, 30.0, 0.0, 30.0],
    )
    perception = _perception(
        _resolve_detection=lambda *_args: resolved,
    )

    metadata = perception.planning_metadata(
        {"yolo": SimpleNamespace(yolov8_inference=[item])}
    )

    assert metadata[0]["center_uv"] == [10.0, 20.0]
    assert metadata[0]["base_xyz"] == [0.3, 0.0, 0.1]
    assert metadata[0]["depth_inlier_ratio"] == 1.0


def test_depth_candidate_is_transformed_to_base(monkeypatch):
    monkeypatch.setattr(
        "llm_arm_control_nodes.perception.robust_center3d_from_obb_depth",
        lambda **_kwargs: (np.array([0.1, 0.2, 1.0]), 0.9),
    )
    transformed = iter((
        SimpleNamespace(point=SimpleNamespace(x=0.3, y=0.4, z=0.5)),
        SimpleNamespace(point=SimpleNamespace(x=0.4, y=0.4, z=0.5)),
    ))
    perception = _perception(
        _camera_intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
        _transform_point=lambda *_args: next(transformed),
        node=SimpleNamespace(get_logger=lambda: _Logger()),
    )
    item = SimpleNamespace(class_name="elongated_object", confidence=0.8)
    points = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]])
    frame = {"depth": object(), "yolo": SimpleNamespace(header=_header(1)), "stamp_ns": 1}

    resolved = perception._resolve_detection(2, item, points, points.mean(axis=0), frame)

    assert resolved.index == 2
    assert resolved.xyz == pytest.approx((0.3, 0.4, 0.5))
    assert resolved.yaw == pytest.approx(0.0)
    assert resolved.depth_inlier_ratio == pytest.approx(0.9)


def test_fresh_match_selects_nearest_same_class_candidate():
    old = _candidate(0.0)
    items = [SimpleNamespace(class_name="elongated_object") for _ in range(2)]
    candidates = iter((_candidate(0.04, stamp=2), _candidate(0.01, stamp=2)))
    perception = _perception(
        current_frame=lambda: object(),
        _detections=lambda _frame: [(0, items[0], None, None), (1, items[1], None, None)],
        _resolve_detection=lambda *_args: next(candidates),
    )

    assert perception.fresh_match(old).xyz[0] == pytest.approx(0.01)


def test_preview_reports_both_safety_sources_without_calling_llm():
    server = _server(
        _state="IDLE",
        _safety=SafetyState(blocked=True),
        abort=_Abort(True),
        _previews={},
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
        get_logger=lambda: logger,
    )

    result = server._goal_callback(SimpleNamespace(preview_id="missing", session_id="test"))

    assert result == GoalResponse.REJECT
    assert "safety state is blocked; abort manager is set" in logger.warnings[0]


def test_preview_records_are_pruned_and_taken_once():
    now = time.monotonic()
    plan = TaskPlan(({"type": "home"},))
    ready = PreviewRecord(TaskPreview("ready", plan, now, 15.0), "s", "home", [], 0, {})
    expired = PreviewRecord(
        TaskPreview("expired", plan, now - 16.0, 15.0), "s", "home", [], 0, {}
    )
    server = _server(
        _previews={"ready": ready, "expired": expired},
        _state="PREVIEW_READY",
        _held_source=None,
        _pending_place=None,
    )

    server._prune_previews_locked(now)

    assert server._previews == {"ready": ready}
    assert server._take_preview_locked("ready") is ready
    assert server._take_preview_locked("ready") is None


def test_go_home_restores_yaml_motion_limits():
    calls = []
    arm = SimpleNamespace(max_velocity=0.2, max_acceleration=0.2)
    server = _server(
        moveit2_arm=arm,
        arm_max_velocity=0.07,
        arm_max_acceleration=0.04,
        home_joints=[0.0] * 6,
        motion=SimpleNamespace(
            move_to_joints=lambda joints, **kwargs: calls.append((joints, kwargs)) or True
        ),
    )

    assert server._go_home()
    assert arm.max_velocity == pytest.approx(0.07)
    assert arm.max_acceleration == pytest.approx(0.04)
    assert calls[0][1]["planning_client"] == "fairino"


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
    server = _server(
        perception=SimpleNamespace(
            fresh_match=lambda _old: _candidate(accepted, class_name, 2)
        )
    )
    assert server._revalidate_candidate(previous, label, limit).xyz[0] == accepted

    server.perception.fresh_match = lambda _old: _candidate(rejected, class_name, 3)
    with pytest.raises(ValueError, match=f"{limit * 1000:g} mm"):
        server._revalidate_candidate(previous, label, limit)


def test_box_timeout_reports_observation_counts():
    server = _server(
        box_retarget_timeout_sec=0.0,
        box_sample_count=5,
        perception=SimpleNamespace(fresh_match=lambda _old: None),
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
        perception=SimpleNamespace(fresh_match=lambda _old: None),
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
        perception=SimpleNamespace(fresh_match=lambda _old: None),
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
        perception=SimpleNamespace(fresh_match=lambda _old: None),
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
        perception=SimpleNamespace(
            fresh_match=lambda _old: _candidate(0.27, "box", stamp=2)
        ),
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
        perception=SimpleNamespace(diagnostics=lambda: {
            "fresh_detection": False,
            "candidate_count": 0,
            "yolo_buffer_count": 0,
            "depth_buffer_count": 0,
            "yolo_publisher_count": 0,
            "depth_publisher_count": 0,
            "rgb_depth_delta_sec": None,
            "camera_info_ready": False,
        }),
    )
    response = SimpleNamespace(success=False, message="")

    server._status(None, response)
    payload = json.loads(response.message)

    assert payload["state"] == "RESETTING"
    assert payload["recovery_active"] is True
    assert payload["reset_message"] == "returning Home"
