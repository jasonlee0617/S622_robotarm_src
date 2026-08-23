import threading
import time
from collections import deque
from types import SimpleNamespace
import json

import numpy as np
import pytest
from rclpy.action import GoalResponse

from llm_arm_control_nodes.llm_control_task_server import LlmControlTaskServer, PreviewRecord
from yolo_perception_nodes.llm_yolo_perception import ResolvedCandidate, RgbdPerception
from llm_arm_control_nodes.task_logic import (
    ClarificationRequired,
    SafetyState,
    TaskPlan,
    TaskPreview,
)
from llm_arm_control_nodes.task.llm_control_state_machine import LlmControlTaskStateMachine
from llm_arm_control_nodes.task.llm_control_state_machine import LlmGraspnetState


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

    def error(self, message):
        self.warnings.append(message)

    def info(self, _message):
        pass


def _server(**values):
    server = object.__new__(LlmControlTaskServer)
    server._lock = threading.RLock()
    server.active_mode = "yolo"
    server._graspnet_state = LlmGraspnetState.WAIT_G.value
    server._graspnet_g_requested = False
    server._graspnet_last_error = ""
    server._mode_switch_error = ""
    server.use_continuous_yolo = True
    for name, value in values.items():
        setattr(server, name, value)
    server._state_machine = LlmControlTaskStateMachine(server)
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
    monkeypatch.setattr("yolo_perception_nodes.llm_yolo_perception.time.monotonic", lambda: next(ticks))

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
        _held_source=None,
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


@pytest.mark.parametrize(
    ("instruction", "action"),
    [
        ("抓取图像中右侧的bolt", {"type": "pick", "source_index": 0}),
        ("放置到盒子", {"type": "place", "source_index": 0}),
    ],
)
def test_visual_preview_creates_task_preview(instruction, action):
    server = _server(
        _state="IDLE",
        _safety=SafetyState(),
        _previews={},
        _held_source=None,
        preview_max_age_sec=30.0,
        base_frame="base_link",
        _motion_block_reason_locked=lambda: "",
        perception=SimpleNamespace(
            current_frame=lambda: object(),
            wait_for_planning_metadata=lambda: [],
        ),
        _llm_plan=lambda *_args: TaskPlan((action,)),
        _enrich_plan=lambda _plan: ([], [], []),
        _set_llm_yolo_inference=lambda _enabled: True,
    )
    response = SimpleNamespace(
        accepted=False, status="", preview_id="", preview_json="", message=""
    )

    server._preview_command(SimpleNamespace(instruction=instruction, session_id="test"), response)

    assert response.accepted is True
    assert response.preview_id in server._previews
    assert json.loads(response.preview_json)["instruction"] == instruction


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

    metadata = perception.planning_metadata({
        "yolo": SimpleNamespace(yolov8_inference=[item]),
        "depth": np.zeros((480, 640), dtype=np.float32),
    })

    assert metadata[0]["center_uv"] == [10.0, 20.0]
    assert metadata[0]["base_xyz"] == [0.3, 0.0, 0.1]
    assert metadata[0]["depth_inlier_ratio"] == 1.0
    assert metadata[0]["image_size"] == [640, 480]


def test_workspace_limits_only_xy():
    server = _server(workspace_min_xy=(-0.9, -0.9), workspace_max_xy=(0.9, 0.9))

    assert server._workspace_ok((0.1, 0.4, -5.0))
    assert server._workspace_ok((0.1, 0.4, 5.0))
    assert not server._workspace_ok((0.91, 0.4, 0.1))
    assert not server._workspace_ok((0.1, -0.91, 0.1))


def test_graspnet_mode_entry_requires_llm_idle_or_holding():
    logger = _Logger()
    server = _server(
        _execution_active=False,
        _state="PREGRASP_POSE",
        active_mode="yolo",
        _previews={"pending": object()},
        get_logger=lambda: logger,
    )

    server._on_active_mode(SimpleNamespace(data="graspnet"))

    assert server.active_mode == "yolo"
    assert logger.warnings

    held = object()
    server._state = "HOLDING"
    server._held_source = held
    calls = []
    server._set_llm_yolo_inference = lambda enabled, *, force=False: calls.append(("inference", enabled, force)) or True
    server._release_llm_yolo_gpu = lambda: calls.append(("release",)) or True
    server._on_active_mode(SimpleNamespace(data="graspnet"))

    assert server.active_mode == "graspnet"
    assert server._held_source is held
    assert server._previews == {}
    assert calls == [("inference", False, True), ("release",)]


def test_graspnet_mode_stays_yolo_when_gpu_release_fails():
    logger = _Logger()
    server = _server(
        _execution_active=False,
        _state="IDLE",
        _previews={},
        _held_source=None,
        get_logger=lambda: logger,
    )
    server._set_llm_yolo_inference = lambda _enabled, *, force=False: True
    server._release_llm_yolo_gpu = lambda: False

    server._on_active_mode(SimpleNamespace(data="graspnet"))

    assert server.active_mode == "yolo"
    assert "release" in server._mode_switch_error


def test_duplicate_mode_switch_is_ignored_while_gpu_handoff_is_active():
    logger = _Logger()
    server = _server(
        _execution_active=False,
        _state="IDLE",
        active_mode="yolo",
        _mode_switch_active=True,
        get_logger=lambda: logger,
    )
    server._set_llm_yolo_inference = lambda *_args, **_kwargs: pytest.fail("must not hand off twice")

    server._on_active_mode(SimpleNamespace(data="graspnet"))

    assert server.active_mode == "yolo"
    assert logger.warnings == ["Ignoring duplicate LLM mode switch request."]


def test_graspnet_g_command_is_accepted_only_in_wait_g():
    server = _server(active_mode="graspnet")

    server._on_llm_motion_command(SimpleNamespace(data="g"))

    assert server._graspnet_state == LlmGraspnetState.COMPUTE.value
    assert not server._graspnet_g_requested
    server._on_llm_motion_command(SimpleNamespace(data="g"))
    assert not server._graspnet_g_requested


def test_graspnet_g_is_rejected_while_holding():
    logger = _Logger()
    server = _server(
        active_mode="graspnet", _held_source=object(), get_logger=lambda: logger
    )

    server._on_llm_motion_command(SimpleNamespace(data="g"))

    assert not server._graspnet_g_requested
    assert logger.warnings == ["Ignoring GraspNet request while an object is held."]


def test_mode_switch_cannot_overtake_an_accepted_graspnet_request():
    logger = _Logger()
    server = _server(
        active_mode="graspnet",
        _execution_active=False,
        _state="IDLE",
        get_logger=lambda: logger,
    )

    server._on_llm_motion_command(SimpleNamespace(data="g"))
    server._on_active_mode(SimpleNamespace(data="yolo"))

    assert server.active_mode == "graspnet"
    assert server._graspnet_state == LlmGraspnetState.COMPUTE.value
    assert logger.warnings == ["Ignoring YOLO mode entry outside LLM GraspNet WAIT_G."]


def test_motion_failure_stops_without_starting_recovery():
    calls = []
    server = _server(
        abort=SimpleNamespace(
            request_abort=lambda reason, command: calls.append((reason, command)) or True,
            cancel_all_motion_now=lambda: calls.append("cancel"),
        ),
        _state="EXECUTING",
    )

    server._stop_for_motion_failure("close_gripper_failed")

    assert calls == [("close_gripper_failed", "stop"), "cancel"]
    assert server._state == "STOPPED"


def test_yolo_mode_releases_graspnet_gpu_before_enabling_yolo():
    logger = _Logger()
    calls = []
    server = _server(
        active_mode="graspnet",
        _execution_active=False,
        _state="IDLE",
        _previews={},
        _held_source=None,
        get_logger=lambda: logger,
    )
    server._release_graspnet_gpu = lambda: calls.append("release_graspnet") or True
    server._set_llm_yolo_inference = lambda enabled, *, force=False: calls.append(("yolo", enabled, force)) or True

    server._on_active_mode(SimpleNamespace(data="yolo"))

    assert server.active_mode == "yolo"
    assert calls == ["release_graspnet", ("yolo", True, True)]


def test_yolo_mode_stays_graspnet_when_gpu_release_fails():
    logger = _Logger()
    server = _server(
        active_mode="graspnet",
        _execution_active=False,
        _state="IDLE",
        _previews={},
        get_logger=lambda: logger,
    )
    server._release_graspnet_gpu = lambda: False
    server._set_llm_yolo_inference = lambda *_args, **_kwargs: pytest.fail("YOLO must not be enabled")

    server._on_active_mode(SimpleNamespace(data="yolo"))

    assert server.active_mode == "graspnet"
    assert "GraspNet" in server._mode_switch_error


def test_deterministic_bolt_pick_bypasses_deepseek():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    server = _server(
        base_frame="base_link",
        pick_classes=frozenset({"elongated_object", "cube", "stone"}),
        place_classes=frozenset({"box"}),
        _sessions={},
        _held_source=None,
        _current_pose=lambda: SimpleNamespace(pose=pose),
        _deepseek=lambda: pytest.fail("deterministic visual selection must not call DeepSeek"),
    )
    metadata = [
        {"index": 0, "class_name": "elongated_object", "center_uv": [10.0, 10.0],
         "image_size": [100, 100], "base_xyz": [0.1, 0.0, 0.1]},
        {"index": 1, "class_name": "elongated_object", "center_uv": [90.0, 10.0],
         "image_size": [100, 100], "base_xyz": [0.8, 0.0, 0.1]},
        {"index": 2, "class_name": "cube", "center_uv": [50.0, 50.0],
         "image_size": [100, 100], "base_xyz": [0.2, 0.0, 0.1]},
    ]

    plan = server._llm_plan("session", "抓取距离机械臂最近的 bolt", metadata)

    assert plan.actions == ({"type": "pick", "source_index": 0},)


def test_depth_candidate_is_transformed_to_base(monkeypatch):
    monkeypatch.setattr(
        "yolo_perception_nodes.llm_yolo_perception.robust_center3d_from_obb_depth",
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


def test_yolo_grasp_profile_and_offsets_define_pick_and_place_poses():
    source = _candidate(0.2, class_name="stone")
    destination = _candidate(0.4, class_name="box")
    server = _server(
        grasp_profiles={"stone": (0.0, -180.0, -45.0)},
        grasp_above=0.04,
        grasp_offset=0.01,
        place_offset=0.08,
        descend_to_box=0.04,
        _pose_from_xyz_quat=lambda xyz, quat: (xyz, quat),
    )

    pick = server._pick_preview_poses(source)
    place = server._place_preview_poses(source, destination)

    assert pick["approach_pick"][0][2] == pytest.approx(0.14)
    assert pick["grasp"][0][2] == pytest.approx(0.11)
    assert pick["carry"][0][2] == pytest.approx(0.18)
    assert place["approach_box"][0][2] == pytest.approx(0.18)
    assert place["release"][0][2] == pytest.approx(0.14)


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
        _held_source=None,
        get_logger=lambda: logger,
    )

    result = server._goal_callback(SimpleNamespace(preview_id="missing", session_id="test"))

    assert result == GoalResponse.REJECT
    assert "safety state is blocked; abort manager is set" in logger.warnings[0]


def test_goal_accepts_ready_preview():
    plan = TaskPlan(({"type": "pick", "source_index": 0},))
    record = PreviewRecord(
        TaskPreview("ready", plan, time.monotonic(), 15.0),
        "session",
        "抓取右侧的物体",
        [{"type": "pick", "source": object()}],
        0,
        {},
    )
    server = _server(
        _state="PREVIEW_READY",
        _safety=SafetyState(),
        _held_source=None,
        _previews={"ready": record},
        _motion_block_reason_locked=lambda: "",
    )

    assert server._goal_callback(SimpleNamespace(preview_id="ready", session_id="session")) == GoalResponse.ACCEPT


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
    )

    server._prune_previews_locked(now)

    assert server._previews == {"ready": ready}
    assert server._take_preview_locked("ready") is ready
    assert server._take_preview_locked("ready") is None


def test_pregrasp_motion_uses_the_configured_cartesian_pose():
    calls = []
    pose = object()
    server = _server(
        pregrasp_pose=pose,
        _move_pose=lambda target, name: calls.append((target, name)) or True,
    )

    assert server._move_to_pregrasp_pose()
    assert calls == [(pose, "Move to pregrasp pose")]


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


def test_space_interrupted_recovery_stays_stopped_and_preserves_held_object():
    held = object()
    server = _server(
        abort=_RecoveryAbort(released=False, stopped=True),
        _held_source=held,
        _previews={"preview": object()},
        _reset_failed=False,
        _safety=SafetyState(blocked=True, command="stop"),
        _execution_active=False,
        _state="RESETTING",
    )

    server._recovery_complete(False)

    assert server._state == "STOPPED"
    assert server._held_source is held
    assert not server._reset_failed


def test_home_failure_after_open_clears_holding_and_reports_reset_failed():
    server = _server(
        abort=_RecoveryAbort(released=True),
        _held_source=object(),
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


def test_status_exposes_recovery_progress_without_changing_existing_fields():
    server = _server(
        abort=_RecoveryAbort(active=True, message="returning Home"),
        _state="RESETTING",
        _held_source=None,
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
