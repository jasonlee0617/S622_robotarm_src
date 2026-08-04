from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from hand_eye_calibration.collector import session as session_module
from hand_eye_calibration.collector.session import (
    CollectorExecutionSession,
    PASS,
    RETRYABLE,
    SESSION_FATAL,
)


def _session(specs=("a", "b", "c")):
    session = object.__new__(CollectorExecutionSession)
    session.sampling_cfg = SimpleNamespace(tool_delta_specs=specs)
    session.root_base_T_ee = object()
    session.last_safe_pose = object()
    session.results = []
    session.attempts = 0
    session.node = SimpleNamespace(
        _should_stop=lambda: False,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: object())),
        wait_for_step_continue=lambda prompt: True,
    )
    session._logger = lambda: SimpleNamespace(warn=lambda *_a, **_k: None, info=lambda *_a, **_k: None)
    session.geometry = SimpleNamespace(
        build_root_relative_candidate=lambda **kwargs: SimpleNamespace(
            idx=kwargs["idx"], description=f"candidate-{kwargs['idx']}", base_T_ee=object(), pose=object(), spec=kwargs["spec"],
        )
    )
    return session


def test_every_root_relative_action_is_attempted_without_recovery(monkeypatch):
    attempted = []
    session = _session()
    session._post_motion_observation = lambda *_args, **_kwargs: (PASS, "visible", object())
    session._try_record_sample = lambda *_args, **_kwargs: (PASS, "recorded")

    def move_candidate(_session, candidate):
        attempted.append(candidate.idx)
        return True, object(), "motion complete"

    monkeypatch.setattr(session_module, "move_candidate", move_candidate)
    assert session._collect_sequence()
    assert attempted == [1, 2, 3]
    assert [result[0] for result in session.results] == [1, 2, 3]


def test_failed_endpoint_recovers_once_then_continues(monkeypatch):
    session = _session(("a", "b"))
    session._post_motion_observation = lambda *_args, **_kwargs: (RETRYABLE, "marker lost", None)
    calls = []
    monkeypatch.setattr(session_module, "move_candidate", lambda *_args: (True, object(), "moved"))
    monkeypatch.setattr(session_module, "recover_last_safe", lambda *_args: (calls.append("recover") or (True, "restored")))
    assert session._collect_sequence()
    assert calls == ["recover", "recover"]


def test_fatal_endpoint_stops_sequence(monkeypatch):
    session = _session()
    calls = []
    monkeypatch.setattr(session_module, "move_candidate", lambda *_args: (calls.append("move") or (True, object(), "moved")))
    session._post_motion_observation = lambda *_args, **_kwargs: (SESSION_FATAL, "clock failed", None)
    assert not session._collect_sequence()
    assert calls == ["move"]


def test_step_mode_waits_between_actions(monkeypatch):
    session = _session()
    waits = []
    session.node.wait_for_step_continue = lambda prompt: (waits.append(prompt) or True)
    session._post_motion_observation = lambda *_a, **_k: (PASS, "visible", object())
    session._try_record_sample = lambda *_a, **_k: (PASS, "recorded")
    monkeypatch.setattr(session_module, "move_candidate", lambda _s, c: (True, object(), "moved"))
    assert session._collect_sequence()
    assert len(waits) == 3  # 3 个候选各等待一次 Enter
    assert "candidate 1/19" in waits[0]


def test_step_mode_stop_breaks_sequence(monkeypatch):
    session = _session(("a", "b"))
    waits = []
    session.node.wait_for_step_continue = lambda prompt: (waits.append(prompt) or len(waits) == 1)
    moved = []
    monkeypatch.setattr(session_module, "move_candidate", lambda _s, c: (moved.append(c.idx) or (True, object(), "moved")))
    session._post_motion_observation = lambda *_a, **_k: (PASS, "visible", object())
    session._try_record_sample = lambda *_a, **_k: (PASS, "recorded")
    assert session._collect_sequence()
    # 第一个候选前允许继续；第二个候选前 wait 返回 False → 中断。
    assert moved == [1]


def test_zero_ippe_clear_frame_requirement_keeps_stable_medoid(monkeypatch):
    session = object.__new__(CollectorExecutionSession)
    observation = SimpleNamespace(pnp_ambiguous=True, tvec=(0.0, 0.0, 0.3), image_stamp_ns=1)
    stable = SimpleNamespace(observations=(observation,), latest_observation=observation)
    recorded = []
    session.sampling_cfg = SimpleNamespace(ippe_min_non_ambiguous_frames=0)
    session.sample_manager = SimpleNamespace(
        diverse=lambda *_: (True, "new SE(3) motion"),
        record=lambda **kwargs: recorded.append(kwargs),
    )
    monkeypatch.setattr(session_module.quality, "camera_model_metrics", lambda *_args, **_kwargs: (True, "strict PnP", {"pixel_error_px": 0.1}))
    monkeypatch.setattr(session_module.quality, "candidate_quality_snapshot", lambda *_args, **_kwargs: object())
    candidate = SimpleNamespace(spec=object(), idx=1)
    data = (stable, observation, None, object(), object(), "captured", "model", "static")

    category, _note = session._try_record_sample(candidate, data, "endpoint")

    assert category == PASS
    assert len(recorded) == 1
