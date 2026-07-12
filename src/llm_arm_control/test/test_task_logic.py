import json

import pytest

from llm_arm_control.task_logic import (
    ClarificationRequired,
    DetectionCandidate,
    SafetyState,
    TaskPreview,
    apply_safety_command,
    build_semantic_history,
    complete_safety_reset,
    consume_preview,
    decide_box_relocation,
    execution_step_count,
    parse_llm_plan,
    preview_status,
    safety_execution_valid,
    validate_instruction,
    validate_plan_intent,
    validate_visual_state,
)


CANDIDATES = (
    DetectionCandidate(0, "elongated_object"),
    DetectionCandidate(1, "box"),
)


def plan(actions, candidates=CANDIDATES, **kwargs):
    return parse_llm_plan(json.dumps({"actions": actions}), candidates, **kwargs)


@pytest.mark.parametrize("value", ["", "[]", '{"actions": []}', '{"actions": [{"type": "unknown"}]}'])
def test_rejects_invalid_or_empty_plans(value):
    with pytest.raises(ValueError):
        parse_llm_plan(value, CANDIDATES)


@pytest.mark.parametrize(
    "action",
    [
        {"type": "pick", "source_index": 0},
        {"type": "place", "destination_index": 1},
        {"type": "pick_place", "source_index": 0, "destination_index": 1},
    ],
)
def test_validates_visual_actions_and_indices(action):
    assert plan([action]).actions[0] == action
    with pytest.raises(ClarificationRequired, match="unavailable"):
        plan([{"type": "pick_place", "source_index": 9, "destination_index": 1}])


def test_only_canonical_visual_classes_are_accepted():
    aliases_are_not_detector_classes = (
        DetectionCandidate(0, "bolt"),
        DetectionCandidate(1, "box"),
    )
    with pytest.raises(ValueError, match="not pickable"):
        plan([{"type": "pick", "source_index": 0}], aliases_are_not_detector_classes)


def test_rejects_ambiguous_class_without_local_disambiguation():
    candidates = CANDIDATES + (DetectionCandidate(2, "elongated_object"),)
    with pytest.raises(ClarificationRequired, match="ambiguous"):
        plan([{"type": "pick_place", "source_index": 0, "destination_index": 1}], candidates)
    assert plan(
        [{"type": "pick_place", "source_index": 0, "destination_index": 1}],
        candidates,
        reject_ambiguous=False,
    )


def test_visual_action_cannot_be_mixed_with_low_level_actions():
    with pytest.raises(ValueError, match="exactly one"):
        plan([
            {"type": "pick", "source_index": 0},
            {"type": "set_gripper", "state": "close"},
        ])


@pytest.mark.parametrize("instruction", ["抓取 pen", "抓取 bolt", "pick up the pen"])
def test_pen_and_bolt_are_valid_language_aliases(instruction):
    validate_instruction(instruction)


@pytest.mark.parametrize("instruction", ["抓取 blot", "pick BLOT", "place the blot in box"])
def test_blot_is_rejected_before_llm_planning(instruction):
    with pytest.raises(ClarificationRequired, match="use 'bolt'"):
        validate_instruction(instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        "抓取 bolt",
        "抓 bolt",
        "把 cube 放到 box",
        "把 pen 放 box",
        "pick the pen",
        "place it in the box",
    ],
)
def test_visual_intent_cannot_degrade_to_gripper_or_pose_actions(instruction):
    low_level = plan([{"type": "set_gripper", "state": "open"}], candidates=())
    with pytest.raises(ValueError, match="must use pick, place, or pick_place"):
        validate_plan_intent(instruction, low_level)


def test_visual_intent_guard_accepts_visual_action():
    validate_plan_intent("抓取 bolt", plan([{"type": "pick", "source_index": 0}]))


@pytest.mark.parametrize("action_type", ["pick", "pick_place"])
def test_pick_requires_empty_gripper_state(action_type):
    validate_visual_state(action_type, holding=False, recovery=False)
    with pytest.raises(ValueError, match="already holding"):
        validate_visual_state(action_type, holding=True, recovery=False)


def test_place_requires_holding_state_and_blocks_during_recovery():
    validate_visual_state("place", holding=True, recovery=False)
    with pytest.raises(ValueError, match="requires an object"):
        validate_visual_state("place", holding=False, recovery=False)
    with pytest.raises(ValueError, match="recovery"):
        validate_visual_state("place", holding=True, recovery=True)


def test_relative_limits_and_default_frame():
    result = plan([{"type": "move_relative", "dx": 0.03, "dy": 0.04}], candidates=())
    assert result.actions[0]["frame_id"] == "base_link"
    with pytest.raises(ValueError, match="translation"):
        plan([{"type": "move_relative", "dx": 0.051}], candidates=())
    with pytest.raises(ValueError, match="rotation"):
        plan([{"type": "move_relative", "dyaw_deg": 15.1}], candidates=())


def test_absolute_pose_requires_unit_quaternion():
    assert plan([
        {"type": "move_absolute", "x": 0.2, "y": 0.3, "z": 0.2,
         "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    ], candidates=())
    with pytest.raises(ValueError, match="normalized"):
        plan([
            {"type": "move_absolute", "x": 0.2, "y": 0.3, "z": 0.2,
             "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 2.0}
        ], candidates=())


@pytest.mark.parametrize("action_type", ["move_relative", "move_absolute"])
def test_motion_frames_are_strictly_base_link(action_type):
    if action_type == "move_relative":
        action = {"type": action_type, "dx": 0.01, "frame_id": "tool0"}
    else:
        action = {
            "type": action_type,
            "x": 0.2,
            "y": 0.3,
            "z": 0.2,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "frame_id": "tool0",
        }
    with pytest.raises(ValueError, match="base_link"):
        plan([action], candidates=())


def test_preview_is_single_use_and_expires_at_boundary():
    preview = TaskPreview("id", plan([{"type": "home"}], candidates=()), 10.0, 15.0)
    assert preview_status(preview, 25.0) == "ready"
    consumed = consume_preview(preview, 25.0)
    assert preview_status(preview, 25.0, consumed) == "consumed"
    assert preview_status(preview, 25.001) == "expired"


@pytest.mark.parametrize(
    ("delta", "decision"),
    [(0.010, "unchanged"), (0.011, "relocate"), (0.050, "relocate"), (0.051, "reject")],
)
def test_box_relocation_uses_five_frame_median(delta, decision):
    samples = [(0.2 + delta, 0.25, 0.1)] * 5
    result = decide_box_relocation((0.2, 0.25, 0.1), samples)
    assert result.decision == decision
    assert result.target_xyz[0] == pytest.approx(0.2 + delta)


def test_box_relocation_requires_five_samples():
    with pytest.raises(ValueError, match="exactly 5"):
        decide_box_relocation((0.2, 0.25, 0.1), [(0.2, 0.25, 0.1)] * 4)


def test_box_relocation_thresholds_are_parameterized():
    samples = [(0.23, 0.25, 0.1)] * 3
    result = decide_box_relocation(
        (0.2, 0.25, 0.1),
        samples,
        sample_count=3,
        retarget_threshold_m=0.02,
        max_shift_m=0.04,
        stability_threshold_m=0.005,
    )
    assert result.decision == "relocate"


def test_box_relocation_rejects_unstable_five_frame_window():
    samples = [
        (0.20, 0.25, 0.1),
        (0.20, 0.25, 0.1),
        (0.20, 0.25, 0.1),
        (0.20, 0.25, 0.1),
        (0.22, 0.25, 0.1),
    ]
    with pytest.raises(ValueError, match="not stable"):
        decide_box_relocation(
            (0.2, 0.25, 0.1), samples, stability_threshold_m=0.01
        )


def test_safety_epoch_permanently_invalidates_old_execution():
    state = SafetyState()
    assert safety_execution_valid(state, 0)

    stopped = apply_safety_command(state, "stop")
    assert stopped.blocked
    assert not safety_execution_valid(stopped, 0)
    resumed = apply_safety_command(stopped, "resume")
    assert safety_execution_valid(resumed, 1)
    assert not safety_execution_valid(resumed, 0)

    resetting = apply_safety_command(resumed, "reset")
    reset_complete = complete_safety_reset(resetting)
    assert safety_execution_valid(reset_complete, 2)
    assert not safety_execution_valid(reset_complete, 1)


def test_feedback_step_count_matches_complete_execution_chain():
    actions = [
        {"type": "pick"},
        {"type": "place"},
        {"type": "pick_place"},
        {"type": "set_gripper"},
        {"type": "retry_place"},
    ]
    assert execution_step_count(actions) == 6 + 7 + 13 + 1 + 7


def test_language_history_excludes_frame_specific_detection_atoms():
    parsed = plan([
        {"type": "pick_place", "source_index": 0, "destination_index": 1},
    ])
    history = build_semantic_history("抓取 bolt，然后放到 box", parsed, CANDIDATES)
    assert history[0] == {"role": "user", "content": "抓取 bolt，然后放到 box"}
    assistant = history[1]["content"]
    assert "source_class" in assistant
    assert "destination_class" in assistant
    assert all(atom not in assistant for atom in ("index", "center", "base_xyz", "coordinates"))


@pytest.mark.parametrize(
    ("action", "semantic_field"),
    [
        ({"type": "pick", "source_index": 0}, "source_class"),
        ({"type": "place", "destination_index": 1}, "destination_class"),
    ],
)
def test_standalone_visual_history_is_semantic(action, semantic_field):
    history = build_semantic_history("instruction", plan([action]), CANDIDATES)
    assert semantic_field in history[1]["content"]


def test_clarification_history_contains_no_llm_detection_selection():
    history = build_semantic_history("抓取 bolt")
    assert json.loads(history[1]["content"]) == {"status": "clarification_required"}
