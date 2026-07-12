"""Pure validation and preview helpers for LLM arm tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import statistics
import time
from typing import Iterable


PICK_CLASSES = frozenset({"elongated_object", "cube"})
PLACE_CLASSES = frozenset({"box"})
PICK_EXECUTION_STEPS = 6
PLACE_EXECUTION_STEPS = 7
PICK_PLACE_EXECUTION_STEPS = 13
RETRY_PLACE_EXECUTION_STEPS = 7
ACTION_FIELDS = {
    "pick": {"type", "source_index"},
    "place": {"type", "destination_index"},
    "pick_place": {"type", "source_index", "destination_index"},
    "move_relative": {"type", "dx", "dy", "dz", "droll_deg", "dpitch_deg", "dyaw_deg", "frame_id"},
    "move_absolute": {"type", "x", "y", "z", "qx", "qy", "qz", "qw", "frame_id"},
    "set_gripper": {"type", "state"},
    "home": {"type"},
}


@dataclass(frozen=True)
class DetectionCandidate:
    index: int
    class_name: str


@dataclass(frozen=True)
class TaskPlan:
    actions: tuple[dict, ...]


@dataclass(frozen=True)
class TaskPreview:
    preview_id: str
    plan: TaskPlan
    created_at: float
    max_age_sec: float = 15.0


@dataclass(frozen=True)
class BoxRelocation:
    decision: str
    target_xyz: tuple[float, float, float]
    displacement_m: float


@dataclass(frozen=True)
class SafetyState:
    epoch: int = 0
    blocked: bool = False
    command: str = ""


class ClarificationRequired(ValueError):
    """The command needs user input instead of a local guess."""


_BLOT_PATTERN = re.compile(r"(?<![A-Za-z0-9_])blot(?![A-Za-z0-9_])", re.IGNORECASE)
_VISUAL_INTENT_PATTERN = re.compile(
    r"抓|夹取|拿|拾取|放|摆放|\bpick(?:\s+up)?\b|\bgrasp\b|\bplace\b|\bput\b",
    re.IGNORECASE,
)
VISUAL_ACTIONS = frozenset({"pick", "place", "pick_place"})


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _candidate_map(candidates: Iterable[DetectionCandidate | dict]) -> dict[int, DetectionCandidate]:
    mapped = {}
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = DetectionCandidate(int(candidate["index"]), str(candidate["class_name"]))
        if candidate.index in mapped:
            raise ValueError("candidate indices must be unique")
        mapped[candidate.index] = candidate
    return mapped


def _selected_candidate(action, field, candidates, allowed_classes, reject_ambiguous):
    selected_index = action.get(field)
    if isinstance(selected_index, bool) or not isinstance(selected_index, int):
        raise ValueError(f"{field} must be an integer")
    try:
        selected = candidates[selected_index]
    except KeyError as exc:
        raise ClarificationRequired("selected detection is unavailable; ask the user to clarify") from exc
    if selected.class_name not in allowed_classes:
        role = "pickable" if field == "source_index" else "a place target"
        raise ValueError(f"class {selected.class_name!r} is not {role}")
    if reject_ambiguous and sum(
        item.class_name == selected.class_name for item in candidates.values()
    ) > 1:
        raise ClarificationRequired(
            f"ambiguous {selected.class_name} selection; ask the user to clarify"
        )
    return selected_index


def _validate_visual(action, candidates, pick_classes, place_classes, reject_ambiguous):
    action_type = action["type"]
    normalized = {"type": action_type}
    if action_type in ("pick", "pick_place"):
        normalized["source_index"] = _selected_candidate(
            action, "source_index", candidates, pick_classes, reject_ambiguous
        )
    if action_type in ("place", "pick_place"):
        normalized["destination_index"] = _selected_candidate(
            action, "destination_index", candidates, place_classes, reject_ambiguous
        )
    if (
        action_type == "pick_place"
        and normalized["source_index"] == normalized["destination_index"]
    ):
        raise ValueError("source and destination must differ")
    return normalized


def validate_instruction(instruction: str) -> None:
    if _BLOT_PATTERN.search(str(instruction)):
        raise ClarificationRequired("Unknown object name 'blot'; use 'bolt'.")


def validate_plan_intent(instruction: str, plan: TaskPlan) -> None:
    if _VISUAL_INTENT_PATTERN.search(str(instruction)) and not any(
        action["type"] in VISUAL_ACTIONS for action in plan.actions
    ):
        raise ValueError(
            "visual manipulation requests must use pick, place, or pick_place; "
            "they cannot degrade to gripper or pose actions"
        )


def validate_visual_state(action_type: str, *, holding: bool, recovery: bool) -> None:
    if action_type not in VISUAL_ACTIONS:
        return
    if recovery:
        raise ValueError("placement recovery must be retried or cleared with Home")
    if action_type == "place" and not holding:
        raise ValueError("place requires an object held by a completed pick")
    if action_type != "place" and holding:
        raise ValueError("the arm is already holding an object; place it or press h first")


def _validate_relative(action):
    values = {
        field: _finite_number(action.get(field, 0.0), field)
        for field in ("dx", "dy", "dz", "droll_deg", "dpitch_deg", "dyaw_deg")
    }
    if math.sqrt(sum(values[field] ** 2 for field in ("dx", "dy", "dz"))) > 0.05 + 1e-12:
        raise ValueError("relative translation exceeds 0.05 m")
    if any(abs(values[field]) > 15.0 for field in ("droll_deg", "dpitch_deg", "dyaw_deg")):
        raise ValueError("relative rotation exceeds 15 degrees")
    frame_id = str(action.get("frame_id", "base_link"))
    if frame_id != "base_link":
        raise ValueError("move_relative frame_id must be base_link")
    return {"type": "move_relative", **values, "frame_id": frame_id}


def _validate_absolute(action):
    values = {
        field: _finite_number(action.get(field), field)
        for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
    }
    norm = math.sqrt(sum(values[field] ** 2 for field in ("qx", "qy", "qz", "qw")))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("absolute pose quaternion must be normalized")
    frame_id = str(action.get("frame_id", "base_link"))
    if frame_id != "base_link":
        raise ValueError("move_absolute frame_id must be base_link")
    return {"type": "move_absolute", **values, "frame_id": frame_id}


def parse_llm_plan(
    text,
    candidates=(),
    *,
    pick_classes=PICK_CLASSES,
    place_classes=PLACE_CLASSES,
    reject_ambiguous=True,
) -> TaskPlan:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("LLM response is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != {"actions"}:
        raise ValueError("LLM response must contain only an actions array")
    actions = data["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
        raise ValueError("actions must contain 1 to 8 items")
    candidate_map = _candidate_map(candidates)
    normalized = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            raise ValueError("each action must be an object with a type")
        action_type = action["type"]
        expected = ACTION_FIELDS.get(action_type)
        if expected is None:
            raise ValueError(f"unsupported action: {action_type}")
        if not set(action).issubset(expected):
            raise ValueError(f"unknown fields for {action_type}")
        if action_type in VISUAL_ACTIONS:
            normalized.append(
                _validate_visual(
                    action, candidate_map, pick_classes, place_classes, reject_ambiguous
                )
            )
        elif action_type == "move_relative":
            normalized.append(_validate_relative(action))
        elif action_type == "move_absolute":
            normalized.append(_validate_absolute(action))
        elif action_type == "set_gripper":
            state = str(action.get("state", "")).lower()
            if state not in ("open", "close"):
                raise ValueError("gripper state must be open or close")
            normalized.append({"type": "set_gripper", "state": state})
        else:
            normalized.append({"type": "home"})
    if any(action["type"] in VISUAL_ACTIONS for action in normalized) and len(normalized) != 1:
        raise ValueError("a visual task must use exactly one pick, place, or pick_place action")
    return TaskPlan(tuple(normalized))


def preview_status(preview: TaskPreview, now=None, consumed_preview_ids=()) -> str:
    if preview.preview_id in consumed_preview_ids:
        return "consumed"
    now = time.monotonic() if now is None else float(now)
    age = now - preview.created_at
    return "ready" if 0.0 <= age <= preview.max_age_sec else "expired"


def consume_preview(preview: TaskPreview, now=None, consumed_preview_ids=()) -> frozenset[str]:
    if preview_status(preview, now, consumed_preview_ids) != "ready":
        raise ValueError("preview is not executable")
    return frozenset(consumed_preview_ids) | {preview.preview_id}


def apply_safety_command(state: SafetyState, command: str) -> SafetyState:
    command = str(command).strip().lower()
    if command in ("stop", "reset"):
        return SafetyState(state.epoch + 1, True, command)
    if command == "resume":
        return SafetyState(state.epoch, False, command)
    return state


def complete_safety_reset(state: SafetyState) -> SafetyState:
    return SafetyState(state.epoch, False, "")


def safety_execution_valid(state: SafetyState, execution_epoch: int) -> bool:
    return not state.blocked and state.epoch == int(execution_epoch)


def execution_step_count(actions: TaskPlan | Iterable[dict]) -> int:
    actions = actions.actions if isinstance(actions, TaskPlan) else actions
    counts = {
        "pick": PICK_EXECUTION_STEPS,
        "place": PLACE_EXECUTION_STEPS,
        "pick_place": PICK_PLACE_EXECUTION_STEPS,
        "retry_place": RETRY_PLACE_EXECUTION_STEPS,
    }
    return sum(counts.get(action.get("type"), 1) for action in actions)


def build_semantic_history(instruction, plan=None, candidates=()):
    user_message = {"role": "user", "content": str(instruction)}
    if plan is None:
        summary = {"status": "clarification_required"}
    else:
        candidate_map = _candidate_map(candidates)
        actions = []
        for action in plan.actions:
            action_type = action["type"]
            semantic = {"type": action_type}
            if action_type in ("pick", "pick_place"):
                semantic["source_class"] = candidate_map[action["source_index"]].class_name
            if action_type in ("place", "pick_place"):
                semantic["destination_class"] = candidate_map[
                    action["destination_index"]
                ].class_name
            elif action_type == "set_gripper":
                semantic["state"] = action["state"]
            elif action_type in ("move_relative", "move_absolute"):
                semantic["frame_id"] = "base_link"
            actions.append(semantic)
        summary = {"actions": actions}
    assistant_message = {
        "role": "assistant",
        "content": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    return user_message, assistant_message


def decide_box_relocation(
    preview_xyz,
    samples,
    *,
    sample_count=5,
    retarget_threshold_m=0.01,
    max_shift_m=0.05,
    stability_threshold_m=0.01,
) -> BoxRelocation:
    sample_count = int(sample_count)
    retarget_threshold_m = _finite_number(retarget_threshold_m, "retarget threshold")
    max_shift_m = _finite_number(max_shift_m, "maximum shift")
    stability_threshold_m = _finite_number(stability_threshold_m, "stability threshold")
    if sample_count < 2:
        raise ValueError("box sample count must be at least two")
    if not 0.0 <= retarget_threshold_m <= max_shift_m:
        raise ValueError("box thresholds must satisfy 0 <= retarget <= maximum shift")
    if stability_threshold_m < 0.0:
        raise ValueError("box stability threshold must be non-negative")
    samples = [tuple(_finite_number(v, "box coordinate") for v in sample) for sample in samples]
    if len(samples) != sample_count or any(len(sample) != 3 for sample in samples):
        raise ValueError(f"box relocation requires exactly {sample_count} XYZ samples")
    target = tuple(statistics.median(sample[axis] for sample in samples) for axis in range(3))
    max_pairwise_xy = max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(samples)
        for right in samples[index + 1:]
    )
    if max_pairwise_xy > stability_threshold_m + 1e-12:
        raise ValueError("box samples are not stable")
    preview_xyz = tuple(_finite_number(v, "preview coordinate") for v in preview_xyz)
    if len(preview_xyz) != 3:
        raise ValueError("preview coordinate must contain XYZ")
    dx = target[0] - float(preview_xyz[0])
    dy = target[1] - float(preview_xyz[1])
    displacement = math.hypot(dx, dy)
    decision = (
        "unchanged" if displacement <= retarget_threshold_m + 1e-12
        else "relocate" if displacement <= max_shift_m + 1e-12
        else "reject"
    )
    return BoxRelocation(decision, target, displacement)
