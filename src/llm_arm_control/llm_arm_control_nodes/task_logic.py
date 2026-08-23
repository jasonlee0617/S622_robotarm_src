"""Pure validation and preview helpers for LLM arm tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from typing import Iterable


PICK_CLASSES = frozenset({"elongated_object", "cube"})
PLACE_CLASSES = frozenset({"box"})
PICK_EXECUTION_STEPS = 6
PLACE_EXECUTION_STEPS = 4
PICK_PLACE_EXECUTION_STEPS = 10
ACTION_FIELDS = {
    "pick": {"type", "source_index"},
    "place": {"type", "destination_index"},
    "pick_place": {"type", "source_index", "destination_index"},
    "move_relative": {"type", "dx", "dy", "dz", "droll_deg", "dpitch_deg", "dyaw_deg", "frame_id"},
    "move_absolute": {"type", "x", "y", "z", "qx", "qy", "qz", "qw", "frame_id"},
    "set_gripper": {"type", "state"},
    "home": {"type"},
}
SYSTEM_PROMPT = """你通过严格校验的本地规划器控制 Fairino 机械臂。
只能返回 JSON，且顶层只能包含：{"actions": [...]}。
允许的动作：
1. {"type":"pick","source_index":int}
2. {"type":"place","destination_index":int}
3. {"type":"pick_place","source_index":int,"destination_index":int}
4. {"type":"move_relative","dx":m,"dy":m,"dz":m,
   "droll_deg":deg,"dpitch_deg":deg,"dyaw_deg":deg,"frame_id":"base_link"}
5. {"type":"move_absolute","x":m,"y":m,"z":m,
   "qx":number,"qy":number,"qz":number,"qw":number,"frame_id":"base_link"}
6. {"type":"set_gripper","state":"open|close"}
7. {"type":"home"}
elongated_object 的别名为 bolt、螺栓、螺丝、pen、笔。抓取且没有目标盒子时使用 pick；
放入盒子时使用 place；同时指定物体和盒子时使用 pick_place。视觉抓取或放置不能替换为
set_gripper、move_relative 或 move_absolute。只能使用候选列表给出的检测索引，不能虚构坐标；
存在歧义时不要猜测，返回不可用索引以便本地校验器拒绝。
center_uv 是图像像素坐标：u 向右增大，v 向下增大。左/右/上/下选择最小/最大 u/v，
中间选择距离图像中心最近的候选。base_xyz 位于 base_link；最近/最远以候选 base_xyz
到 current_pose（tool0）的欧氏距离判断。视觉任务只能返回一个 pick、place 或
pick_place；未明确坐标系的位移一律使用 base_link；最多八个动作。
"""


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
class SafetyState:
    epoch: int = 0
    blocked: bool = False
    command: str = ""


class ClarificationRequired(ValueError):
    """The command needs user input instead of a local guess."""


_VISUAL_INTENT_PATTERN = re.compile(
    r"抓|夹取|拿|拾取|放|摆放|\bpick(?:\s+up)?\b|\bgrasp\b|\bplace\b|\bput\b",
    re.IGNORECASE,
)
VISUAL_ACTIONS = frozenset({"pick", "place", "pick_place"})
_VISUAL_CLASS_ALIASES = {
    "elongated_object": (
        "elongated_object", "elongated object", "bolt", "pen", "螺栓", "螺丝", "笔",
    ),
    "cube": ("cube", "方块", "立方体"),
    "stone": ("stone", "石头"),
    "box": ("box", "盒", "箱"),
}
_PICK_PATTERN = re.compile(r"抓|夹取|拿|拾取|\bpick(?:\s+up)?\b|\bgrasp\b", re.IGNORECASE)
_PLACE_PATTERN = re.compile(r"放|摆放|\bplace\b|\bput\b", re.IGNORECASE)
_GENERIC_PICK_TARGET_PATTERN = re.compile(
    r"物体|目标|东西|\b(object|item|thing|target)\b", re.IGNORECASE
)


def _contains_alias(instruction: str, alias: str) -> bool:
    if any("\u4e00" <= char <= "\u9fff" for char in alias):
        return alias in instruction
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", instruction, re.IGNORECASE))


def _requested_visual_classes(instruction: str) -> tuple[str, ...]:
    return tuple(
        class_name
        for class_name, aliases in _VISUAL_CLASS_ALIASES.items()
        if any(_contains_alias(instruction, alias) for alias in aliases)
    )


def _visual_selector(instruction: str) -> tuple[str, ...] | None:
    lowered = instruction.lower()
    flags = {
        "left": "左" in instruction or "left" in lowered,
        "right": "右" in instruction or "right" in lowered,
        "top": "上" in instruction or "top" in lowered,
        "bottom": "下" in instruction or "bottom" in lowered,
        "center": any(token in instruction for token in ("中间", "中心"))
        or any(token in lowered for token in ("center", "middle")),
        "nearest": any(token in instruction for token in ("最近", "靠近"))
        or any(token in lowered for token in ("nearest", "closest")),
        "farthest": any(token in instruction for token in ("最远", "远离"))
        or "farthest" in lowered,
    }
    if flags["left"] and flags["right"] or flags["top"] and flags["bottom"]:
        raise ClarificationRequired("image direction is contradictory")
    if flags["nearest"] and flags["farthest"]:
        raise ClarificationRequired("distance direction is contradictory")
    image_axes = tuple(axis for axis in ("left", "right", "top", "bottom") if flags[axis])
    distance = tuple(axis for axis in ("nearest", "farthest") if flags[axis])
    if flags["center"] and (image_axes or distance):
        raise ClarificationRequired("center cannot be combined with another spatial selector")
    if image_axes and distance:
        raise ClarificationRequired("image direction cannot be combined with nearest or farthest")
    if flags["center"]:
        return ("center",)
    return image_axes or distance or None


def _select_spatial_candidate(class_names, metadata, selector, current_xyz):
    if isinstance(class_names, str):
        class_names = (class_names,)
    class_names = tuple(class_names)
    candidates = [
        item for item in metadata if str(item.get("class_name")) in class_names
    ]
    label = class_names[0] if len(class_names) == 1 else "pickable object"
    if not candidates:
        raise ClarificationRequired(f"no visible {label} candidate")
    if selector is None:
        if len(candidates) != 1:
            raise ClarificationRequired(
                f"multiple {label} candidates; specify left, right, top, bottom, center, nearest, or farthest"
            )
        return candidates[0]
    if selector[0] in ("nearest", "farthest"):
        try:
            ranked = [
                (math.dist(tuple(float(value) for value in item["base_xyz"]), current_xyz), int(item["index"]), item)
                for item in candidates
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ClarificationRequired("candidate has no usable base_link position") from exc
        return (min if selector[0] == "nearest" else max)(ranked, key=lambda item: (item[0], -item[1]))[2]
    try:
        dimensions = {tuple(int(value) for value in item["image_size"]) for item in candidates}
        if len(dimensions) != 1:
            raise ValueError
        width, height = dimensions.pop()
        if width <= 0 or height <= 0:
            raise ValueError
        ranked = []
        for item in candidates:
            u, v = (float(value) for value in item["center_uv"])
            if selector == ("center",):
                score = math.hypot(u / width - 0.5, v / height - 0.5)
            else:
                score = sum({
                    "left": u / width,
                    "right": 1.0 - u / width,
                    "top": v / height,
                    "bottom": 1.0 - v / height,
                }[axis] for axis in selector)
            ranked.append((score, int(item["index"]), item))
    except (KeyError, TypeError, ValueError) as exc:
        raise ClarificationRequired("candidate has no usable image position") from exc
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def deterministic_visual_plan(instruction, metadata, *, current_xyz, pick_classes, place_classes):
    """Resolve unambiguous visual commands locally; return None for LLM fallback."""
    instruction = str(instruction)
    if not instruction_has_visual_intent(instruction):
        return None
    requested = _requested_visual_classes(instruction)
    source_classes = [name for name in requested if name in pick_classes]
    destination_classes = [name for name in requested if name in place_classes]
    if len(source_classes) > 1 or len(destination_classes) > 1:
        raise ClarificationRequired("multiple visual object classes were requested")
    has_pick, has_place = bool(_PICK_PATTERN.search(instruction)), bool(_PLACE_PATTERN.search(instruction))
    selector = _visual_selector(instruction)
    generic_pick_target = bool(_GENERIC_PICK_TARGET_PATTERN.search(instruction))
    selected_source_classes = tuple(source_classes)
    if not selected_source_classes and has_pick and generic_pick_target:
        selected_source_classes = tuple(sorted(pick_classes))
    if selected_source_classes and destination_classes:
        source = _select_spatial_candidate(
            selected_source_classes, metadata, selector, current_xyz
        )
        destination = _select_spatial_candidate(destination_classes[0], metadata, None, current_xyz)
        return TaskPlan(({"type": "pick_place", "source_index": int(source["index"]),
                          "destination_index": int(destination["index"])},))
    if selected_source_classes and has_pick:
        source = _select_spatial_candidate(
            selected_source_classes, metadata, selector, current_xyz
        )
        return TaskPlan(({"type": "pick", "source_index": int(source["index"])},))
    if destination_classes and has_place:
        destination = _select_spatial_candidate(destination_classes[0], metadata, selector, current_xyz)
        return TaskPlan(({"type": "place", "destination_index": int(destination["index"])},))
    return None


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


def instruction_has_visual_intent(instruction: str) -> bool:
    return bool(_VISUAL_INTENT_PATTERN.search(str(instruction)))


def validate_plan_intent(instruction: str, plan: TaskPlan) -> None:
    if instruction_has_visual_intent(instruction) and not any(
        action["type"] in VISUAL_ACTIONS for action in plan.actions
    ):
        raise ValueError(
            "visual manipulation requests must use pick, place, or pick_place; "
            "they cannot degrade to gripper or pose actions"
        )
    action_types = {action["type"] for action in plan.actions}
    asks_pick = bool(_PICK_PATTERN.search(str(instruction)))
    asks_place = bool(_PLACE_PATTERN.search(str(instruction)))
    if asks_pick and asks_place and action_types != {"pick_place"}:
        raise ValueError("a requested pick and place must use pick_place")
    if asks_pick and not action_types.intersection({"pick", "pick_place"}):
        raise ValueError("a requested pick must include a pick target")


def validate_visual_state(action_type: str, *, holding: bool, recovery: bool) -> None:
    if action_type not in VISUAL_ACTIONS:
        return
    if recovery:
        raise ValueError("placement recovery must be retried or cleared with Home")
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


def preview_status(preview: TaskPreview, now=None) -> str:
    now = time.monotonic() if now is None else float(now)
    age = now - preview.created_at
    return "ready" if 0.0 <= age <= preview.max_age_sec else "expired"


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
