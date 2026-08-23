"""Thread-safe callers use these pure helpers to manage task previews."""

from dataclasses import dataclass

from llm_arm_control_nodes.task_logic import TaskPreview, preview_status


@dataclass
class PreviewRecord:
    preview: TaskPreview
    session_id: str
    instruction: str
    enriched_actions: list[dict]
    safety_epoch: int
    public: dict


def clear_session(sessions: dict, previews: dict[str, PreviewRecord], session_id: str) -> None:
    sessions.pop(session_id, None)
    for preview_id in [key for key, record in previews.items() if record.session_id == session_id]:
        previews.pop(preview_id, None)


def prune(previews: dict[str, PreviewRecord], now=None) -> None:
    for preview_id in [
        key for key, record in previews.items()
        if preview_status(record.preview, now) != "ready"
    ]:
        previews.pop(preview_id, None)


def take(previews: dict[str, PreviewRecord], preview_id: str) -> PreviewRecord | None:
    return previews.pop(preview_id, None)
