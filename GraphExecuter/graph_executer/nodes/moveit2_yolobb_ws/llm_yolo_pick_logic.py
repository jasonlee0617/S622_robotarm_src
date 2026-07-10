from dataclasses import dataclass
import json
import time


@dataclass(frozen=True)
class PickPreview:
    index: int
    class_name: str
    target: tuple[float, float, float]
    frame_stamp_ns: int
    created_monotonic: float


def parse_selected_index(response_text, candidates):
    try:
        data = json.loads(response_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("LLM response is not valid JSON") from exc

    index = data.get("selected_index") if isinstance(data, dict) else None
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("LLM response must contain an integer selected_index")
    valid_indices = {int(candidate["index"]) for candidate in candidates}
    if index not in valid_indices:
        raise ValueError("LLM selected an unavailable detection")
    return index


def make_preview(candidate, now_monotonic=None):
    now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
    return PickPreview(
        index=int(candidate["index"]),
        class_name=str(candidate["class_name"]),
        target=tuple(float(value) for value in candidate["target"]),
        frame_stamp_ns=int(candidate["frame_stamp_ns"]),
        created_monotonic=now_monotonic,
    )


def preview_is_confirmable(preview, current_frame_stamp_ns, max_age_sec, now_monotonic=None):
    if preview is None or int(current_frame_stamp_ns) != preview.frame_stamp_ns:
        return False
    now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
    return 0.0 <= now_monotonic - preview.created_monotonic <= max(0.0, float(max_age_sec))
