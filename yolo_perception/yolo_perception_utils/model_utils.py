from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
    _PKG_SHARE = get_package_share_directory("yolo_perception")
except Exception:
    _PKG_SHARE = None


CANONICAL_CLASS_NAMES = {
    0: "elongated_object",
    1: "box",
    2: "cube",
}


def canonical_class_name(class_id: int, model_names=None) -> str:
    class_id = int(class_id)
    if class_id in CANONICAL_CLASS_NAMES:
        return CANONICAL_CLASS_NAMES[class_id]
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, f"cls{class_id}"))
    if isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return f"cls{class_id}"


def apply_canonical_class_names(result) -> None:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        result.names = {**names, **CANONICAL_CLASS_NAMES}
    elif isinstance(names, (list, tuple)):
        updated = list(names)
        for class_id, class_name in CANONICAL_CLASS_NAMES.items():
            if class_id < len(updated):
                updated[class_id] = class_name
        result.names = updated


def assign_obb_confidence(inference_result, box) -> None:
    inference_result.confidence = float(box.conf.item())


def resolve_yolo_model_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    if _PKG_SHARE is not None:
        return str(Path(_PKG_SHARE) / "models" / path)
    return str(Path(__file__).resolve().parents[1] / "models" / path)
