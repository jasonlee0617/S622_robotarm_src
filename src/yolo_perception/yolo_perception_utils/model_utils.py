from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
    _PKG_SHARE = get_package_share_directory("yolo_perception")
except Exception:
    _PKG_SHARE = None


FOUR_CLASS_OBB_NAMES = {
    0: "box",
    1: "elongated_object",
    2: "cube",
    3: "stone",
}
POSITION_3D_TOPICS = {name: f"/{name}_position_3d" for name in FOUR_CLASS_OBB_NAMES.values()}
RPY_TOPICS = {name: f"/{name}_rpy" for name in FOUR_CLASS_OBB_NAMES.values()}


def require_four_class_obb_model(model_names) -> dict[int, str]:
    """Return the required class table or reject an incompatible model.

    Class IDs are model semantics, never compatibility aliases.  In particular,
    a legacy ``pen, box, cube`` model must not be treated as this four-class
    model because it would silently publish to the wrong ROS topics.
    """
    if isinstance(model_names, dict):
        actual = {int(class_id): str(name) for class_id, name in model_names.items()}
    elif isinstance(model_names, (list, tuple)):
        actual = {class_id: str(name) for class_id, name in enumerate(model_names)}
    else:
        raise ValueError(f"Unsupported YOLO model.names type: {type(model_names).__name__}")

    if actual != FOUR_CLASS_OBB_NAMES:
        raise ValueError(
            "Unsupported YOLO-OBB class contract. Expected "
            f"{FOUR_CLASS_OBB_NAMES}, received {actual}. "
            "Use the four-class yolo-obb-1024.pt model; legacy yolo-obb-gazebo "
            "models are intentionally unsupported."
        )
    return actual


def assign_obb_confidence(inference_result, box) -> None:
    inference_result.confidence = float(box.conf.item())


def resolve_yolo_model_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    if _PKG_SHARE is not None:
        return str(Path(_PKG_SHARE) / "models" / path)
    return str(Path(__file__).resolve().parents[1] / "models" / path)
