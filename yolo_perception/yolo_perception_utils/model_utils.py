from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
    _PKG_SHARE = get_package_share_directory("yolo_perception")
except Exception:
    _PKG_SHARE = None


def resolve_yolo_model_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    if _PKG_SHARE is not None:
        return str(Path(_PKG_SHARE) / "models" / path)
    return str(Path(__file__).resolve().parents[1] / "models" / path)
