from pathlib import Path
from ultralytics import YOLO

try:
    from ament_index_python.packages import get_package_share_directory
    model_path = Path(get_package_share_directory("yolo_perception")) / "models" / "yolo-obb-gazebo-1024.pt"
except Exception:
    model_path = Path(__file__).resolve().parents[1] / "models" / "yolo-obb-gazebo-1024.pt"

model = YOLO(str(model_path))
model.export(
    format="engine",
    imgsz=1024,
    dynamic=False,
    half=True,
    device=0,
)
