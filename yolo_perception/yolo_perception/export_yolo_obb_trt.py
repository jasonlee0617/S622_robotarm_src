from pathlib import Path
from ultralytics import YOLO

try:
    from ament_index_python.packages import get_package_share_directory
    model_path = Path(get_package_share_directory("yolo_perception")) / "models" / "yolo-obb-gazebo.pt"
except Exception:
    model_path = Path(__file__).resolve().parent / "yolo-obb-gazebo.pt"

model = YOLO(str(model_path))
model.export(
    format="engine",
    imgsz=640,
    dynamic=False,
    half=True,
    device=0,
)
