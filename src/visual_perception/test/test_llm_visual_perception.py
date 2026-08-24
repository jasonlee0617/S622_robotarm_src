import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from visual_perception_utils.model_utils import (  # noqa: E402
    assign_obb_confidence,
    FOUR_CLASS_OBB_NAMES,
    POSITION_3D_TOPICS,
    AXIS_3D_TOPICS,
    require_four_class_obb_model,
)
from visual_perception_utils.visualization import draw_detection_diagnostics  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_assigns_obb_box_confidence_to_inference_result():
    inference_result = SimpleNamespace(confidence=0.0)
    box = SimpleNamespace(conf=np.array([0.875], dtype=np.float32))

    assign_obb_confidence(inference_result, box)

    assert isinstance(inference_result.confidence, float)
    assert inference_result.confidence == pytest.approx(0.875)


def test_accepts_only_the_four_class_obb_contract():
    assert require_four_class_obb_model(
        ["box", "elongated_object", "cube", "stone"]
    ) == FOUR_CLASS_OBB_NAMES


def test_rejects_legacy_three_class_gazebo_contract():
    with pytest.raises(ValueError, match="legacy yolo-obb-gazebo"):
        require_four_class_obb_model({0: "pen", 1: "box", 2: "cube"})


def test_four_class_results_route_to_their_semantic_topics():
    assert POSITION_3D_TOPICS == {
        "box": "/box_position_3d",
        "elongated_object": "/elongated_object_position_3d",
        "cube": "/cube_position_3d",
        "stone": "/stone_position_3d",
    }
    assert AXIS_3D_TOPICS["elongated_object"] == "/elongated_object_axis_3d"
    assert AXIS_3D_TOPICS["cube"] == "/cube_axis_3d"
    assert AXIS_3D_TOPICS["stone"] == "/stone_axis_3d"


def test_llm_perception_uses_the_new_result_topics_only():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()
    cmake = (ROOT / "CMakeLists.txt").read_text()

    assert '"/yolo/detected_result"' in source
    assert '"/yolo/detected_result/depth"' in source
    assert '"/camera/detected_result"' in source
    assert "/" + "Yolov8" + "_Inference" not in source
    assert "results[0].plot()" not in source
    assert "draw_detection_center" in source
    assert "draw_obb_major_axis" in source
    assert "ReliabilityPolicy.RELIABLE" in source
    assert not (ROOT / "visual_perception_utils" / "llm_visual_perception.py").exists()
    assert (
        "ament_python_install_package(visual_perception_nodes PACKAGE_DIR visual_perception/nodes)"
        in cmake
    )


def test_llm_perception_entrypoint_is_executable():
    assert os.access(ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py", os.X_OK)


def test_llm_perception_supports_continuous_or_search_gated_inference():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()

    assert "self.model(img, conf=self.conf, imgsz=self.imgsz, verbose=False)" in source
    assert "use_continuous_yolo" in source
    assert "/llm_visual_perception/set_inference_enabled" in source
    assert "/llm_visual_perception/release_gpu" in source
    assert "LLM YOLO inference disabled after CUDA OOM" in source
    assert "def _is_cuda_oom" in source
    assert "torch.cuda.empty_cache()" in source
    assert "if enabled and not self._load_model()" in source
    assert "if not self._inference_enabled or self.model is None:" in source
    assert "MultiThreadedExecutor(num_threads=2)" in source
    assert "Disabling must not wait for a long-running CUDA inference callback." in source


def test_obb_detector_uses_the_same_inference_gate_contract():
    source = (ROOT / "visual_perception" / "nodes" / "yolo_detector_obb.py").read_text()

    assert "use_continuous_yolo" in source
    assert "/yolo_detector_obb/set_inference_enabled" in source
    assert "if not self._inference_enabled:" in source


def test_detection_diagnostics_draws_text():
    image = np.zeros((80, 160, 3), dtype=np.uint8)

    draw_detection_diagnostics(image, (10, 10), ["cube conf=0.90"], (0, 255, 0))

    assert image.any()


def test_llm_perception_source_includes_complete_3d_diagnostics():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()

    assert "draw_detection_diagnostics" in source
    assert 'f"base: {center_base.point.x:.3f},' in source
    assert "depthQ:" in source
    assert "3D unavailable:" in source
