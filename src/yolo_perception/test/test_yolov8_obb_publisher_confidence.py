from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_perception_utils.model_utils import (  # noqa: E402
    assign_obb_confidence,
    FOUR_CLASS_OBB_NAMES,
    POSITION_3D_TOPICS,
    RPY_TOPICS,
    require_four_class_obb_model,
)


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
    assert RPY_TOPICS["elongated_object"] == "/elongated_object_rpy"
    assert RPY_TOPICS["cube"] == "/cube_rpy"
    assert RPY_TOPICS["stone"] == "/stone_rpy"
