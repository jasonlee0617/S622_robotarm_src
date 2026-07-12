from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_perception_utils.model_utils import (  # noqa: E402
    apply_canonical_class_names,
    assign_obb_confidence,
    canonical_class_name,
)


def test_assigns_obb_box_confidence_to_inference_result():
    inference_result = SimpleNamespace(confidence=0.0)
    box = SimpleNamespace(conf=np.array([0.875], dtype=np.float32))

    assign_obb_confidence(inference_result, box)

    assert isinstance(inference_result.confidence, float)
    assert inference_result.confidence == pytest.approx(0.875)


@pytest.mark.parametrize(
    ("class_id", "expected"),
    [(0, "elongated_object"), (1, "box"), (2, "cube")],
)
def test_uses_canonical_public_class_names(class_id, expected):
    model_names = {0: "legacy_class_0", 1: "box", 2: "cube"}

    assert canonical_class_name(class_id, model_names) == expected


def test_updates_result_names_used_by_ultralytics_overlay():
    result = SimpleNamespace(names={0: "legacy_class_0", 1: "box", 2: "cube"})

    apply_canonical_class_names(result)

    assert result.names == {0: "elongated_object", 1: "box", 2: "cube"}
