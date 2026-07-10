from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_perception_utils.visualization import draw_detection_center  # noqa: E402


def test_detection_center_marker_is_green_with_dark_outline():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    draw_detection_center(image, (16, 16))

    assert image[16, 16].tolist() == [0, 255, 0]
    assert image[16, 20].tolist() == [0, 0, 0]
