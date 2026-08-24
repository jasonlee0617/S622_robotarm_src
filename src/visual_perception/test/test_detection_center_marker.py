from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_perception_utils.visualization import (  # noqa: E402
    draw_detection_center,
    draw_obb_major_axis,
)


def test_detection_center_marker_is_green_with_dark_outline():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    draw_detection_center(image, (16, 16))

    assert image[16, 16].tolist() == [0, 255, 0]
    assert image[16, 20].tolist() == [0, 0, 0]


def test_obb_major_axis_draws_on_image():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    corners = np.array([[12, 20], [48, 20], [48, 30], [12, 30]], dtype=np.float32)

    draw_obb_major_axis(image, corners, (0, 255, 0))

    assert np.count_nonzero(image) > 0
