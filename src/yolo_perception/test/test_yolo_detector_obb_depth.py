from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_perception_utils.depth_estimation import robust_center3d_from_obb_depth  # noqa: E402


CAMERA_INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}


def _obb():
    return np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)


def test_center3d_rejects_sparse_depth_points():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[45:50, 45:50] = 1.0

    center, _quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=50,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
    )

    assert center is None


def test_center3d_uses_object_depth_when_mask_has_outliers():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[25:75, 25:75] = 1.0
    depth[25:75:5, 25:75] = 1.8

    center, quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=20,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
    )

    assert center is not None
    assert abs(float(center[2]) - 1.0) < 0.02
    assert quality > 0.6


def test_center3d_can_project_xy_from_obb_center_for_open_box():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[65:75, 25:75] = 1.0

    center, quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=20,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
        xy_from_obb_center=True,
    )

    assert center is not None
    assert abs(float(center[0])) < 0.001
    assert abs(float(center[1])) < 0.001
    assert abs(float(center[2]) - 1.0) < 0.001
    assert quality == 1.0


def test_center3d_rejects_highly_split_depth_distribution():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[:, ::2] = 1.4

    center, _quality = robust_center3d_from_obb_depth(
        poly_2d=_obb(),
        depth=depth,
        camera_intrinsics=CAMERA_INTRINSICS,
        stride=1,
        min_points=20,
        max_points=10000,
        depth_max_range=10.0,
        depth_inlier_m=0.08,
        depth_mad_scale=3.0,
        min_depth_inlier_ratio=0.6,
    )

    assert center is None
