import cv2
import numpy as np


def robust_center3d_from_obb_depth(
    poly_2d: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: dict,
    stride: int,
    min_points: int,
    max_points: int,
    depth_max_range: float,
    depth_inlier_m: float,
    depth_mad_scale: float,
    min_depth_inlier_ratio: float,
    xy_from_obb_center: bool = False,
):
    H, W = depth.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_2d.astype(np.int32)], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(mask > 0)
    if xs.size < 100:
        return None, 0.0

    stride = max(1, int(stride))
    sample_xs = xs[::stride][:max_points]
    sample_ys = ys[::stride][:max_points]
    if sample_xs.size < min_points:
        return None, 0.0

    zs = depth[sample_ys, sample_xs].astype(np.float32)
    valid = np.isfinite(zs) & (zs > 0.0) & (zs <= depth_max_range)
    if int(np.count_nonzero(valid)) < min_points:
        return None, 0.0

    sample_xs = sample_xs[valid].astype(np.float32)
    sample_ys = sample_ys[valid].astype(np.float32)
    zs = zs[valid]

    z_median = float(np.median(zs))
    abs_dev = np.abs(zs - z_median)
    mad = float(np.median(abs_dev))
    cutoff = max(0.001, float(depth_inlier_m))
    if depth_mad_scale > 0.0 and mad > 0.0:
        robust_sigma = 1.4826 * mad
        cutoff = min(cutoff, max(0.005, float(depth_mad_scale) * robust_sigma))

    inlier = abs_dev <= cutoff
    inlier_count = int(np.count_nonzero(inlier))
    if inlier_count < min_points:
        return None, 0.0

    inlier_ratio = float(inlier_count) / float(zs.size)
    if inlier_ratio < min(1.0, max(0.0, float(min_depth_inlier_ratio))):
        return None, inlier_ratio

    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    xs_in = sample_xs[inlier]
    ys_in = sample_ys[inlier]
    zs_in = zs[inlier]
    if xy_from_obb_center:
        u, v = np.mean(poly_2d.reshape(-1, 2), axis=0).astype(np.float32)
        z = np.float32(np.median(zs_in))
        points = np.array([[(u - cx) * z / fx, (v - cy) * z / fy, z]], dtype=np.float32)
        return points[0], inlier_ratio

    points = np.column_stack(((xs_in - cx) * zs_in / fx, (ys_in - cy) * zs_in / fy, zs_in))
    return np.mean(points, axis=0).astype(np.float32), inlier_ratio
