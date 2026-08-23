import cv2
import numpy as np


def draw_detection_center(image, center_xy):
    center = (int(center_xy[0]), int(center_xy[1]))
    cv2.circle(image, center, 4, (0, 0, 0), -1)
    cv2.circle(image, center, 3, (0, 255, 0), -1)


def draw_obb_major_axis(image, corners_2d, color):
    """Draw an unsigned OBB axis with arrowheads at both ends."""
    from yolo_perception_utils.obb_geometry import obb_long_edge

    start, end = obb_long_edge(corners_2d)
    if start is None:
        return
    center = (start + end) * 0.5
    direction = end - start
    length = float((direction @ direction) ** 0.5)
    if length < 1e-6:
        return
    direction /= length
    half = length * 0.42
    first = tuple(np.rint(center - direction * half).astype(int))
    second = tuple(np.rint(center + direction * half).astype(int))
    cv2.arrowedLine(image, first, second, color, 2, tipLength=0.18)
    cv2.arrowedLine(image, second, first, color, 2, tipLength=0.18)


def draw_detection_diagnostics(image, center_xy, lines, color):
    """Draw compact diagnostic text below a detection without leaving the image."""
    x, y = (int(value) for value in center_xy)
    height = image.shape[0]
    for index, line in enumerate(lines, 1):
        cv2.putText(
            image,
            str(line),
            (x, min(height - 5, y + 15 * index)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )
