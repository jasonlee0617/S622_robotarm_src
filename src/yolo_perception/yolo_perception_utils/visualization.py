import cv2


def draw_detection_center(image, center_xy):
    center = (int(center_xy[0]), int(center_xy[1]))
    cv2.circle(image, center, 4, (0, 0, 0), -1)
    cv2.circle(image, center, 3, (0, 255, 0), -1)
