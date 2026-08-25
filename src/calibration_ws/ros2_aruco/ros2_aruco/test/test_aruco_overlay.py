from types import SimpleNamespace

import numpy as np

from ros2_aruco.aruco_node import ArucoNode


class _Publisher:
    def __init__(self, subscribers=1):
        self.subscribers = subscribers
        self.messages = []

    def get_subscription_count(self):
        return self.subscribers

    def publish(self, message):
        self.messages.append(message)


class _Bridge:
    def __init__(self, image):
        self.image = image
        self.rendered = None

    def imgmsg_to_cv2(self, _msg, desired_encoding):
        assert desired_encoding == "bgr8"
        return self.image.copy()

    def cv2_to_imgmsg(self, image, encoding):
        assert encoding == "bgr8"
        self.rendered = image.copy()
        return SimpleNamespace(header=None)


def _node(subscribers=1):
    node = object.__new__(ArucoNode)
    node.visualization_pub = _Publisher(subscribers)
    node.bridge = _Bridge(np.full((100, 100, 3), 255, dtype=np.uint8))
    node.visualization_marker_id = 1
    node.intrinsic_mat = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    node.distortion = np.zeros(5)
    node.marker_size = 0.07
    node.get_logger = lambda: SimpleNamespace(warn=lambda *_args, **_kwargs: None)
    return node


def test_selected_marker_overlay_draws_green_corner_center_and_preserves_header():
    node = _node()
    header = SimpleNamespace(frame_id="camera", stamp=SimpleNamespace())
    corners = [np.array([[[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]]])]

    ArucoNode._publish_visualization(
        node,
        SimpleNamespace(header=header),
        corners,
        np.array([[1]], dtype=np.int32),
        np.zeros((1, 1, 3)),
        np.array([[[0.0, 0.0, 0.5]]]),
    )

    assert len(node.visualization_pub.messages) == 1
    assert node.visualization_pub.messages[0].header is header
    assert np.array_equal(node.bridge.rendered[50, 50], [0, 255, 0])


def test_overlay_skips_color_conversion_without_subscribers():
    node = _node(subscribers=0)
    node.bridge.imgmsg_to_cv2 = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected conversion"))

    ArucoNode._publish_visualization(node, SimpleNamespace(header=object()), [], None, None, None)

    assert node.visualization_pub.messages == []
