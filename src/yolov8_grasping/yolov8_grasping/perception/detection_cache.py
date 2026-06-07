from std_msgs.msg import Float32MultiArray


class DetectionCache:
    def __init__(self, node):
        self.node = node
        self.reset()

    def reset(self):
        self.pen_pos = None
        self.cube_pos = None
        self.box_pos = None
        self.stone_pos = None
        self.pen_rpy = None
        self.cube_rpy = None
        self.stone_rpy = None

    def on_pen_pos(self, msg):
        self.pen_pos = msg

    def on_cube_pos(self, msg):
        self.cube_pos = msg

    def on_box_pos(self, msg):
        self.box_pos = msg

    def on_stone_pos(self, msg):
        self.stone_pos = msg

    def on_pen_rpy(self, msg: Float32MultiArray):
        self.pen_rpy = self._parse_rpy(msg, "/pen_rpy")

    def on_cube_rpy(self, msg: Float32MultiArray):
        self.cube_rpy = self._parse_rpy(msg, "/cube_rpy")

    def on_stone_rpy(self, msg: Float32MultiArray):
        self.stone_rpy = self._parse_rpy(msg, "/stone_rpy")

    def get_position(self, target):
        return {
            "pen": self.pen_pos,
            "cube": self.cube_pos,
            "stone": self.stone_pos,
        }.get(target.value)

    def get_rpy(self, target):
        return {
            "pen": self.pen_rpy,
            "cube": self.cube_rpy,
            "stone": self.stone_rpy,
        }.get(target.value)

    def _parse_rpy(self, msg: Float32MultiArray, topic_name: str):
        if len(msg.data) >= 3:
            return {"roll": msg.data[0], "pitch": msg.data[1], "yaw": msg.data[2]}
        self.node.get_logger().warn(f"⚠ {topic_name} format error")
        return None
