import rclpy


class TargetSelector:
    def __init__(self, node, target_priority=None):
        self.node = node
        self.preferred_target = "pen"
        self.detection_timeout = 3.0
        self.target_priority = list(target_priority or ["pen", "cube", "stone"])

    def set_preference(self, preferred_target: str):
        self.preferred_target = str(preferred_target).lower().strip()

    def set_timeout(self, timeout_sec: float):
        self.detection_timeout = float(timeout_sec)

    def select_target(self, target_type_cls, cache):
        available = {}
        for target in target_type_cls:
            if self._pair_valid(cache.get_position(target), cache.get_rpy(target), cache.box_pos):
                available[target.value] = target

        if not available:
            return None
        if self.preferred_target in available:
            return available[self.preferred_target]
        for name in self.target_priority:
            if name in available:
                return available[name]
        return next(iter(available.values()))

    def _pair_valid(self, obj_pos, obj_rpy: dict, box_pos) -> bool:
        if obj_pos is None or obj_rpy is None or box_pos is None:
            return False
        obj_age = self._msg_age_sec(obj_pos.header.stamp)
        box_age = self._msg_age_sec(box_pos.header.stamp)
        return obj_age < self.detection_timeout and box_age < self.detection_timeout

    def _msg_age_sec(self, msg_header_stamp):
        now = self.node.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg_header_stamp)
        return (now - msg_time).nanoseconds / 1e9
