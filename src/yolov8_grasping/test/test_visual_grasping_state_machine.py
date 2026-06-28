from types import SimpleNamespace

from yolov8_grasping.task.task_types import TargetType, TaskState
from yolov8_grasping.task.visual_grasping_state_machine import VisualGraspingStateMachine


class _Logger:
    def info(self, _msg):
        pass

    def warn(self, _msg):
        pass

    def error(self, _msg):
        pass


class _TargetSelector:
    def __init__(self, priority):
        self._priority = list(priority)

    def _resolve_priority(self):
        return list(self._priority)

    def msg_age_sec(self, _stamp):
        return 0.0


class _DetectionCache:
    def __init__(self, target_msg, target_rpy, box_msg=None):
        self._target_msg = target_msg
        self._target_rpy = target_rpy
        self.box_pos = box_msg

    def get_position(self, target):
        name = target if isinstance(target, str) else target.value
        return self._target_msg if name == "cube" else None

    def get_rpy(self, target):
        name = target if isinstance(target, str) else target.value
        return self._target_rpy if name == "cube" else None


class _TfTools:
    ready = True

    def camera_point_to_base(self, point):
        return point.point


class _PoseTools:
    def make_pose(self, x, y, z, roll_deg, pitch_deg, yaw_deg):
        return SimpleNamespace(
            position=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            attitude=(float(roll_deg), float(pitch_deg), float(yaw_deg)),
        )


def _point(x, y, z):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=object()),
        point=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
    )


def _make_node(box_msg=None):
    return SimpleNamespace(
        target_selector=_TargetSelector(["cube", "pen"]),
        det_cache=_DetectionCache(
            target_msg=_point(0.1, 0.2, 0.03),
            target_rpy={"roll": 0.0, "pitch": 0.0, "yaw": 0.1},
            box_msg=box_msg,
        ),
        tf_tools=_TfTools(),
        pose_tools=_PoseTools(),
        grasp_profiles={
            TargetType.CUBE: SimpleNamespace(roll=0.0, pitch=-180.0, yaw_offset=-135.0, above_z=0.05, grasp_z=0.01)
        },
        place_offset=0.2,
        detection_timeout=3.0,
        current_state=TaskState.SEARCHING,
        active_target=None,
        active_target_rpy=None,
        box_search_pose_tried=False,
        box_search_pose=object(),
        poses={},
        control_gripper=lambda _open: True,
        get_logger=lambda: _Logger(),
    )


def test_searching_does_not_require_box():
    node = _make_node(box_msg=None)
    machine = VisualGraspingStateMachine(node)
    machine._pause = lambda: None

    machine._on_searching()

    assert node.active_target == TargetType.CUBE
    assert node.current_state == TaskState.MOVING_TO_TARGET_ABOVE
    assert set(node.poses) == {"target_above", "target_grasp", "target_lift"}


def test_searching_box_builds_place_poses():
    node = _make_node(box_msg=_point(0.4, 0.5, 0.06))
    node.current_state = TaskState.SEARCHING_BOX
    node.active_target = TargetType.CUBE
    node.active_target_rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.1}
    node.poses = {"target_lift": object()}
    machine = VisualGraspingStateMachine(node)

    machine._on_searching_box()

    assert node.current_state == TaskState.MOVING_TO_BOX_ABOVE
    assert "box_above" in node.poses
    assert "descend_to_box" in node.poses


def test_searching_box_moves_to_search_pose_when_box_missing():
    node = _make_node(box_msg=None)
    node.current_state = TaskState.SEARCHING_BOX
    node.active_target = TargetType.CUBE
    node.active_target_rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.1}
    machine = VisualGraspingStateMachine(node)

    machine._on_searching_box()

    assert node.box_search_pose_tried is True
    assert node.current_state == TaskState.MOVING_TO_BOX_SEARCH_POSE
    assert node.poses["box_search_pose"] is node.box_search_pose


def test_searching_box_errors_after_search_pose_when_box_missing():
    node = _make_node(box_msg=None)
    node.current_state = TaskState.SEARCHING_BOX
    node.active_target = TargetType.CUBE
    node.active_target_rpy = {"roll": 0.0, "pitch": 0.0, "yaw": 0.1}
    node.box_search_pose_tried = True
    machine = VisualGraspingStateMachine(node)

    machine._on_searching_box()

    assert node.current_state == TaskState.ERROR
