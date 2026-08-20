from visual_servo_bringup.task.task_types import TargetType


def test_target_type_uses_canonical_elongated_object_name():
    assert TargetType.ELONGATED_OBJECT.value == "elongated_object"
