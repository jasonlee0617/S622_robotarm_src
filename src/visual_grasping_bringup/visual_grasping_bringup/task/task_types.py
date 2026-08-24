from enum import Enum


class TargetType(Enum):
    ELONGATED_OBJECT = "elongated_object"
    CUBE = "cube"
    STONE = "stone"


class TaskState(Enum):
    IDLE = "idle"
    WAIT_G = "wait_g"
    SEARCHING = "searching"
    OPEN_GRIPPER = "open_gripper"
    MOVING_TO_TARGET_ABOVE = "moving_to_target_above"
    MOVING_TO_TARGET = "moving_to_target"
    GRASPING = "grasping"
    LIFTING_TARGET = "lifting_target"
    SEARCHING_BOX = "searching_box"
    MOVING_TO_BOX_ABOVE = "moving_to_box_above"
    DESCEND_TO_BOX = "descend_to_box"
    RELEASING = "releasing"
    RETURNING_PREGRASP_POSE = "returning_pregrasp_pose"
    COMPLETED = "completed"
    ERROR = "error"
