from enum import Enum


class TargetType(Enum):
    PEN = "pen"
    CUBE = "cube"
    STONE = "stone"


class TaskState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    MOVING_TO_TARGET_ABOVE = "moving_to_target_above"
    MOVING_TO_TARGET = "moving_to_target"
    GRASPING = "grasping"
    LIFTING_TARGET = "lifting_target"
    MOVING_TO_BOX_ABOVE = "moving_to_box_above"
    DESCEND_TO_BOX = "descend_to_box"
    RELEASING = "releasing"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ERROR = "error"
