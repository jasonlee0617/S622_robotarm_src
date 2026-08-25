from enum import Enum


class TargetType(Enum):
    ARUCO = "aruco"
    ELONGATED_OBJECT = "elongated_object"
    CUBE = "cube"
    BOX = "box"
    STONE = "stone"


class TaskState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    MOVING_TO_TARGET_ABOVE = "moving_to_target_above"
    SERVO_TRACK = "servo_track"
    SERVO_HALT_RECOVERY = "servo_halt_recovery"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ERROR = "error"
