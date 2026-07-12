from enum import Enum


class TargetType(Enum):
    ELONGATED_OBJECT = "elongated_object"
    CUBE = "cube"


class TaskState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    MOVING_TO_TARGET_ABOVE = "moving_to_target_above"
    SERVO_TRACK_ABOVE = "servo_track_above"
    SERVO_HALT_RECOVERY = "servo_halt_recovery"
    MOVING_TO_GRASP_GLOBAL = "moving_to_grasp_global"
    GRASPING = "grasping"
    LIFTING_TARGET = "lifting_target"
    SEARCHING_BOX = "searching_box"
    SERVO_TRACK_TO_BOX = "servo_track_to_box"
    MOVING_TO_BOX = "moving_to_box"
    RELEASING = "releasing"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ERROR = "error"
