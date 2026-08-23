from dataclasses import dataclass
from enum import Enum
from typing import Optional

from geometry_msgs.msg import Pose


class GraspState(str, Enum):
    WAIT_READY = "WAIT_READY"
    PREGRASP_POSE = "PREGRASP_POSE"
    WAIT_G = "WAIT_G"
    COMPUTE = "COMPUTE"
    SELECT = "SELECT"
    PLAN = "PLAN"
    PREOPEN = "PREOPEN"
    MOVE_TO_APPROACH = "MOVE_TO_APPROACH"
    APPROACH_TO_GRASP = "APPROACH_TO_GRASP"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    RETURN_PREGRASP = "RETURN_PREGRASP"
    RECOVER = "RECOVER"
    FAILED = "failed"
    RETURN_PREGRASP_FAILED = "return_pregrasp_failed"
    RECOVERY_PREGRASP_FAILED = "recovery_pregrasp_failed"


@dataclass
class GraspCandidate:
    idx: int
    camera_pose: Pose
    score: Optional[float]
    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    preopen_positions: Optional[tuple[float, float]] = None
    base_pose: Optional[Pose] = None
    grasp: Optional[Pose] = None
    approach: Optional[Pose] = None
    lift: Optional[Pose] = None
    reject_reason: str = ""


    
