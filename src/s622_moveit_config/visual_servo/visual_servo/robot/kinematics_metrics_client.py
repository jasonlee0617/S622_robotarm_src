from __future__ import annotations


class KinematicsMetricsClient:
    """Placeholder boundary for planning/kinematics metric services.

    The current visual servo runtime still relies on MoveIt Servo status topics
    for online singularity feedback. This class marks the integration point for
    future calls into fairino_planning_ros metrics services without coupling the
    task node to service-specific message types.
    """

    def __init__(self, node):
        self.node = node

    def available(self) -> bool:
        return False
