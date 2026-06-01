from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ServoStatusAction(str, Enum):
    OK = "ok"
    DECELERATE = "decelerate"
    HALT_RECOVERY = "halt_recovery"


@dataclass(frozen=True)
class ServoStatusDecision:
    action: ServoStatusAction
    message: str


class ServoStatusPolicy:
    """Classify MoveIt Servo status codes into control-loop actions."""

    def __init__(self, decel_codes: set[int], halt_codes: set[int]):
        self.decel_codes = {int(code) for code in decel_codes}
        self.halt_codes = {int(code) for code in halt_codes}

    def decide(self, code: int) -> ServoStatusDecision:
        code = int(code)
        if code == 0:
            return ServoStatusDecision(ServoStatusAction.OK, "")
        if code in self.decel_codes:
            return ServoStatusDecision(
                ServoStatusAction.DECELERATE,
                f"Servo status {code} (decelerate/warning), keep tracking with capped cmd",
            )
        if code in self.halt_codes:
            return ServoStatusDecision(
                ServoStatusAction.HALT_RECOVERY,
                f"Servo status {code} -> HALT recovery",
            )
        return ServoStatusDecision(
            ServoStatusAction.HALT_RECOVERY,
            f"Servo status {code} (unknown) -> conservative recovery",
        )
