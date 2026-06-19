"""Shared types for sample management: families, specs, records, quality."""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Family constants
# ---------------------------------------------------------------------------

class CandidateFamily:
    SPHERE_ANCHOR = "sphere_anchor"
    SPHERE_HEIGHT = "sphere_height"
    SPHERE_SHELL = "sphere_shell"
    SPHERE_ROLL_COVERAGE = "sphere_roll_coverage"


FAMILY_EXECUTION_ORDER = [
    "sphere_anchor",
    "sphere_height",
    "sphere_shell",
    "sphere_roll_coverage",
]


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseOffsetPose:
    """A single family-based base-offset candidate record.

    Canonical definition — imported by collector_config for YAML parsing.
    """

    label: str = ""
    family: str = ""
    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    removable: bool = False
    intent: str = ""
    observability_axis: str = "none"  # pitch | yaw | roll | none
    dedup_protected: bool = False     # skip normal dt/dr dedup


@dataclass(frozen=True)
class CandidateSpec:
    """Resolved candidate with a unique key for dedup."""

    source: str
    base_x: float
    base_y: float
    base_z: float
    pitch: float
    yaw: float
    roll: float
    family: str
    removable: bool
    intent: str = ""
    observability_axis: str = "none"
    dedup_protected: bool = False

    def exact_key(self):
        """Exact-pose key for protected dedup."""
        return (
            round(self.base_x, 4), round(self.base_y, 4), round(self.base_z, 4),
            round(self.pitch, 2), round(self.yaw, 2), round(self.roll, 2),
        )


@dataclass(frozen=True)
class AcceptedSampleQuality:
    center_error_px: float
    margin_px: float
    marker_side_px: float
    distance_m: float
    camera_model_error_px: float
    center_std_px: float
    depth_std_m: float
    angle_std_deg: float
    marker_note: str
    model_note: str
    stable_note: str


@dataclass(frozen=True)
class AcceptedSampleRecord:
    robot_pose: object
    tracking_pose: object
    family: str
    spec: CandidateSpec
    quality: AcceptedSampleQuality
    candidate_idx: int
    candidate_description: str
    recenter_attempted: bool
    recenter_strict_converged: bool
    removable: bool
