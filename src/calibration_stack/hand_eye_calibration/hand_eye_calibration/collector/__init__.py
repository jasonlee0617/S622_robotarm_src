"""Auto-calibration collector package.

Exports public APIs for collector modules: types, governor, store,
geometry, vision, config, validation, and the main node.
"""

from .bootstrap import (
    _CV2_IMPORT_NOTE,
    _IMAGE_CHANNELS_BY_ENCODING,
    _PYTHON_SITE_NOTE,
    _import_cv2_with_aruco,
    _prefer_system_python_extensions,
    _script_build_stamp,
)
from .config import (
    CollectorFramesConfig,
    CollectorMotionConfig,
    CollectorSamplingConfig,
    load_collector_config,
)
from .geometry import CandidatePose, CollectorGeometry, TransformMatrix
from .sample_governor import SampleSetGovernor
from .sample_store import SampleManager
from .sample_types import (
    FAMILY_EXECUTION_ORDER,
    AcceptedSampleQuality,
    AcceptedSampleRecord,
    BaseOffsetPose,
    CandidateFamily,
    CandidateSpec,
)
from .validation import CalibrationValidator
from .vision import (
    QUALITY_CAMERA_MODEL,
    QUALITY_SAMPLING,
    QUALITY_STARTUP,
    ArucoObservation,
    CameraInfoState,
    ImageFrameStatus,
    StableWindowMetrics,
    VisionQualityGate,
)
from .auto_calibration_collector import AutoCalibrationCollector, main
from .session import CollectorExecutionSession

__all__ = [
    "_CV2_IMPORT_NOTE",
    "_IMAGE_CHANNELS_BY_ENCODING",
    "_PYTHON_SITE_NOTE",
    "_import_cv2_with_aruco",
    "_prefer_system_python_extensions",
    "_script_build_stamp",
    "AcceptedSampleQuality",
    "AcceptedSampleRecord",
    "ArucoObservation",
    "AutoCalibrationCollector",
    "BaseOffsetPose",
    "CalibrationValidator",
    "CameraInfoState",
    "CandidateFamily",
    "CandidatePose",
    "CandidateSpec",
    "CollectorExecutionSession",
    "CollectorFramesConfig",
    "CollectorGeometry",
    "CollectorMotionConfig",
    "CollectorSamplingConfig",
    "FAMILY_EXECUTION_ORDER",
    "ImageFrameStatus",
    "QUALITY_CAMERA_MODEL",
    "QUALITY_SAMPLING",
    "QUALITY_STARTUP",
    "SampleManager",
    "SampleSetGovernor",
    "StableWindowMetrics",
    "TransformMatrix",
    "VisionQualityGate",
    "load_collector_config",
    "main",
]
