import os
import pathlib
from datetime import datetime

SAMPLES_DIRECTORY = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/samples'))
CALIBRATIONS_DIRECTORY = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/calibrations'))
CALIBRATION_TYPES = ('eye_in_hand', 'eye_on_base')
TIMESTAMPED_CALIBRATION_NAME = 'robot_calibration'


def resolve_storage_directory(value) -> pathlib.Path | None:
    """Resolve an optional shared calibration/sample directory."""
    text = str(value or '').strip()
    if not text:
        return None
    return pathlib.Path(os.path.expandvars(os.path.expanduser(text)))


def snapshot_filepath(directory: pathlib.Path, name: str, suffix: str, snapshot_id: str | None = None) -> pathlib.Path:
    stem = name if not snapshot_id else f'{name}_{snapshot_id}'
    return directory / f'{stem}{suffix}'


def typed_snapshot_id(snapshot_id: str | None, calibration_type: str) -> str | None:
    """Append the validated calibration type to a timestamp/collision id."""
    if snapshot_id is None:
        return None
    calibration_type = str(calibration_type).strip()
    if calibration_type not in CALIBRATION_TYPES:
        raise ValueError(
            f'calibration_type must be one of {CALIBRATION_TYPES}, got {calibration_type!r}'
        )
    snapshot_id = str(snapshot_id).strip()
    for known_type in CALIBRATION_TYPES:
        if snapshot_id.endswith(f'_{known_type}'):
            if known_type != calibration_type:
                raise ValueError(
                    f'snapshot id {snapshot_id!r} does not match calibration_type '
                    f'{calibration_type!r}'
                )
            return snapshot_id
    return f'{snapshot_id}_{calibration_type}'


def latest_snapshot_filepath(directory: pathlib.Path, name: str, suffix: str) -> pathlib.Path | None:
    matches = [path for path in directory.glob(f'{name}_*{suffix}') if path.is_file()]

    def snapshot_order(path: pathlib.Path):
        snapshot_id = path.name[len(name) + 1:-len(suffix)]
        for calibration_type in CALIBRATION_TYPES:
            type_suffix = f'_{calibration_type}'
            if snapshot_id.endswith(type_suffix):
                snapshot_id = snapshot_id[:-len(type_suffix)]
                break
        return snapshot_id, path.name

    return max(matches, default=None, key=snapshot_order)


def load_filepath(directory: pathlib.Path, name: str, suffix: str, timestamped: bool) -> pathlib.Path:
    if timestamped:
        latest = latest_snapshot_filepath(directory, name, suffix)
        if latest is not None:
            return latest
    return snapshot_filepath(directory, name, suffix)


def next_snapshot_id(directory: pathlib.Path, name: str, calibration_type: str) -> str:
    """Return a timestamp that will not overwrite an existing pair."""
    name = TIMESTAMPED_CALIBRATION_NAME
    base = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = typed_snapshot_id(base, calibration_type)
    index = 1
    while (
        snapshot_filepath(directory, name, '.calib', candidate).exists()
        or snapshot_filepath(directory, name, '.samples', candidate).exists()
    ):
        candidate = typed_snapshot_id(f'{base}_{index:02d}', calibration_type)
        index += 1
    return candidate

# CALIBRATION_NAMESPACE = ''
CALIBRATION_NAMESPACE = '/easy_handeye2/calibration/'

LIST_ALGORITHMS_TOPIC = CALIBRATION_NAMESPACE + 'list_algorithms'
SET_ALGORITHM_TOPIC = CALIBRATION_NAMESPACE + 'set_algorithm'
GET_CURRENT_TRANSFORMS_TOPIC = CALIBRATION_NAMESPACE + 'get_current_transforms'
GET_SAMPLE_LIST_TOPIC = CALIBRATION_NAMESPACE + 'get_sample_list'
TAKE_SAMPLE_TOPIC = CALIBRATION_NAMESPACE + 'take_sample'
REMOVE_SAMPLE_TOPIC = CALIBRATION_NAMESPACE + 'remove_sample'
SAVE_SAMPLES_TOPIC = CALIBRATION_NAMESPACE + 'save_samples'
LOAD_SAMPLES_TOPIC = CALIBRATION_NAMESPACE + 'load_samples'
COMPUTE_CALIBRATION_TOPIC = CALIBRATION_NAMESPACE + 'compute_calibration'
SAVE_CALIBRATION_TOPIC = CALIBRATION_NAMESPACE + 'save_calibration'

CHECK_STARTING_POSE_TOPIC = CALIBRATION_NAMESPACE + 'check_starting_pose'
ENUMERATE_TARGET_POSES_TOPIC = CALIBRATION_NAMESPACE + 'enumerate_target_poses'
SELECT_TARGET_POSE_TOPIC = CALIBRATION_NAMESPACE + 'select_target_pose'
PLAN_TO_SELECTED_TARGET_POSE_TOPIC = CALIBRATION_NAMESPACE + 'plan_to_selected_target_pose'
EXECUTE_PLAN_TOPIC = CALIBRATION_NAMESPACE + 'execute_plan'

# Optional strict manual-calibration companion.  The stock GUI keeps working
# when these services are absent and switches to the assisted flow when they
# are available.
MANUAL_ASSISTANT_NAMESPACE = '/manual_calibration_assistant/'
MANUAL_ASSISTANT_STATUS_SERVICE = MANUAL_ASSISTANT_NAMESPACE + 'status'
MANUAL_ASSISTANT_VALIDATE_SERVICE = MANUAL_ASSISTANT_NAMESPACE + 'validate_latest'
MANUAL_ASSISTANT_REMOVE_SERVICE = MANUAL_ASSISTANT_NAMESPACE + 'remove_latest'
MANUAL_ASSISTANT_SAVE_SERVICE = MANUAL_ASSISTANT_NAMESPACE + 'save'
