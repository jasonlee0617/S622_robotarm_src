import os
import pathlib
from datetime import datetime

SAMPLES_DIRECTORY = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/samples'))
CALIBRATIONS_DIRECTORY = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/calibrations'))


def resolve_storage_directory(value) -> pathlib.Path | None:
    """Resolve an optional shared calibration/sample directory."""
    text = str(value or '').strip()
    if not text:
        return None
    return pathlib.Path(os.path.expandvars(os.path.expanduser(text)))


def snapshot_filepath(directory: pathlib.Path, name: str, suffix: str, snapshot_id: str | None = None) -> pathlib.Path:
    stem = name if not snapshot_id else f'{name}_{snapshot_id}'
    return directory / f'{stem}{suffix}'


def latest_snapshot_filepath(directory: pathlib.Path, name: str, suffix: str) -> pathlib.Path | None:
    matches = [path for path in directory.glob(f'{name}_*{suffix}') if path.is_file()]
    return max(matches, default=None, key=lambda path: path.name)


def load_filepath(directory: pathlib.Path, name: str, suffix: str, timestamped: bool) -> pathlib.Path:
    if timestamped:
        latest = latest_snapshot_filepath(directory, name, suffix)
        if latest is not None:
            return latest
    return snapshot_filepath(directory, name, suffix)


def next_snapshot_id(directory: pathlib.Path, name: str) -> str:
    """Return a timestamp that will not overwrite an existing pair."""
    base = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = base
    index = 1
    while (
        snapshot_filepath(directory, name, '.calib', candidate).exists()
        or snapshot_filepath(directory, name, '.samples', candidate).exists()
    ):
        candidate = f'{base}_{index:02d}'
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
