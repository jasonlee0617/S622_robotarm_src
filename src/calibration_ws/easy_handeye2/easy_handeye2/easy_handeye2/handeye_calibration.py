import pathlib

import yaml
from easy_handeye2_msgs.msg import HandeyeCalibration, HandeyeCalibrationParameters
from rclpy.node import Node, ParameterDescriptor, ParameterType
from rosidl_runtime_py import set_message_fields, message_to_yaml

from . import (
    CALIBRATIONS_DIRECTORY,
    TIMESTAMPED_CALIBRATION_NAME,
    load_filepath,
    resolve_storage_directory,
    snapshot_filepath,
    typed_snapshot_id,
)


def filepath_for_calibration(
    name, storage_directory=None, snapshot_id=None, calibration_type=''
) -> pathlib.Path:
    directory = resolve_storage_directory(storage_directory) or CALIBRATIONS_DIRECTORY
    return snapshot_filepath(
        directory,
        TIMESTAMPED_CALIBRATION_NAME if snapshot_id is not None else name,
        '.calib',
        typed_snapshot_id(snapshot_id, calibration_type),
    )


class HandeyeCalibrationParametersProvider:
    def __init__(self, node: Node):
        self.node = node
        # declare and read parameters
        self.node.declare_parameter('name', '', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.node.declare_parameter('calibration_type', '', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.node.declare_parameter('robot_base_frame', '', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.node.declare_parameter('robot_effector_frame', '', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.node.declare_parameter('tracking_base_frame', '', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.node.declare_parameter('tracking_marker_frame', '', descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.node.declare_parameter('freehand_robot_movement', True)

    def read(self):
        ret = HandeyeCalibrationParameters(
            name=self.node.get_parameter('name').get_parameter_value().string_value,
            calibration_type=self.node.get_parameter('calibration_type').get_parameter_value().string_value,
            robot_base_frame=self.node.get_parameter('robot_base_frame').get_parameter_value().string_value,
            robot_effector_frame=self.node.get_parameter('robot_effector_frame').get_parameter_value().string_value,
            tracking_base_frame=self.node.get_parameter('tracking_base_frame').get_parameter_value().string_value,
            tracking_marker_frame=self.node.get_parameter('tracking_marker_frame').get_parameter_value().string_value,
            freehand_robot_movement=self.node.get_parameter('freehand_robot_movement').get_parameter_value().bool_value,
        )
        return ret


def load_calibration(name, storage_directory=None) -> HandeyeCalibration:
    directory = resolve_storage_directory(storage_directory) or CALIBRATIONS_DIRECTORY
    timestamped = resolve_storage_directory(storage_directory) is not None
    filepath = load_filepath(
        directory,
        TIMESTAMPED_CALIBRATION_NAME if timestamped else name,
        '.calib',
        timestamped=timestamped,
    )
    with open(filepath) as f:
        m = yaml.full_load(f.read())
    ret = HandeyeCalibration()
    set_message_fields(ret, m)
    return ret


def save_calibration(calibration: HandeyeCalibration, storage_directory=None, snapshot_id=None) -> pathlib.Path:
    directory = resolve_storage_directory(storage_directory) or CALIBRATIONS_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    filepath = filepath_for_calibration(
        calibration.parameters.name,
        storage_directory,
        snapshot_id,
        calibration.parameters.calibration_type,
    )
    with open(filepath, 'x' if snapshot_id is not None else 'w') as f:
        f.write(message_to_yaml(calibration))
    return filepath
