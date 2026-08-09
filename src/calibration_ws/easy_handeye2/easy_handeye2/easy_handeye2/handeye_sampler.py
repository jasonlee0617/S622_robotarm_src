import pathlib
from typing import Optional

import easy_handeye2_msgs.msg
import tf2_ros
import yaml
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from rosidl_runtime_py import message_to_yaml, set_message_fields
from easy_handeye2_msgs.msg import Sample, SampleList
import rclpy
from rclpy.time import Duration, Time

from easy_handeye2 import (
    SAMPLES_DIRECTORY,
    TIMESTAMPED_CALIBRATION_NAME,
    load_filepath,
    resolve_storage_directory,
    snapshot_filepath,
    typed_snapshot_id,
)
from easy_handeye2.handeye_calibration import HandeyeCalibrationParameters

import easy_handeye2


class HandeyeSampler:
    """
    Manages the samples acquired from tf.
    """

    def __init__(self, node: rclpy.node.Node, handeye_parameters: HandeyeCalibrationParameters, storage_directory=None):
        self.node = node
        self.handeye_parameters = handeye_parameters
        self.storage_directory = storage_directory

        # Time policy parameters (v8).
        node.declare_parameter("sample_time_policy", "latest_common")
        node.declare_parameter("sample_time_offset_sec", 0.0)
        node.declare_parameter("sample_lookup_timeout_sec", 1.0)

        # tf structures
        self.tfBuffer: tf2_ros.Buffer = Buffer(cache_time=Duration(seconds=2), node=node)
        self.tfListener: tf2_ros.TransformListener = TransformListener(self.tfBuffer, self.node, spin_thread=True)
        self.tfBroadcaster: tf2_ros.TransformBroadcaster = TransformBroadcaster(self.node)

        self.samples: easy_handeye2.msg.SampleList = SampleList()
        self.samples.parameters = handeye_parameters

    def _time_policy(self) -> str:
        return str(self.node.get_parameter("sample_time_policy").value).strip().lower()

    def _time_offset_sec(self) -> float:
        return float(self.node.get_parameter("sample_time_offset_sec").value)

    def _lookup_timeout(self) -> Duration:
        return Duration(seconds=float(self.node.get_parameter("sample_lookup_timeout_sec").value))

    def wait_for_tf_init(self) -> bool:
        base_frame = self.handeye_parameters.robot_base_frame
        effector_frame = self.handeye_parameters.robot_effector_frame
        camera_frame = self.handeye_parameters.tracking_base_frame
        marker_frame = self.handeye_parameters.tracking_marker_frame
        self.node.get_logger().info('Checking that the expected transforms are available in tf')
        self.node.get_logger().info(f'Robot transform: {base_frame} -> {effector_frame}')
        self.node.get_logger().info(f'Tracking transform: {camera_frame} -> {marker_frame}')
        try:
            self.tfBuffer.lookup_transform(base_frame, effector_frame, Time(), Duration(seconds=10))
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(
                'The specified tf frames for the robot base and hand do not seem to be connected')
            self.node.get_logger().error(f'Underlying tf exception: {e}')
            return False
        try:
            self.tfBuffer.lookup_transform(camera_frame, marker_frame, Time(), Duration(seconds=10))
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(
                'The specified tf frames for the tracking system base/camera and marker do not seem to be connected')
            self.node.get_logger().error(f'Underlying tf exception: {e}')
            return False
        self.node.get_logger().info('All expected transforms are available on tf; ready to take samples')
        return True

    def _get_transforms(self, time: Optional[rclpy.time.Time] = None) -> Sample | None:
        policy = self._time_policy()
        if time is None:
            if policy == "latest_common":
                time = Time()
            elif policy == "offset":
                offset_ns = int(self._time_offset_sec() * 1e9)
                time = self.node.get_clock().now() - rclpy.time.Duration(nanoseconds=offset_ns)
            else:
                time = self.node.get_clock().now()

        timeout = self._lookup_timeout()
        try:
            if self.handeye_parameters.calibration_type == 'eye_in_hand':
                robot = self.tfBuffer.lookup_transform(
                    self.handeye_parameters.robot_base_frame,
                    self.handeye_parameters.robot_effector_frame, time, timeout)
            else:
                robot = self.tfBuffer.lookup_transform(
                    self.handeye_parameters.robot_effector_frame,
                    self.handeye_parameters.robot_base_frame, time, timeout)
            tracking = self.tfBuffer.lookup_transform(
                self.handeye_parameters.tracking_base_frame,
                self.handeye_parameters.tracking_marker_frame, time, timeout)
        except tf2_ros.ExtrapolationException as e:
            self.node.get_logger().error(
                f'TF extrapolation (policy={policy} time={time}): {e}', throttle_duration_sec=5.0)
            return None

        ret = Sample()
        ret.robot = robot.transform
        ret.tracking = tracking.transform
        return ret

    def current_transforms(self) -> Sample | None:
        return self._get_transforms()

    def take_sample(self) -> bool:
        try:
            self.node.get_logger().info("Taking a sample...")
            sample = self._get_transforms()
            if sample is None:
                self.node.get_logger().error(
                    "take_sample failed: could not retrieve transforms. "
                    f"Sample list size unchanged ({len(self.samples.samples)}). "
                    f"TF frames: {self.tfBuffer.all_frames_as_string()}"
                )
                return False
            new_samples = self.samples.samples
            new_samples.append(sample)
            self.samples.samples = new_samples
            self.node.get_logger().info(f"Got a sample (total={len(self.samples.samples)})")
            return True
        except Exception as exc:
            self.node.get_logger().error(f"take_sample exception: {exc}")
            return False

    def remove_sample(self, index: int) -> int:
        if 0 <= index < len(self.samples.samples):
            new_samples = self.samples.samples
            del new_samples[index]
            self.samples.samples = new_samples
        return len(self.samples.samples)

    def get_samples(self) -> easy_handeye2_msgs.msg.SampleList:
        return self.samples

    def _directory(self) -> pathlib.Path:
        return resolve_storage_directory(self.storage_directory) or SAMPLES_DIRECTORY

    def _filepath_for_samplelist(self, snapshot_id=None) -> pathlib.Path:
        return snapshot_filepath(
            self._directory(),
            TIMESTAMPED_CALIBRATION_NAME if snapshot_id is not None else self.handeye_parameters.name,
            '.samples',
            typed_snapshot_id(snapshot_id, self.handeye_parameters.calibration_type),
        )

    def load_samples(self) -> bool:
        directory = self._directory()
        timestamped = resolve_storage_directory(self.storage_directory) is not None
        filepath = load_filepath(
            directory,
            TIMESTAMPED_CALIBRATION_NAME if timestamped else self.handeye_parameters.name,
            '.samples',
            timestamped=timestamped,
        )
        with open(filepath) as f:
            m = yaml.full_load(f.read())
            ret = SampleList()
            set_message_fields(ret, m)
            self.samples = ret
        return True

    def save_samples(self, snapshot_id=None) -> pathlib.Path:
        directory = self._directory()
        directory.mkdir(parents=True, exist_ok=True)
        filepath = self._filepath_for_samplelist(snapshot_id)
        with open(filepath, 'x' if snapshot_id is not None else 'w') as f:
            f.write(message_to_yaml(self.samples))
        return filepath
