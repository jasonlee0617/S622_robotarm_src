#!/usr/bin/env python3
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseArray
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.abspath(os.path.join(PKG_ROOT, "..", ".."))
for path in (PKG_ROOT, os.path.join(SRC_ROOT, "pymoveit2"), os.path.join(SRC_ROOT, "manipulation_common")):
    if path not in sys.path:
        sys.path.insert(0, path)
pkg = sys.modules.get("graspnet_bringup")
if pkg is not None and hasattr(pkg, "__path__"):
    inner_pkg = os.path.join(PKG_ROOT, "graspnet_bringup")
    if inner_pkg not in list(pkg.__path__):
        pkg.__path__.append(inner_pkg)

pymoveit2_stub = types.ModuleType("pymoveit2")
pymoveit2_stub.MoveIt2 = object
sys.modules.setdefault("pymoveit2", pymoveit2_stub)

from graspnet_bringup.graspnet_inference_node import (  # noqa: E402
    GraspnetInferenceNode,
    _filter_grasp_group_by_width,
    _graspgroup_to_pose_metadata,
    _support_plane_keep_mask,
)
from graspnet_bringup.graspnetl_grasping_node import (  # noqa: E402
    _CAPTURE_TIME_TF_HISTORY_SEC,
    GraspCandidate,
    GraspnetVisualGraspingNode,
    _apply_orientation_correction,
    _candidate_indices,
    _close_positions_from_width,
    _make_lift_pose,
    _metadata_at,
    _pose_axis,
)


def pose(x=0.0, y=0.0, z=0.0, quat=(0.0, 0.0, 0.0, 1.0)):
    msg = Pose()
    msg.position.x = x
    msg.position.y = y
    msg.position.z = z
    msg.orientation.x = quat[0]
    msg.orientation.y = quat[1]
    msg.orientation.z = quat[2]
    msg.orientation.w = quat[3]
    return msg


class FakePublisher:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def publish(self, msg):
        self.events.append((self.name, msg))


class FakeGraspGroup:
    def __init__(self, widths):
        self.widths = np.asarray(widths, dtype=np.float32)

    def __len__(self):
        return len(self.widths)

    def __getitem__(self, index):
        return FakeGraspGroup(self.widths[index])


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))

    def error(self, message):
        self.messages.append(("error", message))


class GraspnetVisualGraspingNodeTest(unittest.TestCase):
    def test_wait_ready_moves_directly_to_pregrasp_pose(self):
        states = []
        node = SimpleNamespace(
            _tf_ready=lambda: True,
            compute_client=SimpleNamespace(wait_for_service=lambda timeout_sec: True),
            startup_motion_ready=lambda timeout_sec: True,
            _set_state=states.append,
        )

        GraspnetVisualGraspingNode._state_wait_ready(node)

        self.assertEqual(states, ["PREGRASP_POSE"])

    def test_lift_success_returns_to_pregrasp_while_holding_gripper(self):
        states = []
        candidate = SimpleNamespace(lift=pose())
        node = SimpleNamespace(
            _require_candidate=lambda: candidate,
            motion=SimpleNamespace(move_to_pose=lambda *args, **kwargs: True),
            ik_plugin="fairino",
            j2_constraint={},
            _motion_limits_kwargs=lambda: {},
            _set_state=states.append,
        )

        GraspnetVisualGraspingNode._state_lift(node)

        self.assertEqual(states, ["RETURN_PREGRASP"])

        reset = []
        node = SimpleNamespace(
            _move_to_pregrasp_pose=lambda: True,
            _reset_task_cache=lambda: reset.append(True),
            _set_state=states.append,
        )
        GraspnetVisualGraspingNode._state_return_pregrasp(node)

        self.assertEqual(reset, [True])
        self.assertEqual(states[-1], "DONE")

    def test_recoverable_failure_opens_gripper_then_returns_to_pregrasp_pose(self):
        events = []
        node = SimpleNamespace(
            get_logger=lambda: FakeLogger(),
            _set_state=lambda state: events.append(("state", state)),
            abort=SimpleNamespace(
                cancel_all_motion_now=lambda: events.append("cancel"),
                clear=lambda: events.append("clear"),
            ),
            motion=SimpleNamespace(
                control_gripper=lambda **kwargs: events.append(
                    ("gripper", kwargs["open_gripper"])
                ),
            ),
            _move_to_pregrasp_pose=lambda: events.append("pregrasp") or True,
            _reset_task_cache=lambda: events.append("reset"),
        )

        GraspnetVisualGraspingNode._recover(node, "compute_failed")

        self.assertEqual(
            events,
            [
                ("state", "RECOVER"),
                "cancel",
                "clear",
                ("gripper", True),
                "pregrasp",
                "reset",
                ("state", "WAIT_READY"),
            ],
        )

    def test_manual_stop_holds_position_without_returning_to_pregrasp(self):
        events = []
        node = SimpleNamespace(
            abort=SimpleNamespace(
                is_stop_requested=lambda: True,
                cancel_all_motion_now=lambda: events.append("cancel"),
            ),
            _recover=lambda reason: events.append(("recover", reason)),
        )

        GraspnetVisualGraspingNode._recover_unless_stop(node, "lift_failed")

        self.assertEqual(events, ["cancel"])

    def test_capture_time_tf_history_is_longer_than_confirmation_window(self):
        self.assertEqual(_CAPTURE_TIME_TF_HISTORY_SEC, 120.0)

    def test_camera_info_latch_decouples_fresh_rgbd_from_old_info_stamp(self):
        info = CameraInfo()
        info.header.frame_id = "camera_color_optical_frame"
        info.header.stamp.sec = 1
        info.width = 1280
        info.height = 720
        info.k[0] = info.k[4] = 907.77
        info.k[2] = 648.03
        info.k[5] = 360.25
        logger = FakeLogger()
        destroyed = []
        node = SimpleNamespace(
            _lock=threading.Lock(),
            _camera_info=None,
            camera_info_sub=object(),
            info_topic="/camera/camera/aligned_depth_to_color/camera_info",
            get_logger=lambda: logger,
            destroy_subscription=lambda subscription: destroyed.append(subscription),
            _latest=None,
        )

        GraspnetInferenceNode.on_camera_info(node, info)
        rgb = Image()
        depth = Image()
        rgb.header.stamp.sec = depth.header.stamp.sec = 99
        GraspnetInferenceNode.on_synced(node, rgb, depth)

        self.assertEqual(len(destroyed), 1)
        self.assertIsNone(node.camera_info_sub)
        self.assertIs(node._latest[0], rgb)
        self.assertIs(node._latest[1], depth)
        self.assertIs(node._latest[2], info)

    def test_capture_time_tf_failure_never_uses_latest_tf(self):
        class MissingTf:
            def __init__(self):
                self.calls = []

            def lookup_transform(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                raise RuntimeError("capture transform unavailable")

        tf_buffer = MissingTf()
        logger = FakeLogger()
        node = SimpleNamespace(
            base_frame="base_link",
            camera_frame="camera_color_optical_frame",
            _tf_buffer=tf_buffer,
            get_logger=lambda: logger,
        )
        header = SimpleNamespace(
            frame_id="camera_color_optical_frame",
            stamp=Time(sec=42, nanosec=100),
        )

        self.assertIsNone(GraspnetVisualGraspingNode._camera_pose_to_base(node, header, pose()))
        self.assertEqual(len(tf_buffer.calls), 1)
        self.assertIn("tf_at_capture_time_unavailable", logger.messages[0][1])
    def test_support_plane_filter_removes_table_and_keeps_elevated_points(self):
        points = np.array([
            [0.0, 0.0, 0.000],
            [0.1, 0.0, 0.003],
            [0.0, 0.1, 0.030],
            [0.1, 0.1, 0.050],
        ])

        keep = _support_plane_keep_mask(
            points,
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.eye(3),
            distance_threshold_m=0.005,
            max_tilt_deg=15.0,
        )

        np.testing.assert_array_equal(keep, [False, False, True, True])

    def test_support_plane_filter_rejects_vertical_plane(self):
        keep = _support_plane_keep_mask(
            np.array([[0.0, 0.0, 0.1]]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.eye(3),
            distance_threshold_m=0.005,
            max_tilt_deg=15.0,
        )

        self.assertIsNone(keep)

    def test_inference_width_filter_keeps_ordered_feasible_candidates(self):
        group = FakeGraspGroup([0.0626, 0.0400, 0.0050, 0.0802, 0.0610])

        filtered, count = _filter_grasp_group_by_width(group, 0.005, 0.061)

        self.assertEqual(count, 3)
        np.testing.assert_allclose(filtered.widths, [0.0400, 0.0050, 0.0610])

    def test_candidate_indices_keep_published_order(self):
        self.assertEqual(_candidate_indices(5, 3), [0, 1, 2])

    def test_build_candidates_binds_metadata_by_pose_order(self):
        msg = PoseArray()
        msg.poses = [pose(z=0.1), pose(z=0.2), pose(z=0.3)]
        metadata = [0.9, 0.04, 0.02, 0.8, 0.05, 0.03]
        node = SimpleNamespace(max_grasp_candidates=2)

        candidates = GraspnetVisualGraspingNode._build_candidates(node, msg, [0.1, 99.0], metadata)

        self.assertEqual([candidate.idx for candidate in candidates], [0, 1])
        self.assertEqual([candidate.score for candidate in candidates], [0.9, 0.8])
        self.assertEqual([candidate.width_m for candidate in candidates], [0.04, 0.05])
        self.assertEqual([candidate.depth_m for candidate in candidates], [0.02, 0.03])

    def test_metadata_missing_falls_back_to_scores_and_fixed_parameters(self):
        msg = PoseArray()
        msg.poses = [pose(z=0.1)]
        node = SimpleNamespace(max_grasp_candidates=1)

        candidate = GraspnetVisualGraspingNode._build_candidates(node, msg, [0.7], [])[0]

        self.assertEqual(candidate.score, 0.7)
        self.assertEqual(_metadata_at([], 0), (None, None, None))
        self.assertEqual(_close_positions_from_width(None, [0.0305, -0.0305], [0.01, -0.01]), (0.01, -0.01))

    def test_default_close_uses_configured_squeeze_and_lift(self):
        candidate = GraspCandidate(
            idx=0,
            camera_pose=pose(),
            score=1.0,
            width_m=0.04,
            depth_m=0.02,
            base_pose=pose(x=0.5, z=0.03),
        )
        node = SimpleNamespace(
            lift_distance=0.08,
            gripper_open_positions=(0.0305, -0.0305),
            gripper_close_positions=(0.01, -0.01),
            use_graspnet_width_for_final_close=False,
            graspnet_width_squeeze_m=0.01,
            graspnet_to_ee_rpy_deg=[0.0, 0.0, 0.0],
            approach_distance_m=0.08,
        )
        node._apply_orientation_correction = lambda p: _apply_orientation_correction(
            p, node.graspnet_to_ee_rpy_deg
        )
        node._prepare_grasp_pose = lambda p: GraspnetVisualGraspingNode._prepare_grasp_pose(node, p)

        GraspnetVisualGraspingNode.prepare_grasp_pose(node, candidate)

        self.assertAlmostEqual(candidate.lift.position.z, 0.11)
        self.assertAlmostEqual(candidate.approach.position.x, 0.5)
        self.assertEqual(candidate.close_positions, (0.01, -0.01))

    def test_optional_width_close_applies_squeeze(self):
        self.assertEqual(
            _close_positions_from_width(
                0.04,
                [0.0305, -0.0305],
                [0.01, -0.01],
                use_width=True,
                squeeze_m=0.005,
            ),
            (0.015, -0.015),
        )

    def test_graspnet_to_fairino_arm_adapter_maps_approach_to_tcp_z(self):
        raw = pose(quat=(0.619423, 0.443877, -0.557560, 0.329265))

        target = _apply_orientation_correction(raw, [90.0, 0.0, 90.0])

        np.testing.assert_allclose(_pose_axis(target, 2), _pose_axis(raw, 0), atol=1e-6)
        self.assertLess(float(np.degrees(np.arccos(np.dot(_pose_axis(target, 2), [0.0, 0.0, -1.0])))), 15.0)

    def test_validate_candidate_rejects_dangerous_side_grasp(self):
        safe_quat = R.from_matrix(np.diag([1.0, -1.0, -1.0])).as_quat()
        node = SimpleNamespace(
            min_grasp_z=0.005,
            min_grasp_width_m=0.005,
            max_grasp_width_m=0.061,
            max_approach_tilt_deg=35.0,
            max_jaw_z_abs=0.35,
        )
        node._reject_candidate = lambda cand, reason: setattr(cand, "reject_reason", reason)

        below_table_clearance = GraspCandidate(
            idx=0, camera_pose=pose(), score=1.0, width_m=0.04, grasp=pose(z=0.004, quat=safe_quat)
        )
        safe = GraspCandidate(
            idx=1, camera_pose=pose(), score=1.0, width_m=0.04, grasp=pose(z=0.006, quat=safe_quat)
        )
        side = GraspCandidate(
            idx=2, camera_pose=pose(), score=1.0, width_m=0.04, grasp=pose(z=0.006)
        )

        self.assertFalse(GraspnetVisualGraspingNode.validate_candidate(node, below_table_clearance))
        self.assertTrue(below_table_clearance.reject_reason.startswith("z_below_min"))
        self.assertTrue(GraspnetVisualGraspingNode.validate_candidate(node, safe))
        self.assertFalse(GraspnetVisualGraspingNode.validate_candidate(node, side))
        self.assertTrue(side.reject_reason.startswith("approach_tilt"))

    def test_selection_reaches_safe_candidate_after_first_five_rejections(self):
        logger = FakeLogger()
        candidates = [GraspCandidate(idx=index, camera_pose=pose(), score=1.0) for index in range(6)]
        node = SimpleNamespace(
            _grasp_msg=SimpleNamespace(header=object()),
            _candidates=candidates,
            get_logger=lambda: logger,
        )

        def transform(candidate, _header):
            candidate.base_pose = candidate.camera_pose
            return True

        def prepare(candidate):
            candidate.grasp = candidate.base_pose

        def validate(candidate):
            if candidate.idx < 5:
                candidate.reject_reason = "approach_tilt:90.0deg"
                return False
            return True

        node.transform_candidate = transform
        node.prepare_grasp_pose = prepare
        node.validate_candidate = validate
        node.plan_candidate = lambda _candidate: True

        selected = GraspnetVisualGraspingNode._select_executable_candidate(node)

        self.assertEqual(selected.idx, 5)
        self.assertIn("received=6 selected_idx=5", logger.messages[-1][1])

    def test_lift_pose_geometry(self):
        grasp = pose(x=0.5, y=0.0, z=0.03)

        lift = _make_lift_pose(grasp, 0.08)

        self.assertAlmostEqual(lift.position.z, 0.11)

    def test_reject_reason_for_low_z_and_plan_failure(self):
        candidate = GraspCandidate(idx=0, camera_pose=pose(), score=1.0, grasp=pose(z=0.01))
        node = SimpleNamespace(min_grasp_z=0.02)
        node._reject_candidate = lambda cand, reason: setattr(cand, "reject_reason", reason)

        self.assertFalse(GraspnetVisualGraspingNode.validate_candidate(node, candidate))
        self.assertTrue(candidate.reject_reason.startswith("z_below_min"))

        candidate = GraspCandidate(
            idx=1,
            camera_pose=pose(),
            score=0.5,
            grasp=pose(z=0.03),
            lift=pose(z=0.11),
        )
        node = SimpleNamespace()
        node._reject_candidate = lambda cand, reason: setattr(cand, "reject_reason", reason)
        node._can_plan_pose = lambda *args: False

        self.assertFalse(GraspnetVisualGraspingNode.plan_candidate(node, candidate))
        self.assertTrue(candidate.reject_reason.startswith("plan_failed"))

    def test_graspnet_array_metadata_columns(self):
        grasp_array = np.zeros((2, 17), dtype=np.float32)
        grasp_array[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
        grasp_array[:, 0] = [0.9, 0.8]
        grasp_array[:, 1] = [0.04, 0.05]
        grasp_array[:, 3] = [0.02, 0.03]
        grasp_array[:, 13:16] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        group = SimpleNamespace(grasp_group_array=grasp_array)

        poses_np, metadata = _graspgroup_to_pose_metadata(group)

        self.assertEqual(poses_np.shape, (2, 7))
        self.assertEqual(len(metadata), 2)
        for row, expected in zip(metadata, [(0.9, 0.04, 0.02), (0.8, 0.05, 0.03)]):
            for value, expected_value in zip(row, expected):
                self.assertAlmostEqual(value, expected_value)

    def test_publish_results_sends_scores_metadata_before_poses(self):
        events = []
        node = SimpleNamespace(
            score_pub=FakePublisher("scores", events),
            metadata_pub=FakePublisher("metadata", events),
            pose_pub=FakePublisher("poses", events),
        )
        poses_np = np.asarray([[0, 0, 0.1, 0, 0, 0, 1]], dtype=np.float32)
        metadata = [(0.9, 0.04, 0.02)]

        GraspnetInferenceNode._publish_results(node, poses_np, metadata, "camera", Time())

        self.assertEqual([name for name, _ in events], ["scores", "metadata", "poses"])
        self.assertAlmostEqual(events[0][1].data[0], 0.9)
        for value, expected_value in zip(events[1][1].data, [0.9, 0.04, 0.02]):
            self.assertAlmostEqual(value, expected_value)
        self.assertEqual(len(events[2][1].poses), 1)


if __name__ == "__main__":
    unittest.main()
