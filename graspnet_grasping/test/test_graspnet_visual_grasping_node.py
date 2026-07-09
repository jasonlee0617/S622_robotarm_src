#!/usr/bin/env python3
import os
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseArray
from scipy.spatial.transform import Rotation as R

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.abspath(os.path.join(PKG_ROOT, ".."))
for path in (PKG_ROOT, os.path.join(SRC_ROOT, "pymoveit2"), os.path.join(SRC_ROOT, "manipulation_common")):
    if path not in sys.path:
        sys.path.insert(0, path)
pkg = sys.modules.get("graspnet_grasping")
if pkg is not None and hasattr(pkg, "__path__"):
    inner_pkg = os.path.join(PKG_ROOT, "graspnet_grasping")
    if inner_pkg not in list(pkg.__path__):
        pkg.__path__.append(inner_pkg)

pymoveit2_stub = types.ModuleType("pymoveit2")
pymoveit2_stub.MoveIt2 = object
sys.modules.setdefault("pymoveit2", pymoveit2_stub)

from graspnet_grasping.graspnet_inference_node import (  # noqa: E402
    GraspnetInferenceNode,
    _graspgroup_to_pose_metadata,
)
from graspnet_grasping.graspnet_visual_grasping_node import (  # noqa: E402
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


class GraspnetVisualGraspingNodeTest(unittest.TestCase):
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
            min_grasp_z=0.02,
            min_grasp_width_m=0.005,
            max_grasp_width_m=0.061,
            max_approach_tilt_deg=35.0,
            max_jaw_z_abs=0.35,
        )
        node._reject_candidate = lambda cand, reason: setattr(cand, "reject_reason", reason)

        safe = GraspCandidate(idx=0, camera_pose=pose(), score=1.0, width_m=0.04, grasp=pose(z=0.03, quat=safe_quat))
        side = GraspCandidate(idx=1, camera_pose=pose(), score=1.0, width_m=0.04, grasp=pose(z=0.03))

        self.assertTrue(GraspnetVisualGraspingNode.validate_candidate(node, safe))
        self.assertFalse(GraspnetVisualGraspingNode.validate_candidate(node, side))
        self.assertTrue(side.reject_reason.startswith("approach_tilt"))

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
