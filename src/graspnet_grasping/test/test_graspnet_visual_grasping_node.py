#!/usr/bin/env python3
import os
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseArray

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.abspath(os.path.join(PKG_ROOT, ".."))
for path in (PKG_ROOT, os.path.join(SRC_ROOT, "pymoveit2"), os.path.join(SRC_ROOT, "manipulation_common")):
    if path not in sys.path:
        sys.path.insert(0, path)

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
    _make_pregrasp_pose,
    _metadata_at,
)


def pose(x=0.0, y=0.0, z=0.0):
    msg = Pose()
    msg.position.x = x
    msg.position.y = y
    msg.position.z = z
    msg.orientation.w = 1.0
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

    def test_width_and_depth_drive_gripper_and_pregrasp(self):
        candidate = GraspCandidate(
            idx=0,
            camera_pose=pose(),
            score=1.0,
            width_m=0.04,
            depth_m=0.02,
            base_pose=pose(x=0.5, z=0.03),
        )
        node = SimpleNamespace(
            approach_distance=0.10,
            approach_clearance_m=0.04,
            lift_distance=0.08,
            use_pregrasp=True,
            gripper_open_positions=(0.0305, -0.0305),
            gripper_close_positions=(0.01, -0.01),
            graspnet_to_ee_rpy_deg=[0.0, 0.0, 0.0],
        )
        node._apply_orientation_correction = lambda p: _apply_orientation_correction(
            p, node.graspnet_to_ee_rpy_deg
        )
        node._prepare_grasp_pose = lambda p: GraspnetVisualGraspingNode._prepare_grasp_pose(node, p)
        node._approach_distance_for_candidate = (
            lambda c: GraspnetVisualGraspingNode._approach_distance_for_candidate(node, c)
        )

        GraspnetVisualGraspingNode.prepare_grasp_pose(node, candidate)

        self.assertAlmostEqual(candidate.approach_distance_m, 0.06)
        self.assertAlmostEqual(candidate.pregrasp.position.x, 0.44)
        self.assertEqual(candidate.close_positions, (0.02, -0.02))

    def test_pregrasp_and_lift_pose_geometry(self):
        grasp = pose(x=0.5, y=0.0, z=0.03)

        pregrasp = _make_pregrasp_pose(grasp, 0.10)
        lift = _make_lift_pose(grasp, 0.08)

        self.assertAlmostEqual(pregrasp.position.x, 0.4)
        self.assertAlmostEqual(pregrasp.position.y, 0.0)
        self.assertAlmostEqual(pregrasp.position.z, 0.03)
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
            pregrasp=pose(z=0.1),
            grasp=pose(z=0.03),
            lift=pose(z=0.11),
        )
        node = SimpleNamespace(precheck_candidate_plans=True)
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
