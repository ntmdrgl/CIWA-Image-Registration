"""Geometry checks independent of the ETH3D downloads and GUI viewers."""

import unittest

import cv2
import numpy as np

from test_SfM import Reconstruction, project, triangulate


class SfMTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.K = np.array([[700., 0, 370], [0, 700, 230], [0, 0, 1]])
        self.points = rng.uniform([-1, -1, 4], [1, 1, 8], (60, 3))
        self.poses = [(np.eye(3), np.zeros((3, 1))),
                      (np.eye(3), np.array([[-1.], [0], [0]]))]
        self.uv = [project(self.points, pose, self.K)[0] for pose in self.poses]

    def model(self):
        features = [(uv.copy(), None) for uv in self.uv]
        colors = [np.full((len(uv), 3), 128, dtype=np.uint8) for uv in self.uv]
        model = Reconstruction(self.K, features, colors)
        model.seed = (0, 1)
        model.poses = dict(enumerate(self.poses))
        for key, point in enumerate(self.points):
            model.add_point(point.copy(), 0, key, 1, key)
        return model

    def test_triangulation_and_degenerate_baseline(self):
        xyz, valid = triangulate(*self.uv, *self.poses, self.K)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(xyz, self.points, atol=1e-8)
        close_pose = (np.eye(3), np.array([[-1e-5], [0], [0]]))
        close_uv = project(self.points, close_pose, self.K)[0]
        _, valid = triangulate(self.uv[0], close_uv, self.poses[0], close_pose, self.K)
        self.assertFalse(valid.any())

    def test_triangulation_rejects_negative_depth(self):
        behind = self.points.copy()
        behind[:, 2] *= -1
        uv = [project(behind, pose, self.K)[0] for pose in self.poses]
        _, valid = triangulate(*uv, *self.poses, self.K)
        self.assertFalse(valid.any())

    def test_pnp_registers_third_camera_with_outliers(self):
        model = self.model()
        pose = (cv2.Rodrigues(np.array([0.02, -0.03, 0.01]))[0],
                np.array([[-1.5], [0.1], [0.05]]))
        uv = project(self.points, pose, self.K)[0]
        uv[:10] += [80, -60]
        model.features.append((uv, None))
        model.point_ids.append(np.full(len(uv), -1, dtype=int))
        model.colors.append(np.full((len(uv), 3), 128, dtype=np.uint8))
        pairs = np.column_stack((np.arange(len(uv)), np.arange(len(uv))))
        model.match = lambda i, j: pairs
        cv2.setRNGSeed(0)
        self.assertTrue(model.register(2))
        np.testing.assert_allclose(model.poses[2][0], pose[0], atol=1e-6)
        np.testing.assert_allclose(model.poses[2][1], pose[1], atol=1e-6)
        self.assertTrue((model.point_ids[2][:10] == -1).all())
        self.assertTrue((model.point_ids[2][10:] >= 0).all())

    def test_bundle_adjustment_reduces_error_and_fixes_gauge(self):
        model = self.model()
        model.points = list(self.points + np.random.default_rng(2).normal(0, 0.02, self.points.shape))
        before = np.linalg.norm(project(model.points, model.poses[0], self.K)[0] - self.uv[0])
        model.bundle_adjust(max_nfev=80)
        after = np.linalg.norm(project(model.points, model.poses[0], self.K)[0] - self.uv[0])
        self.assertLess(after, before * 0.01)
        np.testing.assert_array_equal(model.poses[0][0], np.eye(3))
        np.testing.assert_array_equal(model.poses[0][1], np.zeros((3, 1)))
        self.assertEqual(model.poses[1][1][0, 0], -1.)

    def test_pruning_clears_both_track_and_keypoint_links(self):
        model = self.model()
        model.features[1][0][0] += 100
        model.prune()
        self.assertEqual(model.tracks[0], {})
        self.assertEqual(model.point_ids[0][0], -1)
        self.assertEqual(model.point_ids[1][0], -1)
        self.assertEqual(len(model.tracks[1]), 2)


if __name__ == "__main__":
    unittest.main()
