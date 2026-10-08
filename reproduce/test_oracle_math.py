"""Checks for label decoding, camera gauge, and identity-sensitive scoring."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from oracle_math import (camera_in_prediction_frame, decode_instance_rgb,
                         fit_similarity, fixed_id_metrics, transform_points, unproject)


class OracleTests(unittest.TestCase):
    def test_packed_ids_preserve_same_class_objects(self):
        ids = np.array([[0, 3001, 3002, 18042, 70000]])
        rgb = np.stack([ids % 256, (ids // 256) % 256, ids // 65536], -1).astype(np.uint8)
        np.testing.assert_array_equal(decode_instance_rgb(rgb), ids)

    def test_similarity_and_camera_projection(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(300, 3))
        points[:, 2] += 8
        expected = (1.7, Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(), np.array([.2, .4, -.1]))
        target = transform_points(points, expected)
        sim = fit_similarity(points, target)
        np.testing.assert_allclose(transform_points(points, sim), target, atol=1e-10)
        camera = np.eye(4)
        camera[:3, 3] = [.3, .1, .4]
        native_camera = camera_in_prediction_frame(camera, sim)
        a = target @ camera[:3, :3].T + camera[:3, 3]
        b = points @ native_camera[:3, :3].T + native_camera[:3, 3]
        np.testing.assert_allclose(a[:, :2] / a[:, 2:], b[:, :2] / b[:, 2:], atol=1e-10)

    def test_swapping_identity_is_not_rematched(self):
        gt = np.array([[3001, 3001, 3002, 3002]])
        swapped = np.array([[3002, 3002, 3001, 3001]])
        r = fixed_id_metrics(swapped, gt, np.ones_like(gt, bool), [3001, 3002], min_pixels=1)
        self.assertEqual(r['mean_iou'], 0)
        r = fixed_id_metrics(gt, gt, np.ones_like(gt, bool), [3001, 3002], min_pixels=1)
        self.assertEqual(r['mean_iou'], 1)

    def test_depth_projection_uses_pixel_intrinsics(self):
        K = np.array([[100., 0, 2], [0, 100., 1], [0, 0, 1]])
        points = unproject(np.ones((3, 5)) * 2, K, np.eye(4))
        np.testing.assert_allclose(points[1, 2], [0, 0, 2])
        np.testing.assert_allclose(points[1, 3], [.02, 0, 2])


if __name__ == '__main__':
    unittest.main()
