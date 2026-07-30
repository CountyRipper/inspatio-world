import unittest

import torch

from utils.mapanything_estimator import resize_for_mapanything, transform_points_c2w


class MapAnythingEstimatorGeometryTest(unittest.TestCase):
    def test_resize_scales_intrinsics(self):
        rgb = torch.zeros((2, 3, 480, 832))
        intrinsic = torch.tensor(
            [[722.0, 0.0, 416.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]]
        )
        resized, scaled = resize_for_mapanything(rgb, intrinsic)
        self.assertEqual(tuple(resized.shape), (2, 3, 294, 518))
        self.assertEqual(tuple(scaled.shape), (2, 3, 3))
        self.assertAlmostEqual(float(scaled[0, 0, 2]), 259.0, places=4)
        self.assertAlmostEqual(float(scaled[0, 1, 2]), 147.0, places=4)

    def test_non_identity_anchor_maps_points_to_canonical_world(self):
        points = torch.tensor([[[1.0, 0.0, 2.0]]])
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        c2w[:3, 3] = torch.tensor([3.0, 4.0, 5.0])
        transformed = transform_points_c2w(points, c2w)
        self.assertTrue(torch.allclose(transformed, torch.tensor([[[5.0, 4.0, 4.0]]])))


if __name__ == "__main__":
    unittest.main()
