import math
import unittest

import torch

from utils.overlap_da3_registration import (
    apply_similarity,
    backproject_world_grid,
    estimate_similarity_registration,
    pose_residual,
    select_v4_runtime_points,
    transform_da3_c2w,
)


class OverlapDA3RegistrationTest(unittest.TestCase):
    def test_same_pixel_registration_recovers_sim3(self):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 80),
            torch.linspace(-1.5, 1.5, 100),
            indexing="ij",
        )
        source = torch.stack((xx, yy, 2.0 + 0.2 * xx + 0.1 * yy), dim=-1)
        angle = math.radians(17.0)
        rotation = torch.tensor([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ])
        scale = 2.25
        translation = torch.tensor([0.4, -0.2, 1.1])
        target = apply_similarity(source, scale, rotation, translation)
        target[::13, ::11] += 4.0
        valid = torch.ones(source.shape[:-1], dtype=torch.bool)
        result = estimate_similarity_registration(
            source, target, valid, min_correspondences=1000
        )
        self.assertAlmostEqual(result.scale, scale, places=4)
        torch.testing.assert_close(result.rotation, rotation, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(
            result.translation, translation, atol=5e-4, rtol=5e-4
        )
        self.assertLess(result.normalized_rmse, 1e-3)

    def test_backprojection_and_camera_mapping(self):
        depth = torch.full((4, 5), 2.0)
        K = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
        local_w2c = torch.eye(4)[:3]
        points, valid = backproject_world_grid(depth, K, w2c=local_w2c)
        self.assertTrue(valid.all())
        torch.testing.assert_close(points[0, 2], torch.tensor([0.0, -1.5, 2.0]))

        rotation = torch.eye(3)
        translation = torch.tensor([1.0, 2.0, 3.0])
        observed = transform_da3_c2w(local_w2c, 2.0, rotation, translation)
        torch.testing.assert_close(observed[:3, 3], translation)
        residual = pose_residual(torch.eye(4), observed)
        self.assertAlmostEqual(residual["rotation_degrees"], 0.0, places=5)
        self.assertAlmostEqual(
            residual["translation"], torch.linalg.norm(translation).item(), places=5
        )

    def test_too_few_correspondences_fails_closed(self):
        source = torch.zeros((4, 4, 3))
        target = source.clone()
        valid = torch.ones((4, 4), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "valid correspondences"):
            estimate_similarity_registration(source, target, valid)

    def test_uniform_sampling_and_exact_trim_count(self):
        source = torch.randn(1000, 3)
        target = 1.5 * source + torch.tensor([0.2, -0.3, 0.4])
        result = estimate_similarity_registration(
            source,
            target,
            torch.ones(1000, dtype=torch.bool),
            min_correspondences=100,
            max_correspondences=100,
            iterations=2,
        )
        self.assertEqual(result.correspondence_count, 1000)
        self.assertEqual(result.sampled_count, 100)
        self.assertEqual(result.inlier_count, 80)
        self.assertAlmostEqual(result.inlier_ratio, 0.8)

    def test_v4_reference_aware_point_filter(self):
        reference = torch.zeros((1, 2, 2, 3))
        da3 = reference.clone()
        da3[0, 0, 1, 0] = 0.5
        valid = torch.ones((1, 2, 2), dtype=torch.bool)
        depth = torch.ones((1, 2, 2))
        depth_valid = torch.ones_like(valid)
        depth_valid[0, 1, 1] = False
        reference_mask = torch.tensor([[[True, True], [False, True]]])
        keep, stats = select_v4_runtime_points(
            da3,
            reference,
            valid,
            depth,
            depth_valid,
            reference_mask,
            voxel_size=0.01,
        )
        expected = torch.tensor([[[True, False], [True, False]]])
        self.assertTrue(torch.equal(keep, expected))
        self.assertEqual(stats["reference_uncovered_points"], 1)
        self.assertEqual(stats["reference_covered_points"], 3)
        self.assertEqual(stats["reference_geometry_consistent_points"], 1)
        self.assertEqual(stats["reference_geometry_rejected_points"], 2)


if __name__ == "__main__":
    unittest.main()
