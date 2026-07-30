import unittest

import torch

from utils.slam3r_incremental import (
    crop_intrinsic,
    prepare_reference_geometry,
    prepare_slam3r_frame,
)


class Slam3RIncrementalGeometryTest(unittest.TestCase):
    def test_wide_frame_resize_and_center_crop(self):
        rgb = torch.zeros((3, 480, 832))
        prepared = prepare_slam3r_frame(rgb, device="cpu")
        self.assertEqual(tuple(prepared.rgb_crop.shape), (3, 224, 224))
        self.assertEqual(tuple(prepared.model_image.shape), (1, 3, 224, 224))
        self.assertEqual(prepared.crop.resized_height, 224)
        self.assertEqual(prepared.crop.resized_width, 388)
        self.assertEqual(prepared.crop.crop_top, 0)
        self.assertEqual(prepared.crop.crop_left, 82)
        self.assertTrue((prepared.model_image == -1).all())

    def test_intrinsic_uses_real_rounded_resize_factors(self):
        rgb = torch.zeros((3, 480, 832))
        crop = prepare_slam3r_frame(rgb, device="cpu").crop
        intrinsic = torch.tensor(
            [[722.0, 0.0, 416.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]]
        )
        cropped = crop_intrinsic(intrinsic, crop)
        self.assertAlmostEqual(
            cropped[0, 0].item(), 722.0 * 388.0 / 832.0, places=4
        )
        self.assertAlmostEqual(
            cropped[1, 1].item(), 780.0 * 224.0 / 480.0, places=4
        )
        self.assertAlmostEqual(cropped[0, 2].item(), 112.0)
        self.assertAlmostEqual(cropped[1, 2].item(), 112.0)

    def test_reference_depth_mask_and_pose_match_crop(self):
        rgb = torch.zeros((3, 4, 8))
        crop = prepare_slam3r_frame(rgb, device="cpu", size=4).crop
        depth = torch.full((4, 8), 2.0)
        mask = torch.ones((4, 8), dtype=torch.bool)
        intrinsic = torch.tensor(
            [[4.0, 0.0, 4.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
        )
        pose = torch.eye(4)
        pose[0, 3] = 1.0
        geometry = prepare_reference_geometry(
            depth, mask, intrinsic, pose, crop, device="cpu"
        )
        self.assertEqual(tuple(geometry.points.shape), (4, 4, 3))
        self.assertTrue(geometry.valid.all())
        self.assertTrue(geometry.mask.all())
        # Original crop columns are 2..5. Its first pixel is x=-1 at z=2,
        # then the camera-to-world translation moves it to x=0.
        torch.testing.assert_close(
            geometry.points[0, 0], torch.tensor([0.0, -1.0, 2.0])
        )


if __name__ == "__main__":
    unittest.main()
