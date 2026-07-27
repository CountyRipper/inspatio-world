import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth.depth_only_align3r import load_align3r_reconstruction  # noqa: E402


class Align3RDepthLoaderTest(unittest.TestCase):
    def test_loads_all_frames_and_normalizes_pose_gauge(self):
        with tempfile.TemporaryDirectory() as temporary:
            reconstruction = Path(temporary)
            for index, value in enumerate((2.0, 3.0)):
                np.save(
                    reconstruction / f"frame_{index:04d}.npy",
                    np.full((2, 3), value, dtype=np.float32),
                )
                Image.new("RGB", (3, 2), (10 + index, 20, 30)).save(
                    reconstruction / f"frame_{index:04d}_rgb.png"
                )
            np.savetxt(
                reconstruction / "pred_intrinsics.txt",
                np.tile(np.array([2, 0, 1, 0, 2, 1, 0, 0, 1]), (2, 1)),
            )
            np.savetxt(
                reconstruction / "pred_traj.txt",
                np.array([
                    [0, 5, 0, 0, 1, 0, 0, 0],
                    [1, 6, 0, 0, 1, 0, 0, 0],
                ]),
            )

            rgb, depth, intrinsics, extrinsics = load_align3r_reconstruction(
                reconstruction,
                frame_count=2,
                output_size=(4, 6),
                output_device=torch.device("cpu"),
            )

            self.assertEqual(tuple(rgb.shape), (2, 3, 4, 6))
            self.assertEqual(tuple(depth.shape), (2, 4, 6))
            self.assertEqual(tuple(intrinsics.shape), (2, 3, 3))
            self.assertEqual(tuple(extrinsics.shape), (2, 3, 4))
            torch.testing.assert_close(depth[0], torch.full((4, 6), 2.0))
            torch.testing.assert_close(intrinsics[:, 0, 0], torch.tensor([4.0, 4.0]))
            torch.testing.assert_close(intrinsics[:, 1, 1], torch.tensor([4.0, 4.0]))
            torch.testing.assert_close(extrinsics[0], torch.eye(4)[:3])
            torch.testing.assert_close(extrinsics[1, :, 3], torch.tensor([-1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
