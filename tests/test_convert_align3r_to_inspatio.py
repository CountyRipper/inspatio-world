import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from convert_align3r_to_inspatio import (  # noqa: E402
    convert,
    quaternion_wxyz_to_matrix,
)
from convert_da3_to_pi3 import read_da3_depth  # noqa: E402


class Align3RAdapterTest(unittest.TestCase):
    def test_quaternion_is_scalar_first(self):
        angle = np.pi / 2
        rotation = quaternion_wxyz_to_matrix(
            np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        )
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            atol=1e-6,
        )

    def test_conversion_preserves_depth_and_normalizes_first_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "align3r"
            output = root / "adapted"
            source.mkdir()
            poses = np.array([
                [0, 3, 4, 5, 1, 0, 0, 0],
                [1, 4, 4, 5, 1, 0, 0, 0],
            ], dtype=np.float64)
            np.savetxt(source / "pred_traj.txt", poses)
            intrinsics = np.array([
                [400, 0, 2, 0, 400, 1, 0, 0, 1],
                [401, 0, 2, 0, 401, 1, 0, 0, 1],
            ], dtype=np.float64)
            np.savetxt(source / "pred_intrinsics.txt", intrinsics)
            expected_depths = []
            for index in range(2):
                depth = np.arange(8, dtype=np.float32).reshape(2, 4) + index + 1
                expected_depths.append(depth)
                np.save(source / f"frame_{index:04d}.npy", depth)
                Image.new("RGB", (4, 2), (10 + index, 20, 30)).save(
                    source / f"frame_{index:04d}_rgb.png"
                )

            manifest = convert(source, output, expected_frames=2)

            self.assertEqual(manifest["frame_count"], 2)
            self.assertLess(manifest["first_c2w_identity_max_error"], 1e-6)
            for index, expected in enumerate(expected_depths):
                actual = read_da3_depth(output / "depth" / f"{index:04d}.png")
                np.testing.assert_array_equal(actual, expected)
            extrinsic = np.loadtxt(output / "extrinsic.txt").reshape(2, 3, 4)
            np.testing.assert_allclose(extrinsic[0], np.eye(4)[:3], atol=1e-6)
            np.testing.assert_allclose(extrinsic[1, :, 3], [-1, 0, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
