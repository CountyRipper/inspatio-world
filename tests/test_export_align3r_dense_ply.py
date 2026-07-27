import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_align3r_dense_ply import VERTEX_DTYPE, export_dense_ply  # noqa: E402


class Align3RDensePlyTest(unittest.TestCase):
    def test_exports_every_valid_pixel_in_normalized_world_gauge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "align3r"
            source.mkdir()
            np.savetxt(
                source / "pred_traj.txt",
                np.array([
                    [0, 10, 0, 0, 1, 0, 0, 0],
                    [1, 11, 0, 0, 1, 0, 0, 0],
                ]),
            )
            np.savetxt(
                source / "pred_intrinsics.txt",
                np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), (2, 1)),
            )
            depths = [
                np.array([[2, 2], [2, -1]], dtype=np.float32),
                np.full((2, 2), 3, dtype=np.float32),
            ]
            for index, depth in enumerate(depths):
                np.save(source / f"frame_{index:04d}.npy", depth)
                rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3) + index
                Image.fromarray(rgb, mode="RGB").save(
                    source / f"frame_{index:04d}_rgb.png"
                )

            output = root / "dense.ply"
            manifest = export_dense_ply(source, output, expected_frames=2)

            self.assertEqual(manifest["point_count"], 7)
            self.assertEqual(manifest["invalid_or_nonpositive_depth_pixels"], 1)
            self.assertEqual(manifest["source_camera_pose"]["translation_max_norm"], 1)
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(), manifest["sha256"]
            )
            raw = output.read_bytes()
            end = raw.index(b"end_header\n") + len(b"end_header\n")
            vertices = np.frombuffer(raw[end:], dtype=VERTEX_DTYPE)
            np.testing.assert_allclose(
                np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
                np.array([
                    [0, 0, 2], [2, 0, 2], [0, 2, 2],
                    [1, 0, 3], [4, 0, 3], [1, 3, 3], [4, 3, 3],
                ]),
            )
            saved_manifest = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(saved_manifest["sha256"], manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
