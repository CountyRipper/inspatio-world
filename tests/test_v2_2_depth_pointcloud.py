import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.v2_2_depth_pointcloud import (
    PLY_DTYPE,
    backproject_world,
    build_align3r_full_frames,
    global_depth_scale,
    selected_frame_indices,
    write_binary_ply,
)


class V22DepthPointCloudTests(unittest.TestCase):
    def test_full_frame_schedule_uses_every_generated_frame(self):
        self.assertEqual(selected_frame_indices(237, 1), list(range(237)))

    def test_frame_schedule_must_include_final_frame(self):
        with self.assertRaises(ValueError):
            selected_frame_indices(237, 5)

    def test_global_depth_scale_recovers_one_sequence_scale(self):
        predicted = np.ones((2, 48, 48), dtype=np.float32) * 2.0
        reference = predicted * 3.5
        mask = np.ones_like(predicted, dtype=bool)
        scale, stats = global_depth_scale(predicted, reference, mask)
        self.assertAlmostEqual(scale, 3.5, places=5)
        self.assertEqual(stats["overlap_pixels"], predicted.size)
        self.assertAlmostEqual(stats["log_mad"], 0.0, places=6)

    def test_backprojection_uses_known_c2w(self):
        depth = np.ones((2, 2), dtype=np.float32)
        intrinsic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [10, 20, 30]
        points = backproject_world(depth, intrinsic, c2w)
        np.testing.assert_allclose(points[0, 0], [10, 20, 31])
        np.testing.assert_allclose(points[1, 1], [11, 21, 31])

    def test_ply_exports_identical_pixel_and_color_stream(self):
        try:
            import cv2
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.array(
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                dtype=np.uint8,
            )
            frame = root / "frame_0000.png"
            cv2.imwrite(str(frame), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            depths_a = np.ones((1, 2, 2), dtype=np.float32)
            depths_b = depths_a * 2
            intrinsic = np.eye(3, dtype=np.float32)
            c2w = np.eye(4, dtype=np.float32)[None]
            mask = np.array([[[True, False], [True, True]]])
            count_a, colors_a = write_binary_ply(
                root / "a.ply", depths_a, [frame], intrinsic, c2w, mask
            )
            count_b, colors_b = write_binary_ply(
                root / "b.ply", depths_b, [frame], intrinsic, c2w, mask
            )
            self.assertEqual(count_a, 3)
            self.assertEqual(count_a, count_b)
            self.assertEqual(colors_a, colors_b)
            header_end = (root / "a.ply").read_bytes().index(b"end_header\n") + len(b"end_header\n")
            records = np.fromfile(root / "a.ply", dtype=PLY_DTYPE, offset=header_end)
            self.assertEqual(len(records), 3)
            np.testing.assert_array_equal(records["red"], [1, 7, 10])

    def test_align3r_full_frame_build_exports_every_frame(self):
        try:
            import cv2
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            frames = prepared / "frames"
            depths = root / "depths"
            output = root / "output"
            frames.mkdir(parents=True)
            depths.mkdir()
            height = width = 48
            for index in range(2):
                rgb = np.full((height, width, 3), 40 + index, dtype=np.uint8)
                cv2.imwrite(str(frames / f"frame_{index:04d}.png"), rgb)
                np.save(
                    depths / f"frame_{index:04d}.npy",
                    np.full((height, width), 2.0, dtype=np.float32),
                )
            np.save(
                prepared / "target_c2w_keyframes.npy",
                np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
            )
            np.save(prepared / "intrinsic.npy", np.eye(3, dtype=np.float32))
            np.save(
                prepared / "reference_depth_keyframes.npy",
                np.full((2, height, width), 6.0, dtype=np.float32),
            )
            (prepared / "prepared_manifest.json").write_text(json.dumps({
                "generated_video": "synthetic.mp4",
                "generated_video_sha256": "video",
                "expected_frames": 2,
                "keyframe_indices": [0, 1],
                "target_c2w_keyframes_sha256": "c2w",
                "intrinsic_sha256": "intrinsic",
                "reference_depth_keyframes_sha256": "reference",
            }))
            build_align3r_full_frames(SimpleNamespace(
                prepared_dir=str(prepared),
                align3r_depth_dir=str(depths),
                output_dir=str(output),
                min_depth=0.1,
                max_log_gradient=0.05,
            ))
            manifest = json.loads(
                (output / "align3r_full_frame_manifest.json").read_text()
            )
            self.assertEqual(manifest["frame_indices"], [0, 1])
            self.assertEqual(manifest["point_count"], 2 * height * width)
            self.assertAlmostEqual(manifest["global_scale"]["scale"], 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
