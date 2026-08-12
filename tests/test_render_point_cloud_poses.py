import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.render_point_cloud import generate_target_c2ws


class RenderPointCloudPoseTest(unittest.TestCase):
    def test_dense_per_frame_schedule_is_not_resmoothed(self):
        yaw_values = [0.0, 12.5, 40.0, 12.5, 40.0]
        identity = torch.eye(4)
        with tempfile.TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "dense.txt"
            trajectory_path.write_text(
                "0 0 0 0 0\n"
                + " ".join(str(value) for value in yaw_values)
                + "\n0 0 0 0 0\n",
                encoding="utf-8",
            )
            targets = generate_target_c2ws(
                trajectory_path,
                identity,
                [identity] * len(yaw_values),
                num_frames=len(yaw_values),
                device=torch.device("cpu"),
                rotation_only=True,
            )
        recovered = [
            np.rad2deg(np.arctan2(float(target[0, 2]), float(target[0, 0])))
            for target in targets
        ]
        self.assertTrue(np.allclose(recovered, yaw_values, atol=1e-5))

    def test_rotation_only_yaw_stays_at_first_camera_center(self):
        angle = np.deg2rad(12.0)
        initial_c2w = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, np.cos(angle), -np.sin(angle), -2.0],
                [0.0, np.sin(angle), np.cos(angle), 0.25],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        moving_source_c2w = initial_c2w.clone()
        moving_source_c2w[:3, 3] += torch.tensor([4.0, 5.0, 6.0])

        with tempfile.TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "pure_yaw.txt"
            trajectory_path.write_text(
                "0 0\n0 40\n1 1\n",
                encoding="utf-8",
            )
            targets = generate_target_c2ws(
                trajectory_path,
                initial_c2w,
                [initial_c2w, moving_source_c2w],
                num_frames=2,
                device=torch.device("cpu"),
                relative_to_source=False,
                rotation_only=True,
            )

        relative = torch.linalg.inv(initial_c2w) @ targets[1]
        yaw = np.deg2rad(40.0)
        expected_rotation = torch.tensor(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(targets[0], initial_c2w, atol=1e-6))
        self.assertTrue(
            torch.allclose(targets[1][:3, 3], initial_c2w[:3, 3], atol=1e-6)
        )
        self.assertTrue(torch.allclose(relative[:3, 3], torch.zeros(3), atol=1e-6))
        self.assertTrue(
            torch.allclose(relative[:3, :3], expected_rotation, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
