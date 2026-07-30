import json
import math
import unittest
from types import SimpleNamespace

import torch

from scripts.run_slam3r_offline_v6 import FrozenSimilarity
from utils.overlap_da3_registration import apply_similarity
from utils.slam3r_incremental import (
    CenterCropTransform,
    ReferenceGeometry,
    Slam3RFrameOutput,
)


class FrozenSimilarityTest(unittest.TestCase):
    def test_fit_freeze_and_json_metrics(self):
        args = SimpleNamespace(
            sim3_confidence_threshold=1.5,
            sim3_max_normalized_rmse=1e-4,
            sim3_max_validation_p90=1e-4,
            sim3_max_candidates=5,
            sim3_max_correspondences=60_000,
        )
        alignment = FrozenSimilarity(args, torch.device("cpu"))
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 20),
            torch.linspace(-1.0, 1.0, 20),
            indexing="ij",
        )
        source = torch.stack((xx, yy, 2.0 + 0.2 * xx * yy), dim=-1)
        angle = math.radians(12.0)
        rotation = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        target = apply_similarity(
            source, 1.7, rotation, torch.tensor([0.3, -0.2, 0.8])
        )
        crop = CenterCropTransform(20, 20, 20, 20, 0, 0, 20)
        for frame_index in range(5):
            output = Slam3RFrameOutput(
                frame_index=frame_index * 4,
                rgb_crop=torch.zeros((3, 20, 20)),
                points_world=source,
                confidence=torch.full((20, 20), 20.0),
                i2p_confidence_mean=20.0,
                l2w_confidence_mean=20.0,
                retrieved_frame_indices=(),
                buffer_frame_indices=(),
                crop=crop,
                initial_frame=True,
            )
            reference = ReferenceGeometry(
                points=target,
                depth=torch.ones((20, 20)),
                valid=torch.ones((20, 20), dtype=torch.bool),
                mask=torch.ones((20, 20), dtype=torch.bool),
                intrinsic=torch.eye(3),
            )
            alignment.append(output, reference)

        self.assertTrue(alignment.maybe_fit(5))
        self.assertTrue(alignment.frozen)
        self.assertAlmostEqual(alignment.scale, 1.7, places=4)
        self.assertTrue(alignment.attempts[-1]["accepted"])
        json.dumps(alignment.attempts, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
