import numpy as np
import torch

from scripts.build_slam3r_offline_v6_2 import (
    apply_sim3,
    best_confidence_voxels,
    strict_reference_priority_fusion,
    weighted_umeyama,
)


def run_checks() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(200, 3))
    angle = np.deg2rad(23.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    target = apply_sim3(source, 1.7, rotation, np.array([0.2, -0.4, 1.1]))
    scale_fit, rotation_fit, translation_fit = weighted_umeyama(source, target)
    np.testing.assert_allclose(
        apply_sim3(source, scale_fit, rotation_fit, translation_fit), target, atol=1e-8
    )

    points = np.array(
        [[0.01, 0.01, 0.01], [0.09, 0.02, 0.01], [0.21, 0.0, 0.0]],
        dtype=np.float32,
    )
    colors = np.eye(3, dtype=np.float32)
    result = best_confidence_voxels(
        points, colors, np.array([2.0, 9.0, 3.0]), np.array([0, 1, 2]), 0.1
    )
    kept_points, kept_colors, kept_confidence, kept_source_ids = result
    assert kept_points.shape[0] == 2
    selected = np.flatnonzero(kept_source_ids == 1)
    assert selected.size == 1
    np.testing.assert_array_equal(kept_points[selected[0]], points[1])
    np.testing.assert_array_equal(kept_colors[selected[0]], colors[1])
    assert kept_confidence[selected[0]] == 9.0

    reference_rgb = torch.tensor([[[[0.8, 0.7, 0.6]]]]).expand(-1, 3, -1, -1)
    historical_rgb = torch.tensor([[[[0.2, 0.3, 0.4]]]]).expand(-1, 3, -1, -1)
    reference_mask = torch.tensor([[[[True, False, False]]]])
    historical_mask = torch.tensor([[[[True, True, False]]]])
    fused, fused_mask, historical_add = strict_reference_priority_fusion(
        reference_rgb, reference_mask, historical_rgb, historical_mask
    )
    assert torch.equal(fused_mask, torch.tensor([[[[True, True, False]]]]))
    assert torch.equal(historical_add, torch.tensor([[[[False, True, False]]]]))
    torch.testing.assert_close(fused[..., 0], reference_rgb[..., 0])
    torch.testing.assert_close(fused[..., 1], historical_rgb[..., 1])
    torch.testing.assert_close(fused[..., 2], torch.zeros_like(fused[..., 2]))


if __name__ == "__main__":
    run_checks()
    print("v6_2 core checks passed")
