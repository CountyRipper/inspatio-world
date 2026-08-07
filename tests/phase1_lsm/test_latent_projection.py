import torch

from phase1_lsm.latent_projection import (
    identity_reprojection_error,
    project_memory_sequence,
)


def test_three_slot_identity_reprojection_is_exact():
    torch.manual_seed(7)
    latent = torch.randn(1, 3, 16, 6, 10, dtype=torch.float16)
    depth = torch.linspace(1.0, 3.0, 48 * 80).reshape(1, 48, 80).repeat(3, 1, 1)
    K = torch.tensor([[60.0, 0.0, 40.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4).repeat(3, 1, 1)
    c2w[:, :3, 3] = torch.tensor([0.25, -0.5, 1.0])
    projected, mask4, occupancy = project_memory_sequence(latent, depth, K, c2w, c2w)
    metrics = identity_reprojection_error(latent, projected, occupancy)

    assert torch.equal(projected, latent)
    assert torch.equal(mask4, torch.ones_like(mask4))
    assert metrics["valid_fraction"] == 1.0
    assert metrics["max_abs_error"] == 0.0


def test_invalid_depth_maps_to_minus_one_mask():
    latent = torch.ones(1, 3, 16, 2, 2, dtype=torch.float16)
    depth = torch.ones(3, 4, 4)
    depth[:, :2, :2] = 0
    K = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4).repeat(3, 1, 1)
    projected, mask4, occupancy = project_memory_sequence(latent, depth, K, c2w, c2w)
    metrics = identity_reprojection_error(latent, projected, occupancy)
    assert torch.equal(mask4, occupancy.expand_as(mask4).to(mask4.dtype).mul(2).sub(1))
    assert (mask4 == -1).any()
    assert metrics["valid_fraction"] < 1.0
    assert metrics["max_abs_error"] == 0.0
