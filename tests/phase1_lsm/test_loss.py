import torch

from phase1_lsm.losses import exact_memory_loss


def test_exact_memory_loss_has_gradient_only_through_prediction():
    prediction = torch.randn(1, 3, 16, 2, 2, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    valid = torch.ones(1, 3, 1, 2, 2, dtype=torch.bool)
    loss, components = exact_memory_loss(prediction, target, valid)
    loss.backward()
    assert prediction.grad is not None
    assert target.grad is None
    assert torch.isfinite(loss)
    assert set(components) == {"smooth_l1", "latent_cosine"}
