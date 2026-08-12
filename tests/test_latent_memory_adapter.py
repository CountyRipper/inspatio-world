import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from world_memory.latent_adapter import (
    LatentMemoryAdapter,
    add_gated_memory_residual,
    attach_latent_memory_adapter,
    gated_memory_residual,
    load_latent_memory_adapter,
    save_latent_memory_adapter,
)


class LatentMemoryAdapterTest(unittest.TestCase):
    def test_parameter_count_for_1_3b(self):
        adapter = LatentMemoryAdapter(model_dim=1536)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 122_880)

    def test_hard_gate_zeros_residual_outside_occupancy(self):
        adapter = LatentMemoryAdapter(model_dim=2)
        with torch.no_grad():
            adapter.proj.weight.fill_(1.0)

        condition = torch.ones(1, 20, 1, 4, 4)
        occupancy = torch.zeros(1, 1, 1, 4, 4)
        occupancy[..., :2, :2] = 1
        residual = gated_memory_residual(adapter, condition, occupancy)

        self.assertTrue(torch.all(residual[..., 0, 0] != 0))
        self.assertEqual(torch.count_nonzero(residual[..., 0, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(residual[..., 1, :]).item(), 0)

    def test_sidecar_save_load_is_identical(self):
        torch.manual_seed(7)
        adapter = LatentMemoryAdapter(model_dim=8)
        condition = torch.randn(1, 20, 2, 4, 4)
        expected = adapter(condition)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_latent_memory_adapter(adapter, path)
            loaded = load_latent_memory_adapter(path)
            actual = loaded(condition)

        self.assertTrue(torch.equal(expected, actual))

    def test_none_bypass_is_exact_and_memory_output_is_finite(self):
        torch.manual_seed(11)
        base = [torch.randn(1, 12, 1, 2, 2)]
        bypassed = add_gated_memory_residual(base, None)
        self.assertIs(bypassed, base)
        self.assertTrue(torch.equal(bypassed[0], base[0]))

        class ToyPatchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 12
                self.patch_embedding = nn.Conv3d(36, 12, (1, 2, 2), (1, 2, 2))

        model = ToyPatchModel()
        adapter = attach_latent_memory_adapter(model)
        condition = torch.randn(1, 20, 1, 4, 4)
        occupancy = torch.ones(1, 1, 1, 4, 4)
        memory_on = add_gated_memory_residual(base, adapter, condition, occupancy)

        self.assertTrue(torch.isfinite(memory_on[0]).all())


if __name__ == "__main__":
    unittest.main()
