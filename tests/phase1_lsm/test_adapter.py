import json

import torch

from phase1_lsm.adapter import (
    ADAPTER_PARAMETER_COUNT,
    MemoryPatchAdapter,
    load_adapter,
    save_adapter,
)


def test_adapter_shape_count_zero_init_and_roundtrip(tmp_path):
    adapter = MemoryPatchAdapter()
    assert adapter.parameter_count == ADAPTER_PARAMETER_COUNT == 122_880
    condition = torch.randn(1, 20, 3, 60, 104)
    output = adapter(condition)
    assert output.shape == (1, 1536, 3, 30, 52)
    assert torch.equal(output, torch.zeros_like(output))

    with torch.no_grad():
        adapter.proj.weight.fill_(0.125)
    save_adapter(adapter, tmp_path)
    restored = MemoryPatchAdapter()
    load_adapter(restored, tmp_path / "memory_adapter.safetensors")
    assert torch.equal(adapter.proj.weight, restored.proj.weight)
    config = json.loads((tmp_path / "memory_adapter_config.json").read_text())
    assert config["parameter_count"] == ADAPTER_PARAMETER_COUNT
    assert config["input_order"] == ["memory_mask4", "projected_memory_latent16"]
