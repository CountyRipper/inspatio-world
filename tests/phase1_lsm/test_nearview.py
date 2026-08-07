from pathlib import Path

import numpy as np
import torch

from phase1_lsm.adapter import (
    MemoryPatchAdapter,
    gated_adapter_residual,
    patch_occupancy_gate,
)
from phase1_lsm.nearview import (
    projection_displacement_statistics,
    choose_wide_offset,
    invalid_raw_l1,
    nearview_controls,
    preservation_invalid_mask,
    validate_nearview_c2w,
    write_nearview_trajectory,
)


def _c2w_from_control_yaw(yaw_degrees: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(yaw_degrees)
    result = np.repeat(np.eye(4, dtype=np.float32)[None], len(angle), axis=0)
    result[:, 0, 0] = np.cos(angle)
    result[:, 0, 2] = np.sin(angle)
    result[:, 2, 0] = -np.sin(angle)
    result[:, 2, 2] = np.cos(angle)
    result[:, :3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return result


def test_nearview_trajectory_and_pose_audit(tmp_path: Path) -> None:
    for offset in (0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0):
        path = write_nearview_trajectory(tmp_path / f"{offset}.txt", offset)
        controls = np.loadtxt(path)
        assert controls.shape == (3, 240)
        _, yaw, _ = nearview_controls(offset)
        np.testing.assert_allclose(controls[1], yaw, atol=1e-9, rtol=0)
        audit = validate_nearview_c2w(_c2w_from_control_yaw(yaw), offset)
        assert audit["max_signed_yaw_error_degrees"] <= 0.1
        assert audit["max_camera_center_drift"] == 0.0


def test_projection_displacement_allows_no_overlap_diagnostic() -> None:
    depth = torch.ones(3, 4, 4)
    K = torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    source = torch.eye(4).repeat(3, 1, 1)
    target = source.clone()
    target[:, 0, 0] = -1.0
    target[:, 2, 2] = -1.0
    stats = projection_displacement_statistics(depth, K, source, target, (2, 2))
    assert stats["no_overlap"] is True
    assert stats["mean_pixel_displacement"] == 0.0
    assert all(item["valid_source_points"] == 0 for item in stats["per_slot"])


def test_preservation_mask_is_strict_occupancy_complement() -> None:
    occupancy = torch.zeros(1, 3, 1, 5, 5, dtype=torch.bool)
    occupancy[:, :, :, 2, 2] = True
    preserve = preservation_invalid_mask(occupancy)
    assert torch.equal(preserve, ~occupancy)
    assert preserve[:, :, :, 1:4, 1:4].sum() == 8 * 3
    assert preserve[:, :, :, 0, 0].all()


def test_invalid_raw_l1_uses_exact_strict_invalid_region() -> None:
    occupancy = torch.tensor([[[[[True, False], [False, True]]]]])
    no_memory = torch.zeros(1, 1, 2, 2, 2)
    prediction = torch.tensor([[[[[9.0, 1.0], [3.0, 9.0]], [[9.0, 5.0], [7.0, 9.0]]]]])
    assert invalid_raw_l1(prediction, no_memory, occupancy).item() == 4.0


def test_invalid_raw_l1_empty_exact_region_is_zero() -> None:
    prediction = torch.randn(1, 3, 16, 2, 2, requires_grad=True)
    no_memory = torch.zeros_like(prediction)
    occupancy = torch.ones(1, 3, 1, 2, 2, dtype=torch.bool)
    loss = invalid_raw_l1(prediction, no_memory, occupancy)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_wide_coverage_selection_is_fixed_before_training() -> None:
    assert choose_wide_offset(1, 0.05, 0.0) == (20.0, False)
    assert choose_wide_offset(-1, 0.049, 0.20) == (-15.0, False)
    assert choose_wide_offset(1, 0.01, 0.049) == (15.0, True)


def test_patch_gate_and_adapter_output_hard_gate() -> None:
    adapter = MemoryPatchAdapter()
    with torch.no_grad():
        adapter.proj.weight.fill_(0.25)
    condition = torch.ones(1, 20, 3, 4, 6)
    occupancy = torch.zeros(1, 1, 3, 4, 6)
    occupancy[:, :, :, :2, :2] = 1
    residual, gate = gated_adapter_residual(adapter, condition, occupancy)
    assert torch.equal(gate, patch_occupancy_gate(occupancy))
    assert set(torch.unique(gate).tolist()) == {0.0, 1.0}
    invalid = (gate == 0).expand_as(residual)
    assert residual[invalid].abs().max().item() == 0.0
    assert torch.count_nonzero(residual[invalid]).item() == 0
    assert torch.count_nonzero(residual[~invalid]).item() > 0


def test_zero_occupancy_adapter_addition_is_exact_zero() -> None:
    adapter = MemoryPatchAdapter()
    with torch.no_grad():
        adapter.proj.weight.normal_()
    condition = torch.randn(1, 20, 3, 4, 6)
    occupancy = torch.zeros(1, 1, 3, 4, 6)
    residual, gate = gated_adapter_residual(adapter, condition, occupancy)
    assert torch.equal(gate, torch.zeros_like(gate))
    assert torch.equal(residual, torch.zeros_like(residual))


def test_patch_gate_rejects_nonbinary_occupancy() -> None:
    occupancy = torch.zeros(1, 1, 3, 4, 6)
    occupancy[..., 0, 0] = 0.5
    try:
        patch_occupancy_gate(occupancy)
    except AssertionError:
        pass
    else:
        raise AssertionError("nonbinary occupancy must be rejected")


def test_shared_a_fork_preconditions() -> None:
    from scripts.phase1_lsm.run_sharedA_hardgate_5deg import (
        cache_torch_equal,
        clone_cache,
    )

    controls = [
        nearview_controls(offset)
        for offset in (0.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0)
    ]
    for branch in controls[1:]:
        for shared, candidate in zip(controls[0], branch):
            assert np.array_equal(shared[:69], candidate[:69])
    cache = [{"k": torch.randn(1, 3, 2), "v": torch.randn(1, 3, 2)}]
    plus_cache = clone_cache(cache)
    minus_cache = clone_cache(cache)
    assert cache_torch_equal(plus_cache, minus_cache)
    assert cache_torch_equal(cache, plus_cache)
    plus_cache[0]["k"].add_(1)
    assert not cache_torch_equal(plus_cache, minus_cache)
    assert cache_torch_equal(cache, minus_cache)
