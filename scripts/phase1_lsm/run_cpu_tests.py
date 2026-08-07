#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tests.phase1_lsm.test_adapter import test_adapter_shape_count_zero_init_and_roundtrip
from tests.phase1_lsm.test_latent_projection import (
    test_invalid_depth_maps_to_minus_one_mask,
    test_three_slot_identity_reprojection_is_exact,
)
from tests.phase1_lsm.test_mask_shape import test_lossless_mask_video_shape_for_vae
from tests.phase1_lsm.test_nearview import (
    test_invalid_raw_l1_empty_exact_region_is_zero,
    test_invalid_raw_l1_uses_exact_strict_invalid_region,
    test_nearview_trajectory_and_pose_audit,
    test_patch_gate_and_adapter_output_hard_gate,
    test_patch_gate_rejects_nonbinary_occupancy,
    test_preservation_mask_is_strict_occupancy_complement,
    test_projection_displacement_allows_no_overlap_diagnostic,
    test_shared_a_fork_preconditions,
    test_wide_coverage_selection_is_fixed_before_training,
    test_zero_occupancy_adapter_addition_is_exact_zero,
)
from tests.phase1_lsm.test_loss import test_exact_memory_loss_has_gradient_only_through_prediction
from tests.phase1_lsm.test_trajectory import (
    test_fixed_controls_do_not_need_resampling,
    test_fixed_trajectory_and_actual_pose_contract,
)


def main() -> None:
    with (
        TemporaryDirectory() as adapter_dir,
        TemporaryDirectory() as trajectory_dir,
        TemporaryDirectory() as nearview_dir,
    ):
        test_adapter_shape_count_zero_init_and_roundtrip(Path(adapter_dir))
        test_fixed_trajectory_and_actual_pose_contract(Path(trajectory_dir))
        test_nearview_trajectory_and_pose_audit(Path(nearview_dir))
    test_fixed_controls_do_not_need_resampling()
    test_three_slot_identity_reprojection_is_exact()
    test_invalid_depth_maps_to_minus_one_mask()
    test_exact_memory_loss_has_gradient_only_through_prediction()
    test_preservation_mask_is_strict_occupancy_complement()
    test_projection_displacement_allows_no_overlap_diagnostic()
    test_invalid_raw_l1_uses_exact_strict_invalid_region()
    test_invalid_raw_l1_empty_exact_region_is_zero()
    test_patch_gate_and_adapter_output_hard_gate()
    test_zero_occupancy_adapter_addition_is_exact_zero()
    test_patch_gate_rejects_nonbinary_occupancy()
    test_shared_a_fork_preconditions()
    test_wide_coverage_selection_is_fixed_before_training()
    test_lossless_mask_video_shape_for_vae()
    print("17/17 Phase 1 CPU tests passed")


if __name__ == "__main__":
    main()
