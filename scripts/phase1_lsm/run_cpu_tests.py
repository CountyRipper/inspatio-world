#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tests.phase1_lsm.test_adapter import (
    test_adapter_shape_count_zero_init_and_roundtrip,
)
from tests.phase1_lsm.test_latent_projection import (
    test_invalid_depth_maps_to_minus_one_mask,
    test_three_slot_identity_reprojection_is_exact,
)
from tests.phase1_lsm.test_loss import (
    test_exact_memory_loss_has_gradient_only_through_prediction,
)
from tests.phase1_lsm.test_trajectory import (
    test_fixed_controls_do_not_need_resampling,
    test_fixed_trajectory_and_actual_pose_contract,
)


def main() -> None:
    with TemporaryDirectory() as adapter_dir, TemporaryDirectory() as trajectory_dir:
        test_adapter_shape_count_zero_init_and_roundtrip(Path(adapter_dir))
        test_fixed_trajectory_and_actual_pose_contract(Path(trajectory_dir))
    test_fixed_controls_do_not_need_resampling()
    test_three_slot_identity_reprojection_is_exact()
    test_invalid_depth_maps_to_minus_one_mask()
    test_exact_memory_loss_has_gradient_only_through_prediction()
    print("6/6 Phase 1 CPU tests passed")


if __name__ == "__main__":
    main()
