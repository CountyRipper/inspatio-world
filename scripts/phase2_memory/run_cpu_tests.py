#!/usr/bin/env python3
from __future__ import annotations

from tests.phase2_memory.test_memory import (
    test_bank_top2_writeback_detach_and_projection,
    test_four_step_backpropagation_is_not_last_step_only,
    test_masked_anchor_changes_only_occupancy,
    test_multimemory_trajectory_and_manifest,
    test_source_occupancy_shape,
)


def main() -> None:
    test_bank_top2_writeback_detach_and_projection()
    test_multimemory_trajectory_and_manifest()
    test_source_occupancy_shape()
    test_masked_anchor_changes_only_occupancy()
    test_four_step_backpropagation_is_not_last_step_only()
    print("5/5 Phase 2 CPU tests passed")


if __name__ == "__main__":
    main()
