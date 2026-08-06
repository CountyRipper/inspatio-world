#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


EXPECTED = [
    ("S0", "P", 0), ("S0", "P", 1),
    ("S0", "N", 0), ("S0", "N", 1),
    ("S1", "P", 0), ("S1", "P", 1),
    ("S1", "N", 0), ("S1", "N", 1),
]
EXPECTED_CHECKPOINT = "ec60de7789df514c2bc85c7e11fa76af575396ea6d0ccb136520a32384630441"
EXPECTED_STEPS = [1000, 750, 500, 250]
EXPECTED_ACTUAL = [1000.0, 937.5, 833.3333129882812, 625.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts/phase1_lsm")
    args = parser.parse_args()
    root = Path(args.artifacts_root)
    results = []
    for source, trajectory, seed in EXPECTED:
        sample_id = f"{source}_{trajectory}_seed{seed}"
        sample_dir = root / "samples" / sample_id
        manifest = json.loads((sample_dir / "manifest.json").read_text())
        assert (manifest["source"], manifest["trajectory"], manifest["seed"]) == (
            source, trajectory, seed
        )
        assert manifest["checkpoint_sha256"] == EXPECTED_CHECKPOINT
        assert manifest["denoising_step_indices"] == EXPECTED_STEPS
        assert manifest["actual_model_timesteps"] == EXPECTED_ACTUAL
        assert manifest["adapter_parameter_count"] == 122_880
        assert manifest["adapter_nonzero_at_capture"] == 0
        assert manifest["identity_reprojection"]["max_abs_error"] == 0.0
        assert manifest["identity_reprojection"]["mean_abs_error"] == 0.0
        assert manifest["pose_validation"]["max_camera_center_drift"] <= 1e-6
        assert manifest["pose_validation"]["max_rotation_speed_degree_per_frame"] <= 0.8
        with safe_open(
            str(sample_dir / "sample.safetensors"), framework="pt", device="cpu"
        ) as handle:
            assert handle.get_slice("z_A").get_shape() == [1, 3, 16, 60, 104]
            assert handle.get_slice("z_B").get_shape() == [1, 3, 16, 60, 104]
            assert handle.get_slice("block18_previous").get_shape() == [1, 3, 16, 60, 104]
            assert handle.get_slice("denoise_step_inputs").get_shape() == [4, 1, 3, 16, 60, 104]
            assert handle.get_slice("transition_noises").get_shape() == [3, 1, 3, 16, 60, 104]
            assert handle.get_slice("raw_depth").get_shape() == [240, 480, 832]
            assert handle.get_slice("planned_c2w").get_shape() == [240, 4, 4]
        results.append({
            "sample_id": sample_id,
            "valid_fraction": manifest["identity_reprojection"]["valid_fraction"],
            "capture_seconds": manifest["capture_seconds"],
            "peak_vram_gib": manifest["peak_vram_gib"],
        })
    audit = {
        "passed": True,
        "num_samples": len(results),
        "sample_order": [item["sample_id"] for item in results],
        "samples": results,
    }
    output = root / "eight_sample_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
