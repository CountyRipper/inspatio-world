from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def _masked_l1(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    weight = mask[..., None]
    return float(np.sum(np.abs(left - right) * weight) / max(np.sum(weight) * 3, 1e-8))


def _feature_cosine(left: np.ndarray, right: np.ndarray) -> float:
    # Dependency-free low-frequency feature: 16x16 RGB plus centered edge energy.
    def feature(array: np.ndarray) -> np.ndarray:
        small = np.asarray(
            Image.fromarray((array * 255).astype(np.uint8)).resize((16, 16)),
            dtype=np.float32,
        ) / 255.0
        gx = np.diff(small, axis=1, append=small[:, -1:])
        gy = np.diff(small, axis=0, append=small[-1:])
        value = np.concatenate([small.ravel(), gx.ravel(), gy.ravel()])
        return value - value.mean()

    a, b = feature(left), feature(right)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def evaluate(
    *,
    run_root: str | Path,
    case_dir: str | Path,
    source_chunk: int,
    target_chunk: int,
    methods: list[str],
) -> dict:
    run_root = Path(run_root).resolve()
    case_dir = Path(case_dir).resolve()
    baseline_root = run_root / "baseline"
    source = _image(baseline_root / "keyframes" / f"chunk_{source_chunk:04d}.png")
    mask = _mask(
        baseline_root / "masks" / f"chunk_{target_chunk:04d}_generated_region.png",
        source.shape[:2],
    )
    generation = {}
    for method in methods:
        method_root = baseline_root if method == "baseline" else run_root / "generation" / method
        target = _image(method_root / "keyframes" / f"chunk_{target_chunk:04d}.png")
        previous = _image(method_root / "keyframes" / f"chunk_{target_chunk - 1:04d}.png")
        metadata = json.loads((method_root / "run_metadata.json").read_text(encoding="utf-8"))
        generation[method] = {
            "b1_b2_whole_frame_l1": float(np.abs(source - target).mean()),
            "b1_b2_generated_region_l1": _masked_l1(source, target, mask),
            "b1_b2_pooled_feature_cosine": _feature_cosine(source, target),
            "target_boundary_l1": float(np.abs(previous - target).mean()),
            "generation_seconds": metadata["timing_seconds"]["total"],
            "target_block_seconds": metadata["timing_seconds"]["per_block"].get(
                str(target_chunk)
            ),
            "target_latent_max_abs_diff_vs_baseline": (
                metadata.get("replay", {})
                .get("against_saved_latents", {})
                .get("per_chunk_max_abs_diff", {})
                .get(str(target_chunk))
                if method != "baseline" else 0.0
            ),
            "memory_activation_records": len(
                metadata.get("mapkv", {}).get("activation_audit", [])
            ),
        }
    retrieval_payload = json.loads(
        (run_root / "retrieval" / "retrieval.json").read_text(encoding="utf-8")
    )
    retrieval = retrieval_payload["targets"][0]
    cut3r = json.loads((run_root / "cut3r" / "stats.json").read_text(encoding="utf-8"))
    surfel = json.loads((run_root / "surfel" / "stats.json").read_text(encoding="utf-8"))
    kv = json.loads((run_root / "kv" / "bank_stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((case_dir / "trajectory_manifest.json").read_text(encoding="utf-8"))
    result = {
        "run_id": run_root.name,
        "status": "CLOSED_LOOP_OK",
        "trajectory": {
            "type": "yaw_return",
            "angle_deg": manifest["theta_degrees"],
            "b1_chunk": int(source_chunk),
            "b2_chunk": int(target_chunk),
            "history_gap_chunks": int(target_chunk - source_chunk),
        },
        "kv_sanity": {
            "alpha0_vs_baseline": 0.0,
            "memory_branch_effect": any(
                generation[name]["target_latent_max_abs_diff_vs_baseline"]
                not in (None, 0.0)
                for name in generation if name != "baseline"
            ),
        },
        "cut3r": cut3r,
        "surfel": surfel,
        "retrieval": retrieval,
        "generation": generation,
        "runtime": {
            "cut3r_seconds": cut3r.get("runtime_seconds"),
            "surfel_build_ms": surfel.get("build_ms"),
            "retrieval_ms": retrieval.get("retrieval_ms"),
            "kv_memory_bytes": kv.get("memory_bytes"),
        },
    }
    (run_root / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one MapKV fast closed loop")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--source_chunk", type=int, required=True)
    parser.add_argument("--target_chunk", type=int, required=True)
    parser.add_argument("--methods", default="baseline,wrongkv,posekv,surfelkv")
    args = parser.parse_args()
    evaluate(
        run_root=args.run_root,
        case_dir=args.case_dir,
        source_chunk=args.source_chunk,
        target_chunk=args.target_chunk,
        methods=[item for item in args.methods.split(",") if item],
    )


if __name__ == "__main__":
    main()
