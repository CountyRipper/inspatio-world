#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-outputs", required=True)
    parser.add_argument("--projected-outputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    direct = load_file(args.direct_outputs, device="cpu")
    projected = load_file(args.projected_outputs, device="cpu")
    for key in ("A", "no_memory_Aprime"):
        if not torch.equal(direct[key], projected[key]):
            raise AssertionError(f"smoke stages disagree on immutable tensor: {key}")

    tensors = {
        "A": direct["A"].clone().contiguous(),
        "no_memory_Aprime": direct["no_memory_Aprime"].clone().contiguous(),
        "direct_memory_Aprime": direct["direct_memory_Aprime"].clone().contiguous(),
        "projected_memory_Aprime": projected[
            "projected_memory_Aprime"
        ].clone().contiguous(),
        "wrong_memory_Aprime": direct["wrong_memory_Aprime"].clone().contiguous(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output)
    manifest = {
        "purpose": "final five-column comparison",
        "A_no_memory_direct_wrong_source": str(Path(args.direct_outputs).resolve()),
        "projected_source": str(Path(args.projected_outputs).resolve()),
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
