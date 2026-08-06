#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from phase1_lsm.data_prep import prepare_condition
from phase1_lsm.trajectory import write_fixed_manifest, write_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("S0", "S1"), required=True)
    parser.add_argument("--trajectory", choices=("P", "N"), required=True)
    parser.add_argument("--output-root", default="artifacts/phase1_lsm")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    trajectory_dir = output_root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    p_path = trajectory_dir / "P.txt"
    n_path = trajectory_dir / "N.txt"
    if not p_path.exists():
        write_trajectory(p_path, 1)
    if not n_path.exists():
        write_trajectory(n_path, -1)
    manifest_path = output_root / "fixed_8_sample_manifest.json"
    if not manifest_path.exists():
        write_fixed_manifest(manifest_path)

    selected = p_path if args.trajectory == "P" else n_path
    sample_dir = prepare_condition(
        args.source,
        args.trajectory,
        selected,
        output_root,
        torch.device(args.device),
    )
    print(sample_dir)


if __name__ == "__main__":
    main()
