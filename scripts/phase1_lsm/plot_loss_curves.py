#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = (
    ("smoke_direct", "Single sample: direct", "#1f77b4"),
    ("smoke_projected", "Single sample: projected", "#ff7f0e"),
    ("fixed8_direct", "Fixed 8: direct", "#2ca02c"),
    ("fixed8_projected", "Fixed 8: projected", "#d62728"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.train_root)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plotted = {}
    for axis, (stage, title, color) in zip(axes.flat, STAGES):
        stage_dir = root / stage
        if stage.startswith("smoke"):
            records = json.loads((stage_dir / "loss_curve.json").read_text())
            steps = [record["step"] for record in records]
            losses = [record["loss"] for record in records]
            label = "optimizer-step loss"
        else:
            records = json.loads((stage_dir / "aggregate_curve.json").read_text())
            steps = [record["step"] for record in records]
            losses = [record["mean_loss"] for record in records]
            label = "8-sample mean"
        axis.plot(steps, losses, color=color, linewidth=2, label=label)
        axis.scatter([steps[0], steps[-1]], [losses[0], losses[-1]], color=color)
        axis.set(title=title, xlabel="optimizer step", ylabel="loss")
        axis.grid(alpha=0.25)
        axis.legend()
        plotted[stage] = {
            "initial": losses[0],
            "final": losses[-1],
            "ratio": losses[-1] / losses[0],
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(plotted, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(plotted, indent=2))


if __name__ == "__main__":
    main()
