#!/usr/bin/env python3
"""Create a dense, exact-length 0 -> 45 -> 0 -> 45 yaw trajectory."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 4:
        raise ValueError("At least four frames are required")

    last = args.frames - 1
    landmarks = np.array([0, round(last / 3), round(2 * last / 3), last])
    yaw = np.interp(
        np.arange(args.frames), landmarks, np.array([0.0, 45.0, 0.0, 45.0])
    )
    pitch = np.zeros(args.frames)
    radius = np.zeros(args.frames)
    with open(args.output, "w") as handle:
        for values in (pitch, yaw, radius):
            handle.write(" ".join(f"{value:.12g}" for value in values) + "\n")
    print(
        f"frames={args.frames} landmarks={landmarks.tolist()} "
        f"yaw={[float(yaw[index]) for index in landmarks]} output={args.output}"
    )


if __name__ == "__main__":
    main()
