#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


PHYSICAL_GPUS = (0, 1, 2)
OOM_MARKERS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
)


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def query_gpus() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        if values[0] in PHYSICAL_GPUS:
            rows.append(dict(zip(
                ("index", "total_mib", "used_mib", "free_mib", "utilization"),
                values,
            )))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    attempted = []
    history = []
    for attempt in range(1, len(PHYSICAL_GPUS) + 1):
        rows = query_gpus()
        candidates = [
            row for row in rows if row["index"] not in attempted
        ]
        if not candidates:
            break
        selected = max(
            candidates,
            key=lambda row: (row["free_mib"], -row["utilization"]),
        )
        gpu = selected["index"]
        attempted.append(gpu)
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(free_local_port()),
            "PYTHONPATH": ".",
        })
        stdout_path = log_root / f"attempt{attempt}_gpu{gpu}.stdout.log"
        stderr_path = log_root / f"attempt{attempt}_gpu{gpu}.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        combined = (
            stdout_path.read_text(errors="replace")
            + stderr_path.read_text(errors="replace")
        ).lower()
        oom = any(marker in combined for marker in OOM_MARKERS)
        status = "SUCCESS" if process.returncode == 0 else "CUDA_OOM" if oom else "ERROR"
        history.append({
            "attempt": attempt,
            "physical_gpu": gpu,
            "pre_attempt_gpu_state": selected,
            "status": status,
            "returncode": process.returncode,
        })
        (log_root / "attempts.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if process.returncode == 0:
            print(json.dumps(history, indent=2))
            return
        if not oom:
            sys.stderr.write(combined[-8000:])
            raise SystemExit(process.returncode)
    raise SystemExit("OOM_NO_GPU")


if __name__ == "__main__":
    main()
