#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PHYSICAL_GPUS = (0, 1, 2)
OOM_MARKERS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
)


def query_gpus() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,uuid",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        index, total, used, free, utilization, uuid = [
            value.strip() for value in line.split(",", 5)
        ]
        index_int = int(index)
        if index_int in PHYSICAL_GPUS:
            rows.append({
                "index": index_int,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_percent": int(utilization),
                "uuid": uuid,
            })
    return rows


def process_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", "0,1,2"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def format_snapshot(rows: list[dict[str, object]], processes: str) -> str:
    lines = [
        "| physical GPU | total MiB | used MiB | free MiB | util % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["index"]):
        lines.append(
            f"| {row['index']} | {row['total_mib']} | {row['used_mib']} | "
            f"{row['free_mib']} | {row['utilization_percent']} |"
        )
    lines.extend(["", "nvidia-smi processes:", "", "~~~text", processes.rstrip(), "~~~"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-base",
        default="artifacts/phase1_lsm/phase1_wide_sharedA_hardgate",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    result_base = Path(args.result_base)
    result_base.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    attempted: list[int] = []
    history: list[dict[str, object]] = []
    python = sys.executable

    while len(attempted) < len(PHYSICAL_GPUS):
        before_rows = query_gpus()
        candidates = [
            row for row in before_rows if int(row["index"]) not in attempted
        ]
        if not candidates:
            break
        selected = max(candidates, key=lambda row: int(row["free_mib"]))
        gpu = int(selected["index"])
        attempted.append(gpu)
        attempt_number = len(attempted)
        root = result_base / f"{run_id}_attempt{attempt_number}_gpu{gpu}"
        root.mkdir()
        before_processes = process_snapshot()
        gpu_log = root / "GPU_ATTEMPTS.md"
        gpu_log.write_text(
            "# GPU attempts\n\n"
            f"- Attempt: {attempt_number}\n"
            f"- Selected physical GPU: {gpu}\n"
            f"- Selection: highest current free memory among untried GPU 0/1/2\n"
            f"- Start: {datetime.now().astimezone().isoformat()}\n\n"
            "## Pre-attempt snapshot\n\n"
            + format_snapshot(before_rows, before_processes)
            + "\n",
            encoding="utf-8",
        )
        command = [
            python,
            "scripts/phase1_lsm/run_wide_sharedA_hardgate.py",
            "--output-root",
            str(root),
            "--checkpoint",
            args.checkpoint,
            "--init-adapter",
            args.init_adapter,
            "--repo-root",
            args.repo_root,
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29640 + attempt_number),
        })
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(
                command,
                cwd=args.repo_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        after_rows = query_gpus()
        after_processes = process_snapshot()
        combined = (
            stdout_path.read_text(errors="replace")
            + "\n"
            + stderr_path.read_text(errors="replace")
        )
        oom = any(marker in combined.lower() for marker in OOM_MARKERS)
        status = "SUCCESS" if process.returncode == 0 else "CUDA_OOM" if oom else "NON_OOM_FAILURE"
        entry = {
            "attempt": attempt_number,
            "physical_gpu": gpu,
            "returncode": process.returncode,
            "status": status,
            "root": str(root),
        }
        history.append(entry)
        with gpu_log.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Post-attempt\n\n"
                f"- Finish: {datetime.now().astimezone().isoformat()}\n"
                f"- Child process exited: True\n"
                f"- Return code: {process.returncode}\n"
                f"- Classification: {status}\n\n"
                "### Post-attempt snapshot\n\n"
                + format_snapshot(after_rows, after_processes)
                + "\n\n### Attempt history\n\n"
                + json.dumps(history, indent=2)
                + "\n"
            )
        if process.returncode == 0:
            (result_base / f"{run_id}_SUCCESS.json").write_text(
                json.dumps({
                    "successful_root": str(root),
                    "successful_physical_gpu": gpu,
                    "attempted_physical_gpus": attempted,
                    "oom_occurred": any(item["status"] == "CUDA_OOM" for item in history),
                    "history": history,
                }, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps(history, indent=2))
            print(f"SUCCESS_ROOT={root}")
            return
        if not oom:
            print(json.dumps(history, indent=2), file=sys.stderr)
            raise SystemExit(process.returncode)

    if history:
        last_root = Path(str(history[-1]["root"]))
        with (last_root / "GPU_ATTEMPTS.md").open("a", encoding="utf-8") as handle:
            handle.write("\nOOM_NO_GPU\n")
    print(json.dumps(history, indent=2), file=sys.stderr)
    raise SystemExit("OOM_NO_GPU")


if __name__ == "__main__":
    main()
