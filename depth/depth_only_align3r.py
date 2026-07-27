"""Persistent subprocess adapter for Align3R historical-memory depth."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


READY_PREFIX = "ALIGN3R_WORKER_READY\t"
RESULT_PREFIX = "ALIGN3R_WORKER_RESULT\t"
ERROR_PREFIX = "ALIGN3R_WORKER_ERROR\t"


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid scalar-first quaternion: {q}")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_align3r_reconstruction(
    reconstruction_dir: Path,
    frame_count: int,
    output_size: Tuple[int, int],
    output_device: torch.device,
):
    """Load one Align3R block and resize RGB-D/K to the memory resolution."""
    reconstruction_dir = Path(reconstruction_dir)
    depths = []
    rgbs = []
    native_shape = None
    for index in range(frame_count):
        depth_path = reconstruction_dir / f"frame_{index:04d}.npy"
        rgb_path = reconstruction_dir / f"frame_{index:04d}_rgb.png"
        if not depth_path.is_file() or not rgb_path.is_file():
            raise FileNotFoundError(
                f"Missing Align3R frame {index}: {depth_path} / {rgb_path}"
            )
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
            raise ValueError(
                f"Misaligned Align3R RGB-D at frame {index}: {rgb.shape}, {depth.shape}"
            )
        if native_shape is None:
            native_shape = depth.shape
        elif depth.shape != native_shape:
            raise ValueError(f"Inconsistent Align3R depth shape: {depth.shape}")
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"Align3R produced invalid depth at frame {index}")
        depths.append(depth)
        rgbs.append(rgb)

    intrinsic_rows = np.atleast_2d(
        np.loadtxt(reconstruction_dir / "pred_intrinsics.txt", dtype=np.float64)
    )
    if intrinsic_rows.shape == (1, 9):
        intrinsic_rows = np.repeat(intrinsic_rows, frame_count, axis=0)
    if intrinsic_rows.shape != (frame_count, 9):
        raise ValueError(f"Unexpected Align3R intrinsics: {intrinsic_rows.shape}")
    intrinsics = intrinsic_rows.reshape(frame_count, 3, 3).astype(np.float32)

    pose_rows = np.atleast_2d(
        np.loadtxt(reconstruction_dir / "pred_traj.txt", dtype=np.float64)
    )
    if pose_rows.shape != (frame_count, 8):
        raise ValueError(f"Unexpected Align3R poses: {pose_rows.shape}")
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    for index, row in enumerate(pose_rows):
        c2w[index, :3, :3] = _quaternion_wxyz_to_matrix(row[4:8])
        c2w[index, :3, 3] = row[1:4]
    c2w = np.linalg.inv(c2w[0])[None] @ c2w
    extrinsics = np.linalg.inv(c2w)[:, :3, :].astype(np.float32)

    depth_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(depths))).to(
        output_device, dtype=torch.float32
    )
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(rgbs))).to(
        output_device, dtype=torch.float32
    ).permute(0, 3, 1, 2).div_(255.0)
    native_height, native_width = native_shape
    height, width = output_size
    if (native_height, native_width) != (height, width):
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
        rgb_tensor = F.interpolate(
            rgb_tensor, size=(height, width), mode="bilinear", align_corners=False
        )
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= width / native_width
    intrinsics[:, 1, :] *= height / native_height
    intrinsic_tensor = torch.from_numpy(intrinsics).to(
        output_device, dtype=torch.float32
    )
    extrinsic_tensor = torch.from_numpy(extrinsics).to(
        output_device, dtype=torch.float32
    )
    return rgb_tensor.clamp_(0, 1), depth_tensor, intrinsic_tensor, extrinsic_tensor


class Align3RDepthEstimator:
    """Run all generated frames of each STAR chunk through a persistent worker."""

    backend_name = "align3r"

    def __init__(
        self,
        *,
        python_executable: str,
        worker_script: str,
        align3r_root: str,
        weights: str,
        work_dir: str,
        cuda_visible_devices: str,
        torch_home: str | None = None,
        xdg_config_home: str | None = None,
        disable_curope: bool = False,
    ):
        for path in (python_executable, worker_script, align3r_root, weights):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing Align3R memory dependency: {path}")
        self.device = torch.device("cpu")
        self.work_dir = Path(work_dir).resolve()
        self.session_dir = self.work_dir / f"worker_pid_{os.getpid()}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.job_index = 0
        self.last_native_shape = None
        self.last_processed_shape = None
        self.last_intrinsics_shape = None
        self.last_extrinsics_shape = None
        self.last_peak_memory_gb = 0.0
        self.last_backend_metadata = {}

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        environment["PYTHONPATH"] = str(align3r_root)
        environment["ALIGN3R_DISABLE_CUROPE"] = "1" if disable_curope else "0"
        if torch_home:
            environment["TORCH_HOME"] = torch_home
        if xdg_config_home:
            environment["XDG_CONFIG_HOME"] = xdg_config_home
        command = [
            python_executable,
            "-u",
            worker_script,
            "--align3r-root", align3r_root,
            "--weights", weights,
            "--device", "cuda",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._read_protocol(READY_PREFIX)

    def _read_protocol(self, prefix: str) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("Align3R worker stdout is unavailable")
        while True:
            line = self.process.stdout.readline()
            if line == "":
                code = self.process.poll()
                raise RuntimeError(f"Align3R worker exited before response (code={code})")
            line = line.rstrip("\n")
            if line.startswith(ERROR_PREFIX):
                payload = json.loads(line[len(ERROR_PREFIX):])
                raise RuntimeError(f"Align3R worker failed: {payload}")
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])
            print(f"[Align3R worker] {line}", flush=True)

    def estimate(self, rgb_chw, output_size, output_device):
        raise ValueError("Align3R memory backend requires full-block generated frames")

    @torch.inference_mode()
    def estimate_block(
        self,
        rgb_tchw: torch.Tensor,
        output_size: Tuple[int, int],
        output_device: torch.device,
    ):
        if rgb_tchw.ndim != 4 or rgb_tchw.shape[1] != 3:
            raise ValueError(f"Expected RGB [T,3,H,W], got {tuple(rgb_tchw.shape)}")
        frame_count = int(rgb_tchw.shape[0])
        job_dir = self.session_dir / f"block_{self.job_index:04d}"
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True)
        frames = (
            rgb_tchw.detach().float().clamp(0, 1)
            .permute(0, 2, 3, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        for index, frame in enumerate(frames):
            Image.fromarray(frame, mode="RGB").save(
                input_dir / f"frame_{index:04d}.png"
            )

        request = {
            "command": "estimate",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "sequence_name": "reconstruction",
            "frame_count": frame_count,
        }
        (job_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        if self.process.stdin is None:
            raise RuntimeError("Align3R worker stdin is unavailable")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._read_protocol(RESULT_PREFIX)
        (job_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        reconstruction_dir = output_dir / "reconstruction"
        tensors = load_align3r_reconstruction(
            reconstruction_dir, frame_count, output_size, output_device
        )
        rgb, depth, intrinsics, extrinsics = tensors
        self.last_native_shape = tuple(
            np.load(reconstruction_dir / "frame_0000.npy", mmap_mode="r").shape
        )
        self.last_processed_shape = tuple(rgb.shape)
        self.last_intrinsics_shape = tuple(intrinsics.shape)
        self.last_extrinsics_shape = tuple(extrinsics.shape)
        self.last_peak_memory_gb = float(result.get("peak_memory_gb", 0.0))
        self.last_backend_metadata = {
            "align3r_job_index": self.job_index,
            "align3r_job_dir": str(job_dir),
            "align3r_elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
            "align3r_retry_count": int(result.get("retry_count", 0)),
            "align3r_retry_reason": result.get("retry_reason"),
            "align3r_retry_reasons": result.get("retry_reasons", []),
            "align3r_retry_mode": result.get("retry_mode", "default"),
            "align3r_temporal_smoothing_weight": float(
                result.get("temporal_smoothing_weight", 0.01)
            ),
            "align3r_flow_loss_weight": float(
                result.get("flow_loss_weight", 0.01)
            ),
        }
        self.job_index += 1
        return tensors

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        process = self.process
        self.process = None
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"command": "close"}) + "\n")
                    process.stdin.flush()
                process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
