import argparse
import gc
import json
import os
import time

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import denoise_block
from utils.misc import set_seed
from utils.render_warper import convert_mask_video

# ============================================================================
# Argument parsing
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--json_path", type=str, help="Path to the json file")
parser.add_argument("--version", type=str, default="version_0", help="Output version subfolder name")

# --- Acceleration options ---
parser.add_argument("--use_tae", action="store_true", help="Use Tiny Auto Encoder (TAE) instead of WanVAE")
parser.add_argument("--tae_checkpoint_path", type=str, default=None, help="Path to TAE checkpoint file")
parser.add_argument("--compile_dit", action="store_true", help="Apply torch.compile to the DiT model")

# --- Training-free historical RGB point-memory baseline ---
parser.add_argument("--historical_memory", action="store_true", help="Enable historical RGB point-cloud memory")
parser.add_argument(
    "--memory_depth_backend",
    choices=["da3", "align3r"],
    default="da3",
    help="Depth backend used to write generated historical-memory frames",
)
parser.add_argument(
    "--memory_map_mode",
    choices=["bounded_voxel", "dense_two_layer", "overlap_voxel_v3"],
    default="bounded_voxel",
    help="Bounded legacy map or append-only dense generated two-layer map",
)
parser.add_argument("--memory_da3_model_path", type=str, default="./checkpoints/DA3", help="DA3 checkpoint used for generated keyframe depth")
parser.add_argument("--memory_depth_device", type=str, default=None, help="Logical CUDA device for DA3 depth (default: DiT device)")
parser.add_argument("--memory_align3r_python", type=str, default=None)
parser.add_argument("--memory_align3r_root", type=str, default=None)
parser.add_argument("--memory_align3r_weights", type=str, default=None)
parser.add_argument("--memory_align3r_work_dir", type=str, default=None)
parser.add_argument("--memory_align3r_gpu", type=str, default=None)
parser.add_argument("--memory_align3r_torch_home", type=str, default=None)
parser.add_argument("--memory_align3r_xdg_config_home", type=str, default=None)
parser.add_argument("--memory_align3r_disable_curope", action="store_true")
parser.add_argument(
    "--memory_update_mode",
    choices=["keyframe", "latent_keyframe", "full_block"],
    default="keyframe",
    help=(
        "Write one legacy STAR-block keyframe, one keyframe per Wan latent, "
        "or every generated RGB frame"
    ),
)
parser.add_argument("--memory_voxel_size", type=float, default=0.02, help="Historical point-cloud voxel size")
parser.add_argument("--memory_max_points", type=int, default=500000, help="Maximum historical point count")
parser.add_argument("--memory_point_size", type=int, default=1, help="Historical point splat size")
parser.add_argument(
    "--memory_anchor_count",
    type=int,
    default=1,
    help="Overlapping DA3 anchors. Only one is implemented; larger values are reserved.",
)
parser.add_argument("--disable_memory_diagnostics", action="store_true", help="Disable historical/fused diagnostic videos")
parser.add_argument("--profile_blocks", action="store_true", help="Record per-block DiT timing")
parser.add_argument("--save_denoised_latents", action="store_true", help="Save final denoised latent tensor for regression checks")

args = parser.parse_args()

# ============================================================================
# Distributed setup
# ============================================================================
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    rank = 0
    set_seed(args.seed)

print(f'[Rank {rank}] Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

# ============================================================================
# Config
# ============================================================================
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

num_frame_per_block = getattr(config, "num_frame_per_block", 3)

# ============================================================================
# Initialize pipeline
# ============================================================================
pipeline = CausalInferencePipeline(config, device=device)

checkpoint_name = "None"
method_name = "default"
if args.checkpoint_path:
    print(f"[Rank {rank}] Loading checkpoint from {args.checkpoint_path}")
    state_dict = load_file(args.checkpoint_path)
    mismatch, missing = pipeline.generator.load_state_dict(state_dict, strict=False)
    print(f"[Rank {rank}] Mismatch: {mismatch}, Missing: {missing}")
    checkpoint_name = args.checkpoint_path.split("/")[-2]
    method_name = args.checkpoint_path.split("/")[-3]

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)

# ============================================================================
# Initialize VAE or TAE
# ============================================================================
tae_model = None

if args.use_tae:
    from utils.taehv import TAEHV

    assert args.tae_checkpoint_path is not None, "--tae_checkpoint_path is required when --use_tae is set"
    print(f"[Rank {rank}] Loading TAE from {args.tae_checkpoint_path}...")

    tae_model = TAEHV(checkpoint_path=args.tae_checkpoint_path).to(device, torch.float16)
    tae_model.eval()

    # TAE warmup
    print(f"[Rank {rank}] Warming up TAE...")
    with torch.no_grad():
        dummy_enc = torch.randn(1, 9, 3, 480, 832, device=device, dtype=torch.float16)
        _ = tae_model.encode_video(dummy_enc, show_progress_bar=False)
        dummy_lat = torch.randn(1, 3, tae_model.latent_channels, 60, 104, device=device, dtype=torch.float16)
        _ = tae_model.decode_video(dummy_lat, show_progress_bar=False)
        del dummy_enc, dummy_lat
    torch.cuda.synchronize(device)
    print(f"[Rank {rank}] TAE warmup complete.")
else:
    pipeline.vae.to(device=device)

# ============================================================================
# torch.compile for DiT
# ============================================================================
if args.compile_dit:
    print(f"[Rank {rank}] Compiling DiT model with torch.compile (mode=max-autotune)...")

    import torch._inductor.config as inductor_config
    inductor_config.fx_graph_cache = True
    torch._dynamo.config.cache_size_limit = 32

    # Use /dev/shm (tmpfs) for inductor cache to avoid fcntl.flock issues
    # on certain filesystems where unlink + flock causes FileNotFoundError
    cache_dir = f"/dev/shm/torchinductor_cache_rank{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    pipeline.generator.model = torch.compile(
        pipeline.generator.model,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
        backend="inductor",
    )
    print(f"[Rank {rank}] DiT model compiled.")

# ============================================================================
# DiT warmup
# ============================================================================
pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)


def reset_kv_cache():
    for block_cache in pipeline.kv_cache1:
        block_cache['k'].detach_().zero_()
        block_cache['v'].detach_().zero_()


print(f"[Rank {rank}] Warming up DiT...")
t_warmup_start = time.time()

with torch.no_grad():
    F_warmup = num_frame_per_block
    dummy_noise = torch.randn(1, F_warmup, 16, 60, 104, device=device, dtype=torch.bfloat16)
    dummy_render = torch.randn(1, F_warmup, 20, 60, 104, device=device, dtype=torch.bfloat16)
    dummy_cond = {"prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)}

    if args.compile_dit:
        # Warm up each distinct kv_size pattern to trigger compilation
        if num_frame_per_block == 1:
            warmup_ctx_sizes = [1, 3, 5, 6]
        else:
            warmup_ctx_sizes = [3, 6]

        for wi, n_ctx in enumerate(warmup_ctx_sizes):
            kv_size = n_ctx * 1560
            dummy_ctx = torch.randn(1, n_ctx, 36, 60, 104, device=device, dtype=torch.bfloat16)
            print(f"[Rank {rank}]   Compile warmup pattern {wi + 1}/{len(warmup_ctx_sizes)} (kv_size={kv_size})...")
            t_pat = time.time()

            for _ in range(3):
                reset_kv_cache()
                denoise_block(
                    pipeline.generator, pipeline.scheduler, dummy_noise, dummy_cond,
                    pipeline.kv_cache1,
                    context_frames=dummy_ctx, context_no_grad=True, context_freqs_offset=0,
                    render_block=dummy_render, denoising_kv_size=kv_size,
                    denoising_steps=pipeline.denoising_step_list,
                )

            torch.cuda.synchronize(device)
            print(f"[Rank {rank}]     Pattern {wi + 1} done ({time.time() - t_pat:.1f}s)")
            torch.cuda.empty_cache()
            gc.collect()
    else:
        # Simple warmup
        dummy_ctx = torch.randn(1, 3, 36, 60, 104, device=device, dtype=torch.bfloat16)
        reset_kv_cache()
        denoise_block(
            pipeline.generator, pipeline.scheduler, dummy_noise, dummy_cond,
            pipeline.kv_cache1,
            context_frames=dummy_ctx, context_no_grad=True, context_freqs_offset=0,
            render_block=dummy_render, denoising_kv_size=1560 * 3,
            denoising_steps=pipeline.denoising_step_list,
        )
        torch.cuda.synchronize(device)

    reset_kv_cache()

del dummy_noise, dummy_render, dummy_cond
torch.cuda.empty_cache()
gc.collect()
print(f"[Rank {rank}] DiT warmup complete ({time.time() - t_warmup_start:.1f}s).")

# ============================================================================
# VAE warmup (only when not using TAE)
# ============================================================================
if not args.use_tae:
    print(f"[Rank {rank}] Warming up VAE...")
    with torch.no_grad():
        vae_mean = pipeline.vae.mean.to(device=device, dtype=torch.bfloat16)
        vae_inv_std = (1.0 / pipeline.vae.std).to(device=device, dtype=torch.bfloat16)
        scale = [vae_mean, vae_inv_std]
        dummy_enc = torch.randn(1, 3, 9, 480, 832, device=device, dtype=torch.bfloat16)
        _ = pipeline.vae.model.encode(dummy_enc, scale)
        pipeline.vae.model.clear_cache()
        dummy_dec = torch.randn(1, 16, 3, 60, 104, device=device, dtype=torch.bfloat16)
        _ = pipeline.vae.model.decode(dummy_dec, scale)
        pipeline.vae.model.clear_cache()
        del dummy_enc, dummy_dec
    torch.cuda.synchronize(device)
    print(f"[Rank {rank}] VAE warmup complete.")

# ============================================================================
# Historical-memory-only models
# ============================================================================
memory_depth_estimator = None
memory_decode_vae = None
if args.historical_memory:
    if args.memory_map_mode == "dense_two_layer" and args.memory_update_mode not in {
        "latent_keyframe", "full_block"
    }:
        raise ValueError(
            "--memory_map_mode dense_two_layer requires "
            "--memory_update_mode latent_keyframe or full_block"
        )
    if args.memory_map_mode == "overlap_voxel_v3":
        if args.memory_update_mode != "latent_keyframe":
            raise ValueError("overlap_voxel_v3 requires latent_keyframe updates")
        if args.memory_depth_backend != "da3":
            raise ValueError("overlap_voxel_v3 currently requires DA3")
        if args.memory_anchor_count != 1:
            raise NotImplementedError(
                "Multi-anchor DA3 windows are reserved but not implemented"
            )
    from utils.wan_wrapper import WanVAEWrapper

    if args.memory_depth_backend == "da3":
        from depth.depth_only_da3 import DA3DepthOnlyEstimator

        memory_depth_device = (
            torch.device(args.memory_depth_device) if args.memory_depth_device else device
        )
        print(
            f"[Rank {rank}] Loading historical-memory DA3 model on {memory_depth_device} "
            f"(map_mode={args.memory_map_mode}, update_mode={args.memory_update_mode})..."
        )
        memory_depth_estimator = DA3DepthOnlyEstimator(
            model_path=args.memory_da3_model_path,
            device=memory_depth_device,
        )
        memory_depth_estimator.backend_name = "da3"
    else:
        from depth.depth_only_align3r import Align3RDepthEstimator

        required_align3r = {
            "--memory_align3r_python": args.memory_align3r_python,
            "--memory_align3r_root": args.memory_align3r_root,
            "--memory_align3r_weights": args.memory_align3r_weights,
            "--memory_align3r_work_dir": args.memory_align3r_work_dir,
            "--memory_align3r_gpu": args.memory_align3r_gpu,
        }
        missing_align3r = [name for name, value in required_align3r.items() if not value]
        if missing_align3r:
            raise ValueError(
                "Align3R memory backend is missing: " + ", ".join(missing_align3r)
            )
        print(
            f"[Rank {rank}] Starting persistent Align3R memory worker on physical GPU "
            f"{args.memory_align3r_gpu} (map_mode={args.memory_map_mode}, "
            f"update_mode={args.memory_update_mode})..."
        )
        memory_depth_estimator = Align3RDepthEstimator(
            python_executable=args.memory_align3r_python,
            worker_script=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "scripts",
                "align3r_memory_worker.py",
            ),
            align3r_root=args.memory_align3r_root,
            weights=args.memory_align3r_weights,
            work_dir=args.memory_align3r_work_dir,
            cuda_visible_devices=args.memory_align3r_gpu,
            torch_home=args.memory_align3r_torch_home,
            xdg_config_home=args.memory_align3r_xdg_config_home,
            disable_curope=args.memory_align3r_disable_curope,
        )

    if not args.use_tae:
        print(f"[Rank {rank}] Loading dedicated block-decode WanVAE...")
        memory_decode_vae = WanVAEWrapper(model_folder=config.wan_model_folder)
        memory_decode_vae = memory_decode_vae.to(device=device, dtype=torch.bfloat16)
        memory_decode_vae.eval()

# ============================================================================
# TAE encode / decode helpers
# ============================================================================
def tae_encode(video_bcthw: torch.Tensor) -> torch.Tensor:
    """Encode [B,C,T,H,W] [-1,1] bf16 -> [B,T_lat,C_lat,H_lat,W_lat] bf16."""
    video = video_bcthw.permute(0, 2, 1, 3, 4)                       # -> [B,T,C,H,W]
    video = ((video * 0.5 + 0.5).clamp(0, 1)).to(torch.float16)      # -> [0,1] fp16
    latent = tae_model.encode_video(video, show_progress_bar=False)   # NTCHW
    return latent.to(torch.bfloat16)


def tae_decode(latent: torch.Tensor) -> torch.Tensor:
    """Decode [B,T_lat,C_lat,H_lat,W_lat] bf16 -> [B,T,C,H,W] [0,1] float32."""
    video = tae_model.decode_video(latent.to(torch.float16), show_progress_bar=False)
    return video.float()


def encode_video(video_bcthw: torch.Tensor) -> torch.Tensor:
    """Unified encode: TAE or VAE depending on args."""
    if args.use_tae:
        return tae_encode(video_bcthw)
    return pipeline.vae.encode_to_latent(video_bcthw).to(device, dtype=torch.bfloat16)


# ============================================================================
# Dataset
# ============================================================================
from datasets.video_dataset import VideoDataset

dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
if args.json_path:
    dataset_config['json_path'] = args.json_path
dataset = VideoDataset(**dataset_config)
print(f"[Rank {rank}] Number of videos: {len(dataset)}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

output_dir = os.path.join(args.output_folder, method_name, checkpoint_name)
os.makedirs(output_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ============================================================================
# Inference loop
# ============================================================================
for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0), desc=f"Rank {rank}"):
    if dist.is_initialized():
        global_idx = i * world_size + rank
    else:
        global_idx = i

    batch = batch_data if isinstance(batch_data, dict) else batch_data[0]
    video_output_dir = os.path.join(
        args.output_folder, method_name, checkpoint_name, args.version
    )
    os.makedirs(video_output_dir, exist_ok=True)

    # Load pre-rendered render/mask videos from batch (produced by offline point cloud rendering)
    render_videos_ori = batch["render_video"].to(device, dtype=torch.bfloat16)
    render_videos_ori = rearrange(render_videos_ori, 'b t c h w -> b c t h w')
    mask_videos_ori = batch["mask_video"].to(device, dtype=torch.bfloat16)
    mask_videos_ori = rearrange(mask_videos_ori, 'b t c h w -> b c t h w')

    # --- VAE Encode ---
    torch.cuda.synchronize(device)
    t_enc_start = time.time()

    if args.historical_memory:
        # Current-block fused conditions are encoded by block_condition_provider.
        render_latent = None
        mask_latent = None
    else:
        render_latent = encode_video(render_videos_ori)
        mask_latent = convert_mask_video(mask_videos_ori)

    text_prompts = batch["text"]
    if "target_video" in batch:
        target_video = batch["target_video"].to(device=device, dtype=torch.bfloat16)
    else:
        target_video = batch["source_video"].to(device=device, dtype=torch.bfloat16)
    target_video = rearrange(target_video, 'b t c h w -> b c t h w')
    latent = encode_video(target_video)

    ref_video = batch["source_video"].to(device=device, dtype=torch.bfloat16)
    ref_video = rearrange(ref_video, 'b t c h w -> b c t h w')
    ref_latent = encode_video(ref_video)

    torch.cuda.synchronize(device)
    t_enc_end = time.time()

    latent_length = latent.shape[1]
    if latent_length % config.num_frame_per_block != 0:
        num_output_frames = latent_length - latent_length % config.num_frame_per_block
    else:
        num_output_frames = latent_length
    sampled_noise = torch.randn(
        [args.num_samples, num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    if not args.historical_memory:
        render_latent = render_latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)
        mask_latent = mask_latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)
    latent = latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)

    # --- DiT inference (decode=False when using TAE, True when using VAE) ---
    memory_controller = None
    block_profile = []
    block_condition_provider = None
    block_output_callback = None

    if args.historical_memory:
        if args.num_samples != 1:
            raise ValueError("Historical memory baseline currently requires --num_samples 1")
        if "target_c2w" not in batch or "target_intrinsic" not in batch:
            raise RuntimeError(
                "Exact target_c2w/intrinsic are missing. Re-run Step 2b with the modified "
                "scripts/render_point_cloud.py before enabling historical memory."
            )

        from pipeline.historical_memory_controller import (
            HistoricalMemoryController,
            TAEBlockDecoder,
            WanVAEBlockDecoder,
        )
        from utils.historical_point_memory import (
            DenseGeneratedPointMemory,
            IncrementalVoxelSurfelMemory,
            RGBPointMemory,
        )

        target_c2w = batch["target_c2w"].to(device=device, dtype=torch.float32)
        target_intrinsic = batch["target_intrinsic"].to(device=device, dtype=torch.float32)
        memory_kwargs = {
            "height": render_videos_ori.shape[-2],
            "width": render_videos_ori.shape[-1],
            "device": device,
            "K": target_intrinsic[0],
            "point_size": args.memory_point_size,
        }
        if args.memory_map_mode in {"dense_two_layer", "overlap_voxel_v3"}:
            if "reference_depth" not in batch:
                raise RuntimeError(
                    "Aligned reference depth is missing. Re-run Step 2b to generate "
                    "render/depth_offline.npy before using dense_two_layer."
                )
            if args.memory_map_mode == "dense_two_layer":
                memory = DenseGeneratedPointMemory(**memory_kwargs)
            else:
                memory = IncrementalVoxelSurfelMemory(
                    **memory_kwargs,
                    voxel_size=args.memory_voxel_size,
                    max_points=args.memory_max_points,
                )
        else:
            memory = RGBPointMemory(
                **memory_kwargs,
                voxel_size=args.memory_voxel_size,
                max_points=args.memory_max_points,
            )
        if args.use_tae:
            block_decoder = TAEBlockDecoder(tae_model)
        else:
            block_decoder = WanVAEBlockDecoder(memory_decode_vae)

        dataset_index = int(batch["index"].reshape(-1)[0].item())
        reference_map_path = (
            dataset.metadata_list[dataset_index]["vggt_depth_path"]
            + "_da3_tmp/frames_pcd"
        )
        memory_controller = HistoricalMemoryController(
            reference_rgb_bcthw=render_videos_ori,
            reference_mask_bcthw=mask_videos_ori,
            target_c2w=target_c2w,
            K=target_intrinsic,
            encode_video=encode_video,
            block_decoder=block_decoder,
            depth_estimator=memory_depth_estimator,
            memory=memory,
            output_dir=video_output_dir,
            output_prefix=str(global_idx),
            rank=rank,
            reference_map_path=reference_map_path,
            memory_update_mode=args.memory_update_mode,
            memory_map_mode=args.memory_map_mode,
            reference_depth_thw=batch.get("reference_depth"),
            memory_anchor_count=args.memory_anchor_count,
            save_diagnostics=not args.disable_memory_diagnostics,
        )
        block_condition_provider = memory_controller.condition_provider
        block_output_callback = memory_controller.output_callback
    elif args.profile_blocks:
        def record_block_timing(*, block_index, latent_start, denoised_latent, dit_ms):
            block_profile.append({
                "block_index": int(block_index),
                "latent_start": int(latent_start),
                "latent_frames": int(denoised_latent.shape[1]),
                "dit_ms": float(dit_ms),
            })

        block_output_callback = record_block_timing

    torch.cuda.synchronize(device)
    t_dit_start = time.time()

    return_latents = args.historical_memory or args.use_tae or args.save_denoised_latents
    result = pipeline.inference(
        noise=sampled_noise,
        text_prompts=text_prompts,
        ref_latent=ref_latent,
        render_latent=render_latent,
        mask_latent=mask_latent,
        decode=not return_latents,
        block_condition_provider=block_condition_provider,
        block_output_callback=block_output_callback,
    )

    torch.cuda.synchronize(device)
    t_dit_end = time.time()

    memory_summary = None
    if memory_controller is not None:
        memory_summary = memory_controller.close()
        print(f"[Rank {rank}] Historical memory summary: {json.dumps(memory_summary, sort_keys=True)}")

    if block_profile:
        with open(
            os.path.join(video_output_dir, f"{global_idx}-block_profile_rank{rank}.json"),
            "w",
        ) as handle:
            json.dump(block_profile, handle, indent=2)

    if args.save_denoised_latents:
        if not return_latents:
            raise AssertionError("save_denoised_latents requires latent output")
        torch.save(
            result.detach().cpu(),
            os.path.join(video_output_dir, f"{global_idx}-denoised_latent_rank{rank}.pt"),
        )

    # --- VAE Decode ---
    torch.cuda.synchronize(device)
    t_dec_start = time.time()

    if args.historical_memory:
        # The callback already decoded and streamed every generated block.
        current_video = None
    elif args.use_tae:
        # result is denoised latents, decode with TAE
        video_out = tae_decode(result)
        current_video = rearrange(video_out, 'b t c h w -> b t h w c').cpu()
    elif args.save_denoised_latents:
        video_out = pipeline.vae.decode_to_pixel(result, use_cache=False)
        video_out = (video_out * 0.5 + 0.5).clamp(0, 1)
        current_video = rearrange(video_out, 'b t c h w -> b t h w c').cpu()
    else:
        # result is already decoded [0,1] video
        current_video = rearrange(result, 'b t c h w -> b t h w c').cpu()

    torch.cuda.synchronize(device)
    t_dec_end = time.time()

    # --- Timing summary ---
    print(f"[Rank {rank}] Video {global_idx} timing: "
          f"VAE Encode={t_enc_end - t_enc_start:.2f}s, "
          f"DiT/online={'(+VAE Dec) ' if not return_latents else ''}{t_dit_end - t_dit_start:.2f}s, "
          f"{'block-streamed' if args.historical_memory else ('TAE' if args.use_tae else 'VAE')} Decode={t_dec_end - t_dec_start:.2f}s, "
          f"Total={t_dec_end - t_enc_start:.2f}s")

    source_video = rearrange(target_video, 'b c t h w -> b t h w c').cpu()
    source_video = (source_video * 0.5 + 0.5).clamp(0, 1)

    render_video = rearrange(render_videos_ori, 'b c t h w -> b t h w c').cpu()
    render_video = (render_video * 0.5 + 0.5).clamp(0, 1)

    pred_video = None if current_video is None else 255.0 * current_video
    source_video_out = 255.0 * source_video
    render_video_out = 255.0 * render_video

    if not args.use_tae:
        pipeline.vae.model.clear_cache()

    for seed_idx in range(args.num_samples):
        if pred_video is not None:
            write_video(os.path.join(video_output_dir, f'{global_idx}-pred_video_rank{rank}.mp4'), pred_video[seed_idx], fps=24)
        write_video(os.path.join(video_output_dir, f'{global_idx}-source_video_rank{rank}.mp4'), source_video_out[seed_idx], fps=24)
        write_video(os.path.join(video_output_dir, f'{global_idx}-render_video_rank{rank}.mp4'), render_video_out[seed_idx], fps=24)

        if 'target_extrinsics' in batch:
            target_extrinsics = batch["target_extrinsics"].float().to(device=device)
            torch.save(target_extrinsics, os.path.join(video_output_dir, f'extrinsics_{global_idx}.pt'))
        if 'target_c2w' in batch:
            torch.save(
                batch["target_c2w"].float(),
                os.path.join(video_output_dir, f'target_c2w_{global_idx}.pt'),
            )

    if args.historical_memory:
        # A completed sample keeps all generated chunks resident through map
        # export. Release it only after every artifact is written so the next
        # video starts from a clean allocator state.
        block_condition_provider = None
        block_output_callback = None
        memory_controller = None
        del memory
        gc.collect()
        torch.cuda.empty_cache()

if memory_depth_estimator is not None and hasattr(memory_depth_estimator, "close"):
    memory_depth_estimator.close()

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

print(f"[Rank {rank}] Inference completed!")
