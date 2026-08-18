from typing import List, Mapping, Optional
import torch
import time
from contextlib import nullcontext
from einops import rearrange
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.render_warper import convert_mask_video


def _model_config_value(model, name):
    if hasattr(model, name):
        return getattr(model, name)
    config = model.config
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)


def denoise_block(
    generator,
    scheduler,
    noisy_input,
    conditional_dict,
    kv_cache,
    *,
    context_frames=None,
    context_no_grad=True,
    context_freqs_offset=0,
    context_kv_size_0=0,
    render_block=None,
    denoising_kv_size=0,
    denoising_kv_size_0=0,
    denoising_steps=None,
    block_id=0,
    after_context_write=None,
    noise_provider=None,
    memory_context=None,
):
    """
    Shared block-based diffusion core: optional context encoding pass + denoising.

    Returns (denoised_pred, noise_before_last_step).
    """
    B, F = noisy_input.shape[:2]
    device, dtype = noisy_input.device, noisy_input.dtype
    noise_before_last_step = None

    if context_frames is not None:
        times_zero = torch.zeros([B, F], device=device, dtype=torch.int64)
        ctx = torch.no_grad() if context_no_grad else nullcontext()
        with ctx:
            generator(
                noisy_image_or_video=context_frames,
                conditional_dict=conditional_dict,
                timestep=times_zero,
                kv_cache=kv_cache,
                render_latent_input=render_block,
                kv_size=(context_kv_size_0, -1),
                freqs_offset=context_freqs_offset,
            )
        if after_context_write is not None:
            after_context_write(
                block_id=block_id,
                kv_cache=kv_cache,
                context_frames=context_frames,
            )

    cache_snapshots = None
    if memory_context is not None and memory_context.alpha != 0.0:
        cache_snapshots = {
            int(layer): {
                "k": kv_cache[int(layer)]["k"].detach().clone(),
                "v": kv_cache[int(layer)]["v"].detach().clone(),
            }
            for layer in memory_context.layer_payloads
        }

    for index, current_timestep in enumerate(denoising_steps):
        is_last_step = (index == len(denoising_steps) - 1)
        timestep = torch.ones([B, F], device=device, dtype=torch.int64) * current_timestep

        ctx = torch.no_grad() if not is_last_step else nullcontext()
        step_memory = (
            memory_context.for_denoising_step(index, len(denoising_steps))
            if memory_context is not None
            else None
        )
        with ctx:
            generator_kwargs = dict(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                kv_size=(denoising_kv_size_0, denoising_kv_size),
                render_latent_input=render_block,
                freqs_offset=6,
            )
            if step_memory is not None:
                generator_kwargs["memory_context"] = step_memory
            _, denoised_pred = generator(**generator_kwargs)

        if is_last_step:
            noise_before_last_step = noisy_input.clone()
        else:
            next_t = denoising_steps[index + 1]
            if noise_provider is None:
                step_noise = torch.randn_like(denoised_pred)
            else:
                step_noise = noise_provider.get_re_noise(
                    block_id=block_id,
                    step_index=index,
                    like=denoised_pred,
                )
            noisy_input = scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                step_noise.flatten(0, 1),
                next_t * torch.ones([B * F], device=device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

    if cache_snapshots is not None:
        per_layer = {}
        maximum = 0.0
        for layer, before in cache_snapshots.items():
            k_diff = float(
                (kv_cache[layer]["k"].float() - before["k"].float()).abs().max().item()
            )
            v_diff = float(
                (kv_cache[layer]["v"].float() - before["v"].float()).abs().max().item()
            )
            per_layer[str(layer)] = {"k_max_abs_diff": k_diff, "v_max_abs_diff": v_diff}
            maximum = max(maximum, k_diff, v_diff)
        memory_context.cache_audit.update(
            {"per_layer": per_layer, "max_abs_diff": maximum, "unchanged": maximum == 0.0}
        )
        if maximum != 0.0:
            raise RuntimeError(
                f"Auxiliary attention mutated runtime kv_cache1 (max_abs_diff={maximum})"
            )

    return denoised_pred, noise_before_last_step


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        time_start = time.time() 
        self.generator = WanDiffusionWrapper(**getattr(args, "generator", {}), is_causal=True)
        print(f"Time taken to initialize generator: {time.time() - time_start} seconds")

        time_start = time.time()
        wan_model_folder = getattr(args, "wan_model_folder", None)
        self.text_encoder = WanTextEncoder(model_folder=wan_model_folder) if text_encoder is None else text_encoder
        print(f"Time taken to initialize text encoder: {time.time() - time_start} seconds")

        time_start = time.time()
        self.vae = WanVAEWrapper(model_folder=wan_model_folder) if vae is None else vae
        print(f"Time taken to initialize vae: {time.time() - time_start} seconds")

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            print("warping denoising step list")
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = None

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        
        self.max_num_context_frames = 6
        self._layout_printed = False
        self.last_query_gates = {}
        self.last_block_latencies = {}
        self.last_virtual_memory_contexts = {}
        self.last_virtual_recent_audits = {}

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        ref_latent: Optional[torch.Tensor] = None,
        render_latent: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
        decode: bool = True,
        noise_provider=None,
        after_context_write=None,
        memory_contexts: Optional[Mapping[int, object]] = None,
        virtual_recent_contexts: Optional[Mapping[int, object]] = None,
        latent_block_interventions: Optional[Mapping[int, object]] = None,
        conditional_dict: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            decode (bool): If True (default), decode latents to pixel space via VAE.
                If False, return denoised latents directly (e.g. for external TAE decoder).
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                When decode=True, normalized to [0, 1]. When decode=False, raw latents.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0, f"num_frames {num_frames} is not a multiple of num_frame_per_block {self.num_frame_per_block}"
        num_blocks = num_frames // self.num_frame_per_block
        layout = self._runtime_layout(height, width)
        self.frame_seq_length = layout["tokens_per_frame"]
        if (
            memory_contexts
            or virtual_recent_contexts
            or latent_block_interventions
            or after_context_write is not None
        ):
            assert self.num_frame_per_block == 3, (
                "The MapKV prototype currently supports num_frame_per_block=3 only"
            )

        num_output_frames = num_frames   # add the initial latent frames
        if conditional_dict is None:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)
 
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )


        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            latent_height=height,
            latent_width=width,
        )
 
        # Step 3: Temporal denoising loop
        print(f"Generating {num_blocks} blocks...")
        t_start_sampling = time.time() 
        all_num_frames = [self.num_frame_per_block] * num_blocks

        start_index = 0
        last_pred = None
        self.last_query_gates = {}
        self.last_block_latencies = {}
        self.last_virtual_memory_contexts = {}
        self.last_virtual_recent_audits = {}
        for block_id, num_block_frame in enumerate(all_num_frames):
            block_start_time = time.time()
            noisy_input = noise[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            ref_block = ref_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            render_block = render_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            mask_block = mask_latent[:, start_index :start_index + num_block_frame ].to(device=noise.device, dtype=noise.dtype)
            render_block = torch.cat([mask_block, render_block], dim=2)

            recent_slot_len = layout["recent_slot_len"]
            kv_size = recent_slot_len

            # Prepare context
            context_frames = None
            zero_latents = torch.zeros_like(ref_block)
            ref_block = torch.cat([ref_block, zero_latents[:, :, :4], zero_latents], dim=2)
            if start_index == 0:
                context_frames = ref_block
            else:
                # Prepare context similar to training pipeline
                zero_latents = torch.zeros_like(last_pred)
                last_pred_padded = torch.cat([last_pred, zero_latents[:, :, :4], zero_latents], dim=2)
                context_frames = torch.cat([ref_block, last_pred_padded], dim=1)
                kv_size = 2 * recent_slot_len
                assert kv_size == layout["kv_size_used_for_nonfirst_block"]

            block_memory = (memory_contexts or {}).get(block_id)
            virtual_plan = (virtual_recent_contexts or {}).get(block_id)
            if block_memory is not None and virtual_plan is not None:
                raise ValueError(
                    f"Block {block_id} cannot use stored-KV and warp-reencode memory together"
                )
            if virtual_plan is not None:
                if block_id == 0 or last_pred is None:
                    raise ValueError("Virtual recent memory requires a previous generated block")
                if int(virtual_plan.target_block) != block_id:
                    raise ValueError(
                        f"Virtual recent target {virtual_plan.target_block} "
                        f"does not match block {block_id}"
                    )
                if int(virtual_plan.source_chunk) >= block_id - 1:
                    raise ValueError(
                        "Virtual recent source must be older than the immediate previous chunk"
                    )
                virtual_recent = virtual_plan.compose(last_pred)
                layer_payloads, writer_audit = (
                    self.encode_clean_latent_as_recent_slot(
                        reference_context=ref_block,
                        clean_recent_latent=virtual_recent,
                        conditional_dict=conditional_dict,
                        selected_layers=virtual_plan.selected_layers,
                        render_block=render_block,
                    )
                )
                block_memory = virtual_plan.make_memory_context(
                    layer_payloads, writer_audit
                )
                self.last_virtual_memory_contexts[block_id] = block_memory
                self.last_virtual_recent_audits[block_id] = virtual_plan.audit
            if block_memory is not None:
                if block_id == 0:
                    raise ValueError("Memory cannot be activated on the first block")
                if block_memory.target_block != block_id:
                    raise ValueError(
                        f"Memory target {block_memory.target_block} does not match block {block_id}"
                    )
                if block_memory.source_chunk >= block_id - 1:
                    raise ValueError(
                        "Memory source must be older than the immediate previous chunk"
                    )
                block_memory = block_memory.with_query_gate(
                    mask_block, layout["token_hw"]
                )
                self.last_query_gates[block_id] = block_memory.query_gate.detach().cpu()
                if virtual_plan is not None:
                    gate = block_memory.query_gate.float()
                    virtual_plan.artifacts["query_gate_tokens"] = (
                        gate.reshape(
                            gate.shape[0],
                            self.num_frame_per_block,
                            *layout["token_hw"],
                        )
                        .detach()
                        .cpu()
                    )
                    virtual_plan.audit.update(
                        {
                            "query_gate_token_fraction": float(
                                gate.mean().item()
                            ),
                            "query_gate_token_min": float(gate.min().item()),
                            "query_gate_token_max": float(gate.max().item()),
                            "query_gate_token_shape": list(gate.shape),
                        }
                    )

            denoised_pred, _ = denoise_block(
                self.generator, self.scheduler, noisy_input, conditional_dict,
                self.kv_cache1,
                context_frames=context_frames,
                context_no_grad=True,
                context_freqs_offset=0,
                render_block=render_block,
                denoising_kv_size=kv_size,
                denoising_steps=self.denoising_step_list,
                block_id=block_id,
                after_context_write=after_context_write,
                noise_provider=noise_provider,
                memory_context=block_memory,
            )
            latent_intervention = (latent_block_interventions or {}).get(block_id)
            if latent_intervention is not None:
                if int(latent_intervention.target_block) != block_id:
                    raise ValueError(
                        f"Latent intervention target {latent_intervention.target_block} "
                        f"does not match block {block_id}"
                    )
                denoised_pred = latent_intervention.apply(denoised_pred)



            # Step 3.2: record the model's output
            output[:, start_index:start_index + num_block_frame] = denoised_pred
            last_pred = denoised_pred.clone().detach()
            self.last_block_latencies[block_id] = time.time() - block_start_time

            # Step 3.4: update the start and end frame indices
            start_index += num_block_frame
 

        # Step 4: Decode the output
        if not decode:
            return output

        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video

    @torch.no_grad()
    def encode_clean_latent_as_recent_slot(
        self,
        *,
        reference_context: torch.Tensor,
        clean_recent_latent: torch.Tensor,
        conditional_dict,
        selected_layers,
        render_block: torch.Tensor,
        canonical_capture: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict]:
        """Run the native [Ref, Recent] t=0 writer in an isolated cache.

        reference_context is the already padded 36-channel reference block.
        The supplied clean latent is padded exactly like runtime last_pred and
        occupies frames t3-t5, so extracted K has native recent-slot RoPE.
        """
        if reference_context.ndim != 5 or clean_recent_latent.ndim != 5:
            raise ValueError("Reference and virtual recent context must be BFCHW")
        batch, frames, channels, height, width = clean_recent_latent.shape
        if frames != self.num_frame_per_block:
            raise ValueError(
                f"Recent re-encode requires {self.num_frame_per_block} frames, got {frames}"
            )
        if channels != 16:
            raise ValueError(f"Virtual recent latent must have 16 channels, got {channels}")
        if tuple(reference_context.shape) != (batch, frames, 36, height, width):
            raise ValueError(
                "Padded reference context shape mismatch: "
                f"{tuple(reference_context.shape)}"
            )
        selected = tuple(dict.fromkeys(int(layer) for layer in selected_layers))
        if not selected or min(selected) < 0 or max(selected) >= self.num_transformer_blocks:
            raise ValueError(
                f"Recent re-encode layers {selected} are invalid for "
                f"{self.num_transformer_blocks} transformer blocks"
            )
        layout = self._runtime_layout(height, width)
        slot_len = layout["recent_slot_len"]
        cache_len = layout["kv_size_used_for_nonfirst_block"]
        num_heads = int(_model_config_value(self.generator.model, "num_heads"))
        dim = int(_model_config_value(self.generator.model, "dim"))
        cache = [
            {
                "k": torch.zeros(
                    [batch, cache_len, num_heads, dim // num_heads],
                    device=clean_recent_latent.device,
                    dtype=clean_recent_latent.dtype,
                ),
                "v": torch.zeros(
                    [batch, cache_len, num_heads, dim // num_heads],
                    device=clean_recent_latent.device,
                    dtype=clean_recent_latent.dtype,
                ),
            }
            for _ in range(self.num_transformer_blocks)
        ]
        zeros = torch.zeros_like(clean_recent_latent)
        padded_recent = torch.cat(
            [clean_recent_latent, zeros[:, :, :4], zeros], dim=2
        )
        context = torch.cat([reference_context, padded_recent], dim=1)
        timestep_zero = torch.zeros(
            [batch, frames],
            device=clean_recent_latent.device,
            dtype=torch.int64,
        )
        self.generator(
            noisy_image_or_video=context,
            conditional_dict=conditional_dict,
            timestep=timestep_zero,
            kv_cache=cache,
            render_latent_input=render_block,
            kv_size=(0, -1),
            freqs_offset=0,
            canonical_capture=canonical_capture,
        )
        result = {}
        layer_stats = {}
        for layer in selected:
            k = cache[layer]["k"][:, slot_len:2 * slot_len].detach().clone()
            v = cache[layer]["v"][:, slot_len:2 * slot_len].detach().clone()
            if k.shape != v.shape or k.shape[1] != slot_len:
                raise RuntimeError(
                    f"Recent-slot writer produced invalid layer {layer}: "
                    f"K={tuple(k.shape)} V={tuple(v.shape)}"
                )
            result[layer] = (k, v)
            layer_stats[str(layer)] = {
                "k_abs_mean": float(k.float().abs().mean().item()),
                "v_abs_mean": float(v.float().abs().mean().item()),
                "shape": list(k.shape),
            }
        element_size = clean_recent_latent.element_size()
        writer_audit = {
            "mode": "native_clean_timestep_zero_context_writer",
            "context_shape": list(context.shape),
            "reference_slot_range": [0, slot_len],
            "recent_slot_range": [slot_len, 2 * slot_len],
            "rope_layout": "recent_slot_t3_t5",
            "runtime_cache_mutated": False,
            "temporary_cache_bytes": int(
                self.num_transformer_blocks
                * 2
                * batch
                * cache_len
                * num_heads
                * (dim // num_heads)
                * element_size
            ),
            "selected_layers": list(selected),
            "layer_stats": layer_stats,
        }
        return result, writer_audit

    @torch.no_grad()
    def encode_clean_latent_as_reference_slot(
        self,
        clean_latent: torch.Tensor,
        text_prompts: List[str],
        selected_layers,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Encode generated x0 with the true reference-slot t0-t2 RoPE layout."""
        if clean_latent.ndim != 5:
            raise ValueError(
                f"clean_latent must be [B,F,C,H,W], got {tuple(clean_latent.shape)}"
            )
        batch, frames, _, height, width = clean_latent.shape
        if frames != self.num_frame_per_block:
            raise ValueError(
                f"Reference re-encode requires {self.num_frame_per_block} frames, got {frames}"
            )
        selected = tuple(dict.fromkeys(int(layer) for layer in selected_layers))
        if not selected or min(selected) < 0 or max(selected) >= self.num_transformer_blocks:
            raise ValueError(
                f"Reference re-encode layers {selected} are invalid for "
                f"{self.num_transformer_blocks} transformer blocks"
            )

        layout = self._runtime_layout(height, width)
        slot_len = layout["recent_slot_len"]
        num_heads = int(_model_config_value(self.generator.model, "num_heads"))
        dim = int(_model_config_value(self.generator.model, "dim"))
        cache = [
            {
                "k": torch.zeros(
                    [batch, slot_len, num_heads, dim // num_heads],
                    device=clean_latent.device,
                    dtype=clean_latent.dtype,
                ),
                "v": torch.zeros(
                    [batch, slot_len, num_heads, dim // num_heads],
                    device=clean_latent.device,
                    dtype=clean_latent.dtype,
                ),
            }
            for _ in range(self.num_transformer_blocks)
        ]
        zeros = torch.zeros_like(clean_latent)
        reference_context = torch.cat(
            [clean_latent, zeros[:, :, :4], zeros], dim=2
        )
        timestep_zero = torch.zeros(
            [batch, frames],
            device=clean_latent.device,
            dtype=torch.int64,
        )
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        # The causal model uses render_latent_input is not None to distinguish
        # a pre-concatenated 36-channel context-writer input from a 16-channel
        # denoising input. Its value is intentionally ignored for kv_size < 0.
        writer_render_sentinel = torch.zeros(
            (batch, frames, 20, height, width), device=clean_latent.device,
            dtype=clean_latent.dtype,
        )
        self.generator(
            noisy_image_or_video=reference_context,
            conditional_dict=conditional_dict,
            timestep=timestep_zero,
            kv_cache=cache,
            render_latent_input=writer_render_sentinel,
            kv_size=(0, -1),
            freqs_offset=0,
        )
        result = {}
        for layer in selected:
            k = cache[layer]["k"][:, :slot_len]
            v = cache[layer]["v"][:, :slot_len]
            if k.shape != v.shape or k.shape[1] != slot_len:
                raise RuntimeError(
                    f"Reference-slot writer produced invalid layer {layer}: "
                    f"K={tuple(k.shape)} V={tuple(v.shape)}"
                )
            result[layer] = (k, v)
        return result

    def _runtime_layout(self, latent_height, latent_width):
        patch_size = tuple(_model_config_value(self.generator.model, "patch_size"))
        if len(patch_size) != 3 or patch_size[0] != 1:
            raise ValueError(f"Unsupported causal patch size: {patch_size}")
        if latent_height % patch_size[1] or latent_width % patch_size[2]:
            raise ValueError(
                f"Latent size {(latent_height, latent_width)} is not divisible by {patch_size[1:]}"
            )
        h_token = latent_height // patch_size[1]
        w_token = latent_width // patch_size[2]
        tokens_per_frame = h_token * w_token
        recent_slot_len = self.num_frame_per_block * tokens_per_frame
        layout = {
            "latent_hw": (latent_height, latent_width),
            "token_hw": (h_token, w_token),
            "tokens_per_frame": tokens_per_frame,
            "recent_slot_len": recent_slot_len,
            "kv_size_used_for_nonfirst_block": 2 * recent_slot_len,
        }
        assert tokens_per_frame == h_token * w_token
        assert recent_slot_len == self.num_frame_per_block * tokens_per_frame
        assert layout["kv_size_used_for_nonfirst_block"] == 2 * recent_slot_len
        if not self._layout_printed:
            print(f"[MapKV layout] {layout}")
            self._layout_printed = True
        return layout

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        latent_height=None,
        latent_width=None,
    ):
        """
        Initialize or reuse KV cache for the Wan model.
        Uses detach() + zero_() to safely reuse cache without gradient issues.
        Cache is allocated only once; subsequent calls only zero the existing tensors.
        """
        if latent_height is None:
            latent_height = int(getattr(self.args, "height", 480)) // 8
        if latent_width is None:
            latent_width = int(getattr(self.args, "width", 832)) // 8
        layout = self._runtime_layout(latent_height, latent_width)
        kv_cache_size = layout["kv_size_used_for_nonfirst_block"]

        if self.kv_cache1 is not None and len(self.kv_cache1) == self.num_transformer_blocks \
                and self.kv_cache1[0]["k"].shape[0] == batch_size \
                and self.kv_cache1[0]["k"].shape[1] == kv_cache_size \
                and self.kv_cache1[0]["k"].dtype == dtype \
                and self.kv_cache1[0]["k"].device == device:
            for block_cache in self.kv_cache1:
                block_cache["k"].detach_().zero_()
                block_cache["v"].detach_().zero_()
            return

        num_heads = int(_model_config_value(self.generator.model, "num_heads"))
        dim = int(_model_config_value(self.generator.model, "dim"))

        print(f"Initializing kv cache with size: {kv_cache_size}")
        self.kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            self.kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, dim // num_heads], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, dim // num_heads], dtype=dtype, device=device),
            })
