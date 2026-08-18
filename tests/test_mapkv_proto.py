from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mapkv_proto.cut3r.surfel_index import KVSurfel, SurfelIndex
from mapkv.latent_control import LatentBlockIntervention
from mapkv.surfel_index import write_oriented_disk_preview
from mapkv_proto.deterministic_noise import DeterministicNoiseBundle
from mapkv_proto.kv_bank import KVBank, KVBankWriter
from mapkv_proto.memory_context import ActiveLayerMemory, reference_blind_gate
from mapkv.kv_bank import KVChunkBank, resolve_memory_layers
from mapkv.retrieval import GeometryChunkRetriever
from mapkv.slot_evaluation import _select_best_slot
from mapkv_proto.reference_kv_bank import ReferenceKVBankWriter
from mapkv.surfel_index import (
    SurfelCell,
    SurfelIndex as VoxelSurfelIndex,
)
from mapkv_proto.trajectory_builder import (
    build_control_phases,
    build_exact_c2w,
    build_yaw_samples,
    monotonic_index,
    phase_by_name,
    plateau_middle_chunk,
    rgb_length_for_latents,
    validate_exact_case,
)
from pipeline.causal_inference import CausalInferencePipeline, denoise_block


def test_runtime_layout_accepts_mapping_model_config():
    fake = SimpleNamespace(
        generator=SimpleNamespace(
            model=SimpleNamespace(config={"patch_size": [1, 2, 2]})
        ),
        num_frame_per_block=3,
        _layout_printed=False,
    )
    layout = CausalInferencePipeline._runtime_layout(fake, 60, 104)
    assert layout == {
        "latent_hw": (60, 104),
        "token_hw": (30, 52),
        "tokens_per_frame": 1560,
        "recent_slot_len": 4680,
        "kv_size_used_for_nonfirst_block": 9360,
    }


def test_reference_blind_gate_follows_upstream_mask_semantics():
    valid = torch.ones(1, 3, 4, 4, 6)
    invalid = -torch.ones_like(valid)
    assert torch.count_nonzero(reference_blind_gate(valid, (2, 3))) == 0
    gate = reference_blind_gate(invalid, (2, 3), smooth_kernel=1)
    torch.testing.assert_close(gate, torch.ones_like(gate))


def test_exact_control_trajectory_is_block_aligned_and_revisits_same_pose():
    phases, ramp_blocks = build_control_phases(30.0, temporal_stride=4.0)
    assert ramp_blocks == 5
    num_blocks = phases[-1].stop_block
    latent_length = num_blocks * 3
    rgb_length = rgb_length_for_latents(latent_length, 4.0)
    yaw, labels = build_yaw_samples(phases, rgb_length)
    source_chunk = plateau_middle_chunk(phase_by_name(phases, "B1_hold"))
    target_chunk = plateau_middle_chunk(phase_by_name(phases, "B2_hold"))
    source_rgb = monotonic_index(source_chunk * 3 + 1, latent_length, rgb_length)
    target_rgb = monotonic_index(target_chunk * 3 + 1, latent_length, rgb_length)
    base_c2w = np.eye(4)
    base_c2w[:3, 3] = [1.0, 2.0, 3.0]
    target_c2w = build_exact_c2w(base_c2w, yaw)
    validation = validate_exact_case(
        target_c2w=target_c2w,
        yaw_degrees=yaw,
        pitch_degrees=np.zeros_like(yaw),
        roll_degrees=np.zeros_like(yaw),
        source_chunk=source_chunk,
        target_chunk=target_chunk,
        source_rgb_index=source_rgb,
        target_rgb_index=target_rgb,
        phase_labels=labels,
    )
    assert validation["valid"] is True
    assert validation["temporal_gap_chunks"] >= 4
    assert validation["B1_B2_rotation_distance_degrees"] < 1e-6
    assert validation["B1_B2_translation_distance"] == 0.0
    assert all(0.4 <= speed <= 0.6 for speed in validation["ramp_speeds_degrees_per_rgb_frame"])


def test_noise_bundle_roundtrip_and_provider_does_not_use_global_rng(tmp_path):
    bundle = DeterministicNoiseBundle.create(
        shape=(1, 3, 1, 2, 2),
        num_blocks=1,
        num_denoising_steps=3,
        seed=17,
        device="cpu",
        dtype=torch.float32,
    )
    path = tmp_path / "noise_bundle.pt"
    bundle.save(path)
    loaded = DeterministicNoiseBundle.load(path)
    torch.testing.assert_close(loaded.initial_noise, bundle.initial_noise, rtol=0, atol=0)

    class Generator:
        def __call__(self, *, noisy_image_or_video, kv_size, **kwargs):
            if kv_size[1] < 0:
                return torch.zeros_like(noisy_image_or_video)
            return torch.zeros_like(noisy_image_or_video), noisy_image_or_video + 0.25

    class Scheduler:
        @staticmethod
        def add_noise(x, noise, timestep):
            return x + noise

    kwargs = dict(
        generator=Generator(),
        scheduler=Scheduler(),
        noisy_input=loaded.get_initial(device="cpu", dtype=torch.float32),
        conditional_dict={},
        kv_cache=[],
        denoising_steps=torch.tensor([3, 2, 1]),
        block_id=0,
        noise_provider=loaded,
    )
    state_before = torch.random.get_rng_state().clone()
    first, _ = denoise_block(**kwargs)
    state_after = torch.random.get_rng_state().clone()
    second, _ = denoise_block(**kwargs)
    assert torch.equal(state_before, state_after)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_kv_bank_captures_only_clean_recent_slot(tmp_path):
    caches = []
    for layer in range(4):
        base = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1) + 100 * layer
        caches.append({"k": base.clone(), "v": (base + 50).clone()})
    writer = KVBankWriter(
        tmp_path,
        selected_layers=(-2, -1),
        num_layers=4,
        recent_slot_len=4,
        frames_per_block=2,
        tokens_per_frame=2,
        dtype=torch.float32,
    )
    writer(block_id=1, kv_cache=caches, context_frames=torch.zeros(1, 4, 1, 1, 1))

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert sorted(metadata["chunks"]["0"]["layers"]) == ["2", "3"]
    assert metadata["chunks"]["0"]["layers"]["2"]["k_stats"]["l2_norm"] > 0
    assert not (tmp_path / "chunk_0000/layer_00.pt").exists()

    bank = KVBank(tmp_path)
    payloads = bank.materialize(
        0,
        selected_layers=(-2, -1),
        num_layers=4,
        device="cpu",
        dtype=torch.float32,
        pin_memory=False,
    )
    torch.testing.assert_close(payloads[2][0], caches[2]["k"][:, 4:8])
    torch.testing.assert_close(payloads[3][1], caches[3]["v"][:, 4:8])
    capture = KVChunkBank(tmp_path).capture_manifest()
    assert capture["capture_type"] == "clean_context"
    assert capture["chunks"][0]["layers"]["2"]["k_stats"]["mean"] == pytest.approx(
        caches[2]["k"][:, 4:8].mean().item()
    )


def test_auxiliary_attention_is_strictly_opt_in_and_cache_safe(monkeypatch):
    import wan.modules.causal_model as causal_model

    calls = []

    def fake_attention(q, k, v):
        calls.append((k.clone(), v.clone()))
        return q + v.mean(dim=1, keepdim=True)

    monkeypatch.setattr(causal_model, "attention", fake_attention)
    module = causal_model.CausalWanSelfAttention(
        dim=4, num_heads=2, qk_norm=False
    )
    with torch.no_grad():
        identity = torch.eye(4)
        for projection in (module.q, module.k, module.v, module.o):
            projection.weight.copy_(identity)
            projection.bias.zero_()

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]])
    freqs = torch.ones(2, 1, 1, dtype=torch.complex128)
    cache = {
        "k": torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2),
        "v": torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2) + 20,
    }
    cache_before = {name: tensor.clone() for name, tensor in cache.items()}

    baseline = module(x, None, freqs, kv_cache=cache, kv_size=(0, 4))
    assert len(calls) == 1

    calls.clear()
    alpha_zero = ActiveLayerMemory(
        k=torch.full((1, 2, 2, 2), -5.0),
        v=torch.full((1, 2, 2, 2), 200.0),
        alpha=0.0,
        query_gate=torch.ones(1, 2),
        source_chunk=0,
    )
    alpha_zero_out = module(
        x, None, freqs, kv_cache=cache, kv_size=(0, 4), layer_memory=alpha_zero
    )
    assert len(calls) == 1
    torch.testing.assert_close(alpha_zero_out, baseline, rtol=0, atol=0)

    calls.clear()
    active = ActiveLayerMemory(
        k=torch.full((1, 2, 2, 2), -5.0),
        v=torch.full((1, 2, 2, 2), 200.0),
        alpha=0.1,
        query_gate=torch.ones(1, 2),
        source_chunk=0,
    )
    active_out = module(
        x, None, freqs, kv_cache=cache, kv_size=(0, 4), layer_memory=active
    )
    assert len(calls) == 2
    assert not torch.equal(active_out, baseline)
    for name in cache:
        torch.testing.assert_close(cache[name], cache_before[name], rtol=0, atol=0)

    with pytest.raises(AssertionError, match="writer"):
        module(x, None, freqs, kv_cache=cache, kv_size=(0, -1), layer_memory=active)


def test_residual_memory_attention_is_separate_and_cache_safe(monkeypatch):
    import wan.modules.causal_model as causal_model

    calls = []

    def fake_attention(q, k, v):
        calls.append(k.shape[1])
        return q + v.mean(dim=1, keepdim=True)

    monkeypatch.setattr(causal_model, "attention", fake_attention)
    module = causal_model.CausalWanSelfAttention(dim=4, num_heads=2, qk_norm=False)
    with torch.no_grad():
        for projection in (module.q, module.k, module.v, module.o):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
    x = torch.ones(1, 2, 4)
    freqs = torch.ones(2, 1, 1, dtype=torch.complex128)
    cache = {
        "k": torch.zeros(1, 4, 2, 2),
        "v": torch.zeros(1, 4, 2, 2),
    }
    before = {key: value.clone() for key, value in cache.items()}
    baseline = module(x, None, freqs, kv_cache=cache, kv_size=(0, 4))
    audit = {}
    memory = ActiveLayerMemory(
        k=torch.ones(1, 2, 2, 2),
        v=torch.full((1, 2, 2, 2), 3.0),
        alpha=0.1,
        query_gate=torch.ones(1, 2),
        source_chunk=1,
        audit_record=audit,
        injection_mode="residual_memory_attention",
    )
    output = module(
        x, None, freqs, kv_cache=cache, kv_size=(0, 4), layer_memory=memory
    )
    assert calls == [6, 6, 2]
    assert not torch.equal(output, baseline)
    assert audit["injection_mode"] == "residual_memory_attention"
    for key in cache:
        torch.testing.assert_close(cache[key], before[key], rtol=0, atol=0)


def test_uniform_layers_and_voxel_surfel_retrieval_roundtrip(tmp_path):
    assert resolve_memory_layers("uniform8", 30) == (0, 4, 8, 12, 17, 21, 25, 29)
    yy, xx = np.mgrid[-1:1:6j, -1:1:8j]
    points = np.stack([xx, yy, np.full_like(xx, 2.0)], axis=-1).astype(np.float32)
    confidence = np.full(points.shape[:2], 3.0, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    index = VoxelSurfelIndex(voxel_size=0.15)
    index.insert_frame(points, confidence, pose, 1, grid_hw=(6, 8))
    index.insert_frame(points, confidence, pose, 3, grid_hw=(6, 8))
    index.insert_frame(points, confidence, pose, 3, grid_hw=(6, 8))
    path = tmp_path / "surfel.npz"
    index.save(path)
    restored = VoxelSurfelIndex.load(path)
    retriever = GeometryChunkRetriever(
        min_history_gap_chunks=2,
        use_view_alignment=False,
        use_occlusion=True,
    )
    intrinsic = np.array([[20.0, 0, 16], [0, 20.0, 12], [0, 0, 1]])
    result, diagnostics = retriever.retrieve(
        restored,
        pose,
        intrinsic,
        source_image_size=(24, 32),
        image_size=(24, 32),
        current_chunk=6,
        top_k=1,
    )
    # Simple observation voting intentionally does not partition one surfel's
    # vote by repeated chunk_weights. The tie is deterministic by chunk id.
    assert result["selected_chunks"] == [1]
    assert result["voting_mode"] == "simple_observing_chunk_vote"
    assert result["scores"]["1"] == pytest.approx(result["scores"]["3"])
    assert result["num_visible_surfels"] > 0
    assert diagnostics["coverage"].any()


def test_retrieval_without_pose_clusters_uses_single_chunk_clusters():
    index = VoxelSurfelIndex(
        0.1,
        [
            SurfelCell(
                voxel_key=(0, 0, 20),
                xyz=np.array([0.0, 0.0, 2.0], dtype=np.float32),
                confidence=3.0,
                normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                radius=0.2,
                rgb_preview=None,
                first_seen_chunk=1,
                last_seen_chunk=1,
                observing_chunks=[1],
                chunk_weights={1: 1.0},
                view_dirs={
                    1: np.array([0.0, 0.0, -1.0], dtype=np.float32)
                },
                observation_weight=1.0,
            )
        ],
    )
    retriever = GeometryChunkRetriever(
        min_history_gap_chunks=2,
        use_view_alignment=False,
        use_occlusion=True,
    )
    intrinsic = np.array(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]
    )
    result, _ = retriever.retrieve(
        index,
        np.eye(4),
        intrinsic,
        source_image_size=(11, 11),
        image_size=(11, 11),
        current_chunk=5,
        top_k=1,
        chunk_clusters=None,
    )
    assert result["selected_chunks"] == [1]
    assert result["retrieved"][0]["cluster_chunks"] == [1]


def test_current_surfel_filters_recent_geometry_before_zbuffer():
    def cell(z, chunk):
        return SurfelCell(
            voxel_key=(0, 0, int(z * 10)),
            xyz=np.array([0.0, 0.0, z], dtype=np.float32),
            confidence=3.0,
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            radius=0.4,
            rgb_preview=None,
            first_seen_chunk=chunk,
            last_seen_chunk=chunk,
            observing_chunks=[chunk],
            chunk_weights={chunk: 1.0},
            view_dirs={
                chunk: np.array([0.0, 0.0, -1.0], dtype=np.float32)
            },
            observation_weight=1.0,
        )

    # Chunk 4 is nearer at the same pixel but is immediate-recent for target 5.
    # It must be removed before z-buffering so old chunk 1 remains visible.
    index = VoxelSurfelIndex(0.1, [cell(2.0, 1), cell(1.0, 4)])
    retriever = GeometryChunkRetriever(
        min_history_gap_chunks=2,
        use_view_alignment=False,
        use_occlusion=True,
    )
    intrinsic = np.array(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]
    )
    result, diagnostics = retriever.retrieve(
        index,
        np.eye(4),
        intrinsic,
        source_image_size=(11, 11),
        image_size=(11, 11),
        current_chunk=5,
        top_k=1,
    )
    assert result["eligibility_before_zbuffer"] is True
    assert result["selected_chunks"] == [1]
    assert 4 not in result["eligible_chunks"]
    assert np.all(diagnostics["visible"]["indices"] == 0)


def test_radius_normal_fusion_crosses_voxel_boundaries():
    def observation(x, chunk):
        return SurfelCell(
            voxel_key=(0, 0, 20),
            xyz=np.array([x, 0.0, 2.0], dtype=np.float32),
            confidence=3.0,
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            radius=0.05,
            rgb_preview=None,
            first_seen_chunk=chunk,
            last_seen_chunk=chunk,
            observing_chunks=[chunk],
            chunk_weights={chunk: 1.0},
            view_dirs={
                chunk: np.array([0.0, 0.0, -1.0], dtype=np.float32)
            },
            observation_weight=1.0,
        )

    index = VoxelSurfelIndex(0.1, [observation(0.099, 1)])
    merged = index.merge_observations(
        [observation(0.101, 3)],
        position_threshold=0.05,
        normal_cosine=0.6,
    )
    assert merged["merged"] == 1
    assert merged["cross_voxel_merges"] == 1
    assert len(index.cells) == 1
    assert index.cells[0].observing_chunks == [1, 3]


def test_surfel_merge_serialization_and_causal_visibility_vote(tmp_path):
    original = KVSurfel(
        position=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        radius=0.4,
        confidence=3.0,
        observing_chunks=[1],
        created_chunk=1,
    )
    index = SurfelIndex([original])
    close_observation = KVSurfel(
        position=np.array([0.01, 0.0, 2.0], dtype=np.float32),
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        radius=0.4,
        confidence=99.0,
        observing_chunks=[3],
        created_chunk=3,
    )
    merge = index.merge(
        [close_observation], position_threshold=0.05, normal_cosine=0.6
    )
    assert merge["merged"] == 1
    assert len(index.surfels) == 1
    assert index.surfels[0].observing_chunks == [1, 3]
    assert index.surfels[0].confidence == 3.0

    index.surfels.extend(
        [
            KVSurfel(
                position=np.array([0.5, 0.0, 3.0], dtype=np.float32),
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                radius=0.1,
                confidence=0.5,
                observing_chunks=[2],
                created_chunk=2,
            ),
            KVSurfel(
                position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                radius=0.8,
                confidence=1000.0,
                observing_chunks=[4],
                created_chunk=4,
            ),
        ]
    )
    path = tmp_path / "surfels.npz"
    index.save(path, tmp_path / "surfels.ply")
    loaded = SurfelIndex.load(path)
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    plan, rendered = loaded.retrieve(
        target_chunk=5,
        c2w=np.eye(4),
        intrinsic=intrinsic,
        image_hw=(11, 11),
        oracle_chunk=1,
    )
    assert 4 not in plan["candidate_chunks"]
    assert plan["selected_chunks"] == [1]
    assert plan["oracle_hit"] is True
    assert rendered["coverage"].sum() > 0
    assert not np.any(rendered["surfel_id"] == 2), "future-created surfel leaked into target"
    chunk_one_coverage = loaded.coverage_for_chunk(
        rendered, chunk_id=1, target_chunk=5
    )
    chunk_two_coverage = loaded.coverage_for_chunk(
        rendered, chunk_id=2, target_chunk=5
    )
    assert chunk_one_coverage.sum() > 0
    assert chunk_two_coverage.sum() == 0, "occluded chunk must have empty coverage"
    with pytest.raises(ValueError, match="causally valid"):
        loaded.coverage_for_chunk(rendered, chunk_id=4, target_chunk=5)


def test_reference_and_both_slot_counterfactuals_are_exact_and_cache_safe(monkeypatch):
    import wan.modules.causal_model as causal_model

    calls = []

    def fake_attention(q, k, v):
        calls.append({"k": k.clone(), "v": v.clone()})
        return q + v.mean(dim=1, keepdim=True)

    monkeypatch.setattr(causal_model, "attention", fake_attention)
    module = causal_model.CausalWanSelfAttention(dim=4, num_heads=2, qk_norm=False)
    with torch.no_grad():
        for projection in (module.q, module.k, module.v, module.o):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()

    x = torch.ones(1, 2, 4)
    freqs = torch.ones(2, 1, 1, dtype=torch.complex128)
    cache = {
        "k": torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2),
        "v": torch.cat(
            [
                torch.full((1, 2, 2, 2), 10.0),
                torch.full((1, 2, 2, 2), 20.0),
            ],
            dim=1,
        ),
    }
    before = {name: value.clone() for name, value in cache.items()}
    historical_recent = torch.full((1, 2, 2, 2), 100.0)
    historical_ref = torch.full((1, 2, 2, 2), 200.0)

    ref_memory = ActiveLayerMemory(
        k=historical_recent,
        v=historical_recent,
        ref_k=historical_ref,
        ref_v=historical_ref,
        alpha=1.0,
        query_gate=torch.ones(1, 2),
        source_chunk=1,
        injection_mode="replace_ref_delta",
    )
    module(x, None, freqs, kv_cache=cache, kv_size=(0, 4), layer_memory=ref_memory)
    assert len(calls) == 2
    torch.testing.assert_close(calls[1]["v"][:, :2], historical_ref)
    torch.testing.assert_close(calls[1]["v"][:, 2:4], cache["v"][:, 2:4])

    calls.clear()
    both_memory = ActiveLayerMemory(
        k=historical_recent,
        v=historical_recent,
        ref_k=historical_ref,
        ref_v=historical_ref,
        alpha=1.0,
        query_gate=torch.ones(1, 2),
        source_chunk=1,
        injection_mode="replace_both_delta",
    )
    module(x, None, freqs, kv_cache=cache, kv_size=(0, 4), layer_memory=both_memory)
    assert len(calls) == 2
    torch.testing.assert_close(calls[1]["v"][:, :2], historical_ref)
    torch.testing.assert_close(calls[1]["v"][:, 2:4], historical_recent)
    for name in cache:
        torch.testing.assert_close(cache[name], before[name], rtol=0, atol=0)


def test_reference_slot_reencode_uses_t0_t2_writer_layout():
    class FakeTextEncoder:
        def __call__(self, text_prompts):
            assert text_prompts == ["static scene"]
            return {"prompt_embeds": torch.zeros(1, 1, 4)}

    class FakeGenerator:
        def __init__(self):
            self.model = SimpleNamespace(num_heads=2, dim=4)
            self.call = None

        def __call__(self, **kwargs):
            self.call = kwargs
            context = kwargs["noisy_image_or_video"]
            assert context.shape == (1, 3, 36, 2, 2)
            assert kwargs["kv_size"] == (0, -1)
            assert kwargs["freqs_offset"] == 0
            assert kwargs["render_latent_input"].shape == (1, 3, 20, 2, 2)
            assert torch.count_nonzero(kwargs["render_latent_input"]) == 0
            assert kwargs["timestep"].shape == (1, 3)
            assert torch.count_nonzero(kwargs["timestep"]) == 0
            for layer, cache in enumerate(kwargs["kv_cache"]):
                cache["k"].fill_(layer + 1)
                cache["v"].fill_(layer + 11)
            return torch.zeros(1)

    fake = SimpleNamespace(
        num_frame_per_block=3,
        num_transformer_blocks=2,
        generator=FakeGenerator(),
        text_encoder=FakeTextEncoder(),
        _runtime_layout=lambda height, width: {
            "recent_slot_len": 6,
            "tokens_per_frame": 2,
        },
    )
    clean = torch.randn(1, 3, 16, 2, 2)
    payloads = CausalInferencePipeline.encode_clean_latent_as_reference_slot(
        fake, clean, ["static scene"], (0, 1)
    )
    assert tuple(payloads) == (0, 1)
    assert torch.count_nonzero(payloads[0][0] != 1) == 0
    assert torch.count_nonzero(payloads[1][1] != 12) == 0


def test_reference_kv_bank_records_rope_layout_and_roundtrips(tmp_path):
    writer = ReferenceKVBankWriter(
        tmp_path,
        selected_layers=(0, 1),
        num_layers=2,
        slot_len=3,
        frames_per_block=3,
        tokens_per_frame=1,
        dtype=torch.float32,
    )
    payloads = {
        layer: (
            torch.full((1, 3, 1, 2), float(layer + 1)),
            torch.full((1, 3, 1, 2), float(layer + 10)),
        )
        for layer in (0, 1)
    }
    writer.write_chunk(chunk_id=4, layer_payloads=payloads)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["slot_kind"] == "reference"
    assert metadata["rope_layout"] == "reference_slot_t0_t2"
    assert metadata["chunks"]["4"]["capture_type"] == "clean_reference_reencode"
    loaded = KVBank(tmp_path).materialize(
        4,
        selected_layers=(0, 1),
        num_layers=2,
        device="cpu",
        dtype=torch.float32,
        pin_memory=False,
    )
    torch.testing.assert_close(loaded[1][0], payloads[1][0])


def test_direct_latent_control_has_exact_hard_and_soft_limits():
    predicted = torch.zeros(1, 3, 2, 2, 2)
    memory = torch.ones_like(predicted)
    hard = LatentBlockIntervention(
        target_block=7,
        source_chunk=2,
        clean_latent=memory,
        strength=1.0,
    )
    soft = LatentBlockIntervention(
        target_block=7,
        source_chunk=2,
        clean_latent=memory,
        strength=0.25,
    )
    torch.testing.assert_close(hard.apply(predicted), memory, rtol=0, atol=0)
    torch.testing.assert_close(
        soft.apply(predicted), torch.full_like(predicted, 0.25), rtol=0, atol=0
    )
    assert hard.audit["mode"] == "direct_clean_x0_block_override"
    assert hard.audit["output_delta_l1"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape"):
        hard.apply(torch.zeros(1, 2, 2, 2, 2))


def test_oriented_disk_surfel_preview_is_visualization_only(tmp_path):
    cell = SurfelCell(
        voxel_key=(0, 0, 2),
        xyz=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        confidence=3.0,
        normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        radius=0.2,
        rgb_preview=None,
        first_seen_chunk=1,
        last_seen_chunk=3,
        observing_chunks=[1, 3],
        chunk_weights={1: 1.0, 3: 2.0},
        view_dirs={},
        observation_weight=3.0,
    )
    index = VoxelSurfelIndex(0.1, [cell])
    before = index.cells[0].xyz.copy()
    output = tmp_path / "disk.png"
    write_oriented_disk_preview(index, output, max_disks=10)
    assert output.exists() and output.stat().st_size > 0
    np.testing.assert_array_equal(index.cells[0].xyz, before)


def test_slot_selection_does_not_call_both_required_for_a_near_tie():
    groups = {
        "recent": {
            "source_specificity_error_margin": 0.320,
            "correct_improvement_vs_baseline": 0.140,
            "correct_intervention_l1": 0.160,
            "correct_b1_b2_generated_region_l1": 0.020,
        },
        "reference": {
            "source_specificity_error_margin": 0.010,
            "correct_improvement_vs_baseline": 0.010,
            "correct_intervention_l1": 0.066,
            "correct_b1_b2_generated_region_l1": 0.151,
        },
        "both": {
            "source_specificity_error_margin": 0.325,
            "correct_improvement_vs_baseline": 0.137,
            "correct_intervention_l1": 0.160,
            "correct_b1_b2_generated_region_l1": 0.023,
        },
    }
    ranking = [
        (0.462, 0.160, "both"),
        (0.460, 0.160, "recent"),
        (0.020, 0.066, "reference"),
    ]
    assert _select_best_slot(groups, ranking) == "recent"
