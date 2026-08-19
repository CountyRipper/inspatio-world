from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mapkv_proto.cut3r.surfel_index import KVSurfel, SurfelIndex
from mapkv.latent_control import LatentBlockIntervention
from mapkv.locality_evaluation import _rotation_warp
from mapkv.surfel_index import (
    surfel_display_axis_labels,
    surfel_display_coordinates,
    write_oriented_disk_preview,
)
from mapkv.surfel_rgb_options import sample_historical_rgb
from mapkv_proto.deterministic_noise import DeterministicNoiseBundle
from mapkv_proto.kv_bank import KVBank, KVBankWriter
from mapkv_proto.memory_context import (
    ActiveLayerMemory,
    MemoryContext,
    reference_blind_gate,
    support_preserving_query_gate,
)
from mapkv.kv_bank import KVChunkBank, resolve_memory_layers
from mapkv.canonical_kv import _memory_token_gate, _warp_token_payload
from mapkv.cut3r_adapter import (
    _depth_to_world,
    _intrinsics_after_cut3r_resize_crop,
    _load_intrinsics,
    _reuse_previous_depths,
)
from mapkv.retrieval import GeometryChunkRetriever
from mapkv.reentry_memory import (
    MemoryEpisodeState,
    PerSurfaceRefreshLifecycle,
    ReentryEpisodeLifecycle,
    ReentryMemoryLifecycle,
    erode_binary_coverage,
    inward_feather_token_gate,
)
from mapkv.reentry_wre import score_view_adaptive_observations
from mapkv.report_framework import (
    ArchitectureChange,
    ArchitectureEdge,
    ArchitectureSnapshot,
    node,
    write_architecture_bundle,
)
from mapkv.slot_evaluation import _select_best_slot
from mapkv.warp_reencode import (
    WarpReencodePlan,
    build_continuous_virtual_recent_plans,
    build_rotation_target_to_source_grid,
    reference_protected_coverage,
    strong_memory_coverage,
    warp_latent,
)
from scripts.render_point_cloud import read_da3_depth
from mapkv_proto.reference_kv_bank import ReferenceKVBankWriter
from mapkv_proto.retrieval import RetrievalPlan
from mapkv.surfel_index import (
    SurfelCell,
    SurfelIndex as VoxelSurfelIndex,
)
from mapkv_proto.trajectory_builder import (
    build_control_phases,
    build_exact_c2w,
    build_source_protected_revisit_phases,
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


def test_surfel_gate_uses_geometry_without_reference_mask():
    coverage = torch.zeros(2, 3)
    coverage[0, 0] = 1.0
    context = MemoryContext(
        target_block=4,
        source_chunk=1,
        layer_payloads={},
        selected_layers=(0,),
        selected_step_indices=(0,),
        alpha=1.0,
        gate_mode="surfel",
        smooth_kernel=1,
        coverage=coverage,
    )
    # A fully reference-valid upstream mask must not suppress a pure surfel gate.
    gated = context.with_query_gate(torch.ones(1, 3, 1, 4, 6), (2, 3))
    gate = gated.query_gate.reshape(1, 3, 2, 3)
    assert torch.count_nonzero(gate) > 0
    torch.testing.assert_close(gate[:, 0], gate[:, 1])
    torch.testing.assert_close(gate[:, 1], gate[:, 2])


def test_surfel_exact_gate_tokenizes_same_mask_without_dilation():
    coverage = torch.zeros(1, 3, 4, 4)
    coverage[:, :, 1, 1] = 1.0
    context = MemoryContext(
        target_block=4,
        source_chunk=1,
        layer_payloads={},
        selected_layers=(0,),
        selected_step_indices=(0,),
        alpha=1.0,
        gate_mode="surfel_exact",
        smooth_kernel=1,
        coverage=coverage,
    )
    gated = context.with_query_gate(
        torch.ones(1, 3, 1, 4, 4), (4, 4)
    )
    gate = gated.query_gate.reshape(1, 3, 4, 4).float()
    torch.testing.assert_close(gate, coverage)
    assert int(torch.count_nonzero(gate)) == 3


def test_support_preserving_query_gate_keeps_small_historical_object():
    coverage = torch.zeros(1, 3, 8, 8)
    coverage[:, :, 3, 3] = 1.0
    gate = support_preserving_query_gate(
        coverage,
        batch=1,
        frames=3,
        token_hw=(4, 4),
        device=torch.device("cpu"),
        feather_kernel=3,
    )
    assert gate.shape == (1, 3, 4, 4)
    assert float(gate.max()) == 1.0
    assert torch.all(gate[:, :, 1, 1] == 1.0)
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


def test_strong_memory_coverage_preserves_binary_core_and_dilates():
    hard = torch.zeros(1, 3, 7, 7)
    hard[:, :, 3, 3] = 1.0
    memory = strong_memory_coverage(hard, dilation_kernel=3)
    assert set(memory.unique().tolist()) == {0.0, 1.0}
    assert int(torch.count_nonzero(memory[0, 0])) == 9
    assert torch.all(memory[:, :, 3, 3] == 1.0)


def test_surfel_rgb_is_sampled_from_real_first_seen_observation(tmp_path):
    image_path = tmp_path / "generated.png"
    Image.new("RGB", (16, 16), (17, 83, 201)).save(image_path)
    sequence_path = tmp_path / "sequence.json"
    sequence_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "chunk_id": 0,
                        "shape": [16, 16],
                        "image_path": str(image_path),
                        "camera_pose": np.eye(4).tolist(),
                        "intrinsics": [
                            [8.0, 0.0, 8.0],
                            [0.0, 8.0, 8.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cell = SurfelCell(
        voxel_key=(0, 0, 10),
        xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        confidence=2.0,
        normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        radius=0.1,
        rgb_preview=None,
        first_seen_chunk=0,
        last_seen_chunk=0,
        observing_chunks=[0],
        chunk_weights={0: 1.0},
        view_dirs={0: np.array([0.0, 0.0, -1.0], dtype=np.float32)},
        observation_weight=1.0,
    )
    colors, valid, chunks, stats = sample_historical_rgb(
        VoxelSurfelIndex(0.1, [cell]), sequence_path
    )
    assert valid.tolist() == [True]
    assert chunks.tolist() == [0]
    assert colors.tolist() == [[17, 83, 201]]
    assert stats["invented_colors"] is False


def _complete_test_architecture() -> ArchitectureSnapshot:
    roles = (
        "input",
        "generation",
        "geometry",
        "address",
        "payload",
        "context",
        "attention",
        "output",
        "evaluation",
    )
    nodes = tuple(
        node(
            id=f"node_{role}",
            label_zh=f"模块 {role}",
            label_en=f"{role} module",
            role=role,
            column=index,
            row=0,
            summary=f"complete {role} stage",
            change_type="modified" if role == "attention" else "unchanged",
            focus=role == "attention",
            files=("mapkv/example.py",),
        )
        for index, role in enumerate(roles)
    )
    return ArchitectureSnapshot(
        name="完整测试 Pipeline",
        focus_zh="测试 attention 模块",
        focus_en="test attention focus",
        nodes=nodes,
        edges=tuple(
            ArchitectureEdge(nodes[index].id, nodes[index + 1].id)
            for index in range(len(nodes) - 1)
        ),
        changes=(
            ArchitectureChange(
                component_id="node_attention",
                change_type="modified",
                before="全局作用",
                after="局部作用",
                affected_files=("mapkv/example.py",),
                rationale="验证空间局部性",
            ),
        ),
    )


def test_report_framework_rejects_incomplete_pipeline():
    snapshot = ArchitectureSnapshot(
        name="incomplete",
        focus_zh="缺失模块",
        focus_en="missing modules",
        nodes=(
            node(
                id="only_input",
                label_zh="输入",
                label_en="input",
                role="input",
                column=0,
                row=0,
                summary="input only",
                focus=True,
            ),
        ),
        edges=(),
    )
    with pytest.raises(ValueError, match="missing roles"):
        snapshot.validate()


def test_report_framework_writes_graph_state_and_change_artifacts(tmp_path):
    snapshot = _complete_test_architecture()
    paths = write_architecture_bundle(tmp_path, snapshot)
    assert all(Path(path).is_file() for path in paths.values())
    svg = (tmp_path / "assets" / "architecture_graph.svg").read_text()
    assert "本次关注：测试 attention 模块" in svg
    assert "已修改 · 本次 Focus" in svg
    state = json.loads((tmp_path / "architecture_state.json").read_text())
    assert len(state["nodes"]) == 9
    changes = json.loads(
        (tmp_path / "architecture_changes.json").read_text()
    )
    assert changes[0]["before"] == "全局作用"
    assert changes[0]["after"] == "局部作用"
    markdown = (tmp_path / "architecture.md").read_text()
    assert "完整 Pipeline" in markdown
    assert "mapkv/example.py" in markdown


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


def test_partial_overlap_trajectory_uses_distinct_b2_pose():
    phases, ramp_blocks = build_control_phases(
        30.0,
        temporal_stride=4.0,
        revisit_theta_degrees=20.0,
    )
    assert ramp_blocks == 5
    assert phase_by_name(phases, "A_to_B2").blocks == 4
    num_blocks = phases[-1].stop_block
    latent_length = num_blocks * 3
    rgb_length = rgb_length_for_latents(latent_length, 4.0)
    yaw, labels = build_yaw_samples(phases, rgb_length)
    source_chunk = plateau_middle_chunk(phase_by_name(phases, "B1_hold"))
    target_chunk = plateau_middle_chunk(phase_by_name(phases, "B2_hold"))
    source_rgb = monotonic_index(source_chunk * 3 + 1, latent_length, rgb_length)
    target_rgb = monotonic_index(target_chunk * 3 + 1, latent_length, rgb_length)
    validation = validate_exact_case(
        target_c2w=build_exact_c2w(np.eye(4), yaw),
        yaw_degrees=yaw,
        pitch_degrees=np.zeros_like(yaw),
        roll_degrees=np.zeros_like(yaw),
        source_chunk=source_chunk,
        target_chunk=target_chunk,
        source_rgb_index=source_rgb,
        target_rgb_index=target_rgb,
        phase_labels=labels,
        expected_rotation_degrees=10.0,
    )
    assert validation["valid"] is True
    assert validation["B1_B2_rotation_distance_degrees"] == pytest.approx(10.0)
    assert "same_view_rotation" not in validation["checks"]


def test_rotation_only_metric_warp_is_identity_for_equal_poses():
    image = np.linspace(0, 1, 6 * 8 * 3, dtype=np.float32).reshape(6, 8, 3)
    intrinsics = np.array(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]
    )
    warped, valid, homography = _rotation_warp(
        image, np.eye(4), np.eye(4), intrinsics
    )
    np.testing.assert_allclose(homography, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(warped, image, atol=1e-7)
    np.testing.assert_array_equal(valid, np.ones((6, 8), dtype=np.float32))


def test_rotation_grid_matches_metric_homography_and_identity_sampling():
    intrinsics = np.array(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]
    )
    source = np.eye(4)
    target = np.eye(4)
    angle = np.deg2rad(10.0)
    target[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    grid, valid, homography = build_rotation_target_to_source_grid(
        source, target, intrinsics, intrinsics, (6, 8)
    )
    image = np.zeros((6, 8, 3), dtype=np.float32)
    _, _, metric_homography = _rotation_warp(
        image, source, target, intrinsics
    )
    np.testing.assert_allclose(homography, metric_homography, atol=1e-8)
    assert valid.float().mean() < 1.0

    identity_grid, identity_valid, identity_h = (
        build_rotation_target_to_source_grid(
            source, source, intrinsics, intrinsics, (6, 8)
        )
    )
    latent = torch.arange(3 * 2 * 6 * 8, dtype=torch.float32).reshape(
        1, 3, 2, 6, 8
    )
    warped = warp_latent(latent, identity_grid.repeat(3, 1, 1, 1))
    torch.testing.assert_close(warped, latent, rtol=0, atol=1e-5)
    torch.testing.assert_close(identity_valid, torch.ones_like(identity_valid))
    np.testing.assert_allclose(identity_h, np.eye(3), atol=1e-8)


def test_virtual_recent_uses_history_only_inside_camera_coverage():
    intrinsics = np.array(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]]
    )
    grid, _, _ = build_rotation_target_to_source_grid(
        np.eye(4), np.eye(4), intrinsics, intrinsics, (6, 8)
    )
    coverage = torch.zeros(1, 3, 6, 8)
    coverage[..., :4] = 1.0
    plan = WarpReencodePlan(
        target_block=5,
        source_chunk=1,
        historical_latent=torch.ones(1, 3, 2, 6, 8),
        target_to_source_grid=grid.repeat(3, 1, 1, 1),
        coverage=coverage,
        selected_layers=(0,),
        selected_step_indices=(0,),
    )
    current = torch.zeros(1, 3, 2, 6, 8)
    virtual = plan.compose(current)
    torch.testing.assert_close(virtual[..., :4], torch.ones_like(virtual[..., :4]))
    torch.testing.assert_close(virtual[..., 4:], torch.zeros_like(virtual[..., 4:]))
    assert plan.audit["coverage_fraction"] == pytest.approx(0.5)


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

    calls.clear()
    selected = ActiveLayerMemory(
        k=torch.full((1, 1, 2, 2), -5.0),
        v=torch.full((1, 1, 2, 2), 200.0),
        alpha=1.0,
        query_gate=torch.ones(1, 2),
        source_chunk=0,
        injection_mode="selected_recent_delta",
    )
    selected_out = module(
        x,
        None,
        freqs,
        kv_cache=cache,
        kv_size=(0, 4),
        layer_memory=selected,
    )
    assert [item[0].shape[1] for item in calls] == [6, 5]
    assert not torch.equal(selected_out, baseline)


def test_canonical_writer_capture_and_recent_fallback_are_cache_safe(monkeypatch):
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
        for projection in (module.q, module.k, module.v, module.o):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    freqs = torch.ones(2, 1, 1, dtype=torch.complex128)
    writer_cache = {
        "k": torch.zeros(1, 2, 2, 2),
        "v": torch.zeros(1, 2, 2, 2),
    }
    capture = {}
    module(
        x,
        None,
        freqs,
        kv_cache=writer_cache,
        kv_size=(0, -1),
        canonical_capture=capture,
    )
    torch.testing.assert_close(capture["k_projected_pre_norm"], x)
    torch.testing.assert_close(capture["v"].flatten(2), x)

    cache = {
        "k": torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2),
        "v": torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2) + 20,
    }
    before = {key: value.clone() for key, value in cache.items()}
    calls.clear()
    memory_k = torch.full((1, 2, 2, 2), -7.0)
    memory_v = torch.full((1, 2, 2, 2), 77.0)
    memory = ActiveLayerMemory(
        k=memory_k,
        v=memory_v,
        alpha=1.0,
        query_gate=torch.ones(1, 2),
        source_chunk=0,
        injection_mode="canonical_recent_delta",
        memory_slot_gate=torch.tensor([[True, False]]),
    )
    module(x, None, freqs, kv_cache=cache, kv_size=(0, 4), layer_memory=memory)
    assert len(calls) == 2
    auxiliary_k, auxiliary_v = calls[1]
    torch.testing.assert_close(auxiliary_k[:, 2], memory_k[:, 0])
    torch.testing.assert_close(auxiliary_v[:, 2], memory_v[:, 0])
    torch.testing.assert_close(auxiliary_k[:, 3], cache["k"][:, 3])
    torch.testing.assert_close(auxiliary_v[:, 3], cache["v"][:, 3])
    for key in cache:
        torch.testing.assert_close(cache[key], before[key])


def test_canonical_token_warp_and_memory_gate_preserve_layout():
    payload = torch.arange(1 * 2 * 2 * 3 * 4, dtype=torch.float32).reshape(
        1, 12, 4
    )
    yy, xx = torch.meshgrid(
        torch.arange(2, dtype=torch.float32),
        torch.arange(3, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack(
        [2 * (xx + 0.5) / 3 - 1, 2 * (yy + 0.5) / 2 - 1], dim=-1
    ).repeat(2, 1, 1, 1)
    warped = _warp_token_payload(payload, grid, frames=2, token_hw=(2, 3))
    torch.testing.assert_close(warped, payload, rtol=0, atol=1e-5)
    coverage = torch.zeros(1, 2, 4, 6)
    coverage[:, :, 1, 1] = 1.0
    gate = _memory_token_gate(coverage, (2, 3))
    assert gate.shape == (1, 12)
    assert int(torch.count_nonzero(gate)) == 2


def test_retrieval_plan_loads_selected_token_indices(tmp_path):
    np.savez_compressed(
        tmp_path / "tokens.npz",
        token_indices=np.array([0, 7, 12], dtype=np.int64),
    )
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_chunk": 5,
                        "selected_chunks": [1],
                        "selected_token_indices_path": "tokens.npz",
                    }
                ]
            }
        )
    )
    plan = RetrievalPlan(tmp_path / "plan.json")
    torch.testing.assert_close(
        plan.load_token_indices(5), torch.tensor([0, 7, 12])
    )


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
    index.insert_frame(points, confidence, pose, 2, grid_hw=(6, 8))
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
                calibrated_confidence=1.0,
                consistent_observations=3,
                stable=True,
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
            calibrated_confidence=1.0,
            consistent_observations=3,
            stable=True,
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


def test_candidate_control_filters_noncandidate_before_zbuffer():
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
            view_dirs={chunk: np.array([0.0, 0.0, -1.0], dtype=np.float32)},
            observation_weight=1.0,
            calibrated_confidence=1.0,
            consistent_observations=3,
            stable=True,
        )

    index = VoxelSurfelIndex(0.1, [cell(2.0, 7), cell(1.0, 14)])
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
        current_chunk=20,
        top_k=1,
        candidate_chunks=[7],
    )
    assert result["selected_chunks"] == [7]
    assert result["retrieval_scope"] == "explicit_candidate_control"
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


def test_fixed_intrinsics_resize_depth_backprojection_and_previous_freeze(
    tmp_path,
):
    intrinsic_path = tmp_path / "intrinsics.txt"
    intrinsic_path.write_text(
        "[[723.32342529, 0., 416.], [0., 781.30383301, 240.], "
        "[0., 0., 1.]]"
    )
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (832, 480)).save(image_path)
    intrinsic = _load_intrinsics(intrinsic_path)
    resized = _intrinsics_after_cut3r_resize_crop(
        intrinsic, image_path, (288, 512)
    )
    assert resized[0, 0] == pytest.approx(445.1221, rel=1e-5)
    assert resized[1, 1] == pytest.approx(480.8024, rel=1e-5)
    assert resized[0, 2] == pytest.approx(256.0, abs=1e-6)
    depth = np.ones((2, 3), dtype=np.float32)
    camera, world = _depth_to_world(depth, resized, np.eye(4))
    np.testing.assert_allclose(camera, world)
    np.testing.assert_allclose(camera[..., 2], 1.0)

    scene = SimpleNamespace(
        im_depthmaps=torch.nn.Parameter(torch.zeros(3, 4))
    )
    hook = _reuse_previous_depths(
        scene,
        [
            np.ones((2, 2), dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ],
    )
    scene.im_depthmaps.sum().backward()
    assert torch.count_nonzero(scene.im_depthmaps.grad[:2]) == 0
    torch.testing.assert_close(
        scene.im_depthmaps.grad[2], torch.ones(4)
    )
    torch.testing.assert_close(
        scene.im_depthmaps.data[1].exp(), torch.full((4,), 2.0)
    )
    hook.remove()


def test_renderer_reads_pi3_uint16_depth_proxy(tmp_path):
    depth_root = tmp_path / "depth"
    depth_root.mkdir()
    (tmp_path / "metadata.txt").write_text("1.0 3.0\n")
    path = depth_root / "000000.png"
    Image.fromarray(
        np.asarray([[0, 65535]], dtype=np.uint16)
    ).save(path)
    decoded = read_da3_depth(path)
    np.testing.assert_allclose(decoded, [[1.0, 3.0]], atol=1e-5)


def test_surfel_promotes_only_after_three_consistent_observations():
    def observation(chunk):
        return SurfelCell(
            voxel_key=(0, 0, 20),
            xyz=np.array([0.0, 0.0, 2.0], dtype=np.float32),
            confidence=3.0,
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            radius=0.1,
            rgb_preview=None,
            first_seen_chunk=chunk,
            last_seen_chunk=chunk,
            observing_chunks=[chunk],
            chunk_weights={chunk: 1.0},
            view_dirs={chunk: np.array([0.0, 0.0, -1.0])},
            camera_forwards={chunk: np.array([0.0, 0.0, 1.0])},
            camera_centers={chunk: np.zeros(3)},
            source_pixels={chunk: np.array([5.0, 5.0])},
            calibrated_confidence=1.0,
            observation_weight=1.0,
        )

    index = VoxelSurfelIndex(0.05, [observation(1)])
    index.merge_observations(
        [observation(2)],
        position_threshold=0.1,
        normal_cosine=0.8,
        min_stable_observations=3,
    )
    assert index.cells[0].stable is False
    index.merge_observations(
        [observation(3)],
        position_threshold=0.1,
        normal_cosine=0.8,
        min_stable_observations=3,
    )
    assert index.cells[0].stable is True
    assert index.cells[0].consistent_observations == 3


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


def test_virtual_recent_reencode_uses_native_t3_t5_writer_layout():
    class FakeGenerator:
        def __init__(self):
            self.model = SimpleNamespace(num_heads=2, dim=4)
            self.call = None

        def __call__(self, **kwargs):
            self.call = kwargs
            context = kwargs["noisy_image_or_video"]
            assert context.shape == (1, 6, 36, 2, 2)
            assert kwargs["kv_size"] == (0, -1)
            assert kwargs["freqs_offset"] == 0
            assert kwargs["timestep"].shape == (1, 3)
            for layer, cache in enumerate(kwargs["kv_cache"]):
                cache["k"][:, :6].fill_(layer + 1)
                cache["v"][:, :6].fill_(layer + 11)
                cache["k"][:, 6:12].fill_(layer + 101)
                cache["v"][:, 6:12].fill_(layer + 111)
            return torch.zeros(1)

    fake = SimpleNamespace(
        num_frame_per_block=3,
        num_transformer_blocks=2,
        generator=FakeGenerator(),
        _runtime_layout=lambda height, width: {
            "recent_slot_len": 6,
            "kv_size_used_for_nonfirst_block": 12,
            "tokens_per_frame": 2,
        },
    )
    reference = torch.zeros(1, 3, 36, 2, 2)
    recent = torch.randn(1, 3, 16, 2, 2)
    payloads, audit = CausalInferencePipeline.encode_clean_latent_as_recent_slot(
        fake,
        reference_context=reference,
        clean_recent_latent=recent,
        conditional_dict={"prompt_embeds": torch.zeros(1, 1, 4)},
        selected_layers=(0, 1),
        render_block=torch.zeros(1, 3, 20, 2, 2),
    )
    assert tuple(payloads) == (0, 1)
    assert torch.count_nonzero(payloads[0][0] != 101) == 0
    assert torch.count_nonzero(payloads[1][1] != 112) == 0
    assert audit["rope_layout"] == "recent_slot_t3_t5"
    assert audit["runtime_cache_mutated"] is False


def test_continuous_virtual_recent_reprojects_short_term_fallback():
    recent = torch.arange(1 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(
        1, 3, 1, 2, 2
    )
    center_grid = torch.zeros(3, 2, 2, 2)
    plan = WarpReencodePlan(
        target_block=4,
        source_chunk=1,
        historical_latent=torch.zeros_like(recent),
        target_to_source_grid=center_grid,
        coverage=torch.zeros(1, 3, 2, 2),
        selected_layers=(0,),
        selected_step_indices=(0,),
        recent_target_to_source_grid=center_grid,
        recent_coverage=torch.ones(1, 3, 2, 2),
        mode="continuous_geometry_reprojected_virtual_recent",
    )
    virtual = plan.compose(recent)
    expected = warp_latent(recent, center_grid)
    torch.testing.assert_close(virtual, expected)
    assert not torch.equal(virtual, recent)
    assert plan.audit["short_term_recent_reprojected"] is True
    assert plan.audit["recent_warp_coverage_fraction"] == pytest.approx(1.0)


def test_repaired_continuous_virtual_recent_keeps_raw_short_term_fallback():
    recent = torch.randn(1, 3, 2, 4, 4)
    plan = WarpReencodePlan(
        target_block=4,
        source_chunk=1,
        historical_latent=torch.randn_like(recent),
        target_to_source_grid=torch.zeros(3, 4, 4, 2),
        coverage=torch.zeros(1, 3, 4, 4),
        selected_layers=(0,),
        selected_step_indices=(0,),
        recent_target_to_source_grid=None,
        query_gate_mode="surfel_exact",
        mode="masked_continuous_warp_reencode",
    )
    virtual = plan.compose(recent)
    torch.testing.assert_close(virtual, recent)
    torch.testing.assert_close(plan.artifacts["warped_recent"], recent.cpu())
    assert plan.audit["short_term_recent_reprojected"] is False
    assert plan.audit["warped_recent_vs_raw_recent_l1"] == 0.0

    plan.coverage[:, :, 1, 1] = 1.0
    payloads = {
        0: (
            torch.zeros(1, 3, 1, 2),
            torch.zeros(1, 3, 1, 2),
        )
    }
    context = plan.make_memory_context(payloads, {"runtime_cache_mutated": False})
    assert context.gate_mode == "surfel_exact"
    torch.testing.assert_close(context.coverage, plan.coverage)
    assert plan.audit["query_gate_uses_latent_composition_mask"] is True


def test_continuous_plan_is_visibility_driven_and_causal(tmp_path):
    latents = torch.randn(1, 12, 16, 4, 4)
    latent_path = tmp_path / "latents.pt"
    torch.save(latents, latent_path)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 16, axis=0)
    pose_path = tmp_path / "poses.npy"
    np.save(pose_path, poses)
    intrinsics = np.array(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]
    )
    intrinsic_path = tmp_path / "intrinsics.txt"
    intrinsic_path.write_text(repr(intrinsics.tolist()))
    cell = SurfelCell(
        voxel_key=(0, 0, -2),
        xyz=np.array([0.0, 0.0, -2.0], dtype=np.float32),
        confidence=3.0,
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        radius=0.5,
        rgb_preview=None,
        first_seen_chunk=0,
        last_seen_chunk=0,
        observing_chunks=[0],
        chunk_weights={0: 1.0},
        view_dirs={},
        observation_weight=1.0,
    )
    index_path = tmp_path / "surfel.npz"
    VoxelSurfelIndex(0.1, [cell]).save(index_path)
    sequence_path = tmp_path / "sequence.json"
    sequence_path.write_text(
        json.dumps(
            {
                "cut3r_predicted_pose_used_for_map": False,
                "coordinate_frame": "known_control_world",
                "frames": [
                    {
                        "chunk_id": 0,
                        "shape": [4, 4],
                        "intrinsics": intrinsics.tolist(),
                    }
                ],
            }
        )
    )
    plans, selections = build_continuous_virtual_recent_plans(
        source_latents_path=latent_path,
        source_chunk=0,
        target_pose_path=pose_path,
        intrinsics_path=intrinsic_path,
        surfel_index_path=index_path,
        surfel_sequence_path=sequence_path,
        latent_length=12,
        rgb_length=16,
        frames_per_block=3,
        latent_hw=(4, 4),
        image_hw=(4, 4),
        selected_layers=(0,),
        selected_step_indices=(0,),
        alpha=1.0,
        feather_kernel=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        min_history_gap_chunks=2,
    )
    assert tuple(plans) == (2, 3)
    assert [item["target_chunk"] for item in selections] == [2, 3]
    assert all(
        item["status"] == "scheduled_visible_support"
        for item in selections
    )
    assert all(
        plan.recent_target_to_source_grid is not None
        for plan in plans.values()
    )
    assert all(
        item["geometry_frames"][0]["future_geometry_used"] is False
        for item in selections
    )

    repaired, repaired_selections = build_continuous_virtual_recent_plans(
        source_latents_path=latent_path,
        source_chunk=0,
        target_pose_path=pose_path,
        intrinsics_path=intrinsic_path,
        surfel_index_path=index_path,
        surfel_sequence_path=sequence_path,
        latent_length=12,
        rgb_length=16,
        frames_per_block=3,
        latent_hw=(4, 4),
        image_hw=(4, 4),
        selected_layers=(0,),
        selected_step_indices=(0,),
        alpha=1.0,
        feather_kernel=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        min_history_gap_chunks=2,
        warp_short_term_recent=False,
        query_gate_mode="surfel_exact",
    )
    assert all(
        plan.recent_target_to_source_grid is None
        and plan.query_gate_mode == "surfel_exact"
        and plan.mode == "masked_continuous_warp_reencode"
        for plan in repaired.values()
    )
    assert all(
        item["short_term_recent"] == "raw_last_pred"
        and item["attention_query_gate"] == "surfel_exact"
        and item["same_mask_controls_latent_and_attention"] is True
        for item in repaired_selections
    )


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


def test_surfel_display_coordinates_flip_world_z_without_mutating_geometry():
    world = np.asarray(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32
    )
    before = world.copy()
    displayed = surfel_display_coordinates(world)
    unflipped = surfel_display_coordinates(world, display_z_flipped=False)
    np.testing.assert_allclose(
        displayed,
        np.asarray([[1.0, -3.0, 2.0], [-1.0, 3.0, -2.0]]),
    )
    np.testing.assert_allclose(
        unflipped,
        np.asarray([[1.0, 3.0, 2.0], [-1.0, -3.0, -2.0]]),
    )
    np.testing.assert_array_equal(world, before)
    assert surfel_display_axis_labels() == ("x", "-z (display)", "y")


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


def test_source_protected_trajectory_is_exact_and_block_aligned():
    phases, ramps = build_source_protected_revisit_phases(
        b1_theta_degrees=45.0,
        leave_theta_degrees=-20.0,
        b2_theta_degrees=35.0,
        temporal_stride=4.0,
        frames_per_block=3,
        requested_speed_degrees_per_frame=0.5,
    )
    assert ramps == {
        "A_to_B1": 8,
        "B1_to_Leave": 11,
        "Leave_to_B2": 10,
    }
    assert [phase.name for phase in phases] == [
        "A0_hold",
        "A_to_B1",
        "B1_hold",
        "B1_to_Leave",
        "Leave_hold",
        "Leave_to_B2",
        "B2_hold",
    ]
    assert phase_by_name(phases, "B1_hold").start_yaw == 45.0
    assert phase_by_name(phases, "Leave_hold").start_yaw == -20.0
    assert phase_by_name(phases, "B2_hold").start_yaw == 35.0
    assert plateau_middle_chunk(phase_by_name(phases, "B1_hold")) == 11
    assert plateau_middle_chunk(phase_by_name(phases, "B2_hold")) == 37


def test_surfel_reference_blind_observation_roundtrips(tmp_path):
    cells = [
        SurfelCell(
            voxel_key=(0, 0, 1),
            xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            confidence=2.0,
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            radius=0.1,
            rgb_preview=None,
            first_seen_chunk=3,
            last_seen_chunk=8,
            observing_chunks=[3, 8],
            chunk_weights={3: 1.0, 8: 1.0},
            view_dirs={},
            reference_blind_at_write={3: 0.1, 8: 0.9},
            source_pixels={
                3: np.array([5.0, 7.0], dtype=np.float32),
                8: np.array([9.0, 11.0], dtype=np.float32),
            },
            image_center_margins={3: 0.2, 8: 0.8},
            observation_weight=2.0,
        )
    ]
    path = tmp_path / "surfel_index.npz"
    VoxelSurfelIndex(0.1, cells).save(path)
    loaded = VoxelSurfelIndex.load(path)
    assert loaded.cells[0].reference_blind_at_write == pytest.approx(
        {3: 0.1, 8: 0.9}
    )
    np.testing.assert_array_equal(
        loaded.cells[0].source_pixels[8],
        np.array([9.0, 11.0], dtype=np.float32),
    )
    assert loaded.cells[0].image_center_margins == pytest.approx(
        {3: 0.2, 8: 0.8}
    )
    np.testing.assert_array_equal(
        loaded.generated_only_cell_indices(8), np.array([0], dtype=np.int32)
    )
    assert loaded.generated_only_cell_indices(3).size == 0


def test_source_protected_gate_is_zero_for_reference_valid_tokens():
    coverage = torch.ones(1, 3, 4, 4)
    mask = -torch.ones(1, 3, 1, 4, 4)
    mask[:, :, :, :, :2] = 1.0
    context = MemoryContext(
        target_block=9,
        source_chunk=2,
        layer_payloads={},
        selected_layers=(0,),
        selected_step_indices=(0,),
        alpha=1.0,
        gate_mode="surfel_source_protected",
        smooth_kernel=3,
        coverage=coverage,
        reference_protection_kernel=1,
    )
    gate = context.with_query_gate(mask, (4, 4)).query_gate.reshape(
        1, 3, 4, 4
    )
    assert torch.count_nonzero(gate[:, :, :, :2]) == 0
    assert torch.count_nonzero(gate[:, :, :, 2:]) > 0
    protected = reference_protected_coverage(mask, dilation_kernel=1)
    assert torch.all(protected[:, :, :, :2] == 1)
    assert torch.all(protected[:, :, :, 2:] == 0)


def test_reentry_lifecycle_reads_once_after_two_absent_blocks():
    lifecycle = ReentryMemoryLifecycle(absent_blocks=2)
    first = lifecycle.step(visible=True, read_support=True)
    assert first.state_after == MemoryEpisodeState.VISIBLE_RECENT
    assert first.read_long_term is False
    one_absent = lifecycle.step(visible=False, read_support=False)
    assert one_absent.state_after == MemoryEpisodeState.VISIBLE_RECENT
    assert one_absent.absence_count == 1
    absent = lifecycle.step(visible=False, read_support=False)
    assert absent.state_after == MemoryEpisodeState.ABSENT
    waiting = lifecycle.step(visible=True, read_support=False)
    assert waiting.state_after == MemoryEpisodeState.REENTERED
    assert waiting.read_long_term is False
    served = lifecycle.step(visible=True, read_support=True)
    assert served.state_after == MemoryEpisodeState.SERVED
    assert served.read_long_term is True
    handoff = lifecycle.step(visible=True, read_support=True)
    assert handoff.state_after == MemoryEpisodeState.SERVED
    assert handoff.read_long_term is False


def test_reentry_episode_continuously_reads_after_true_absence():
    lifecycle = ReentryEpisodeLifecycle(absent_blocks=2)
    assert lifecycle.step(visible=True, read_support=True).read_long_term is False
    assert lifecycle.step(visible=False, read_support=False).read_long_term is False
    absent = lifecycle.step(visible=False, read_support=False)
    assert absent.state_after == MemoryEpisodeState.ABSENT
    waiting = lifecycle.step(visible=True, read_support=False)
    assert waiting.state_after == MemoryEpisodeState.REENTRY_ACTIVE
    assert waiting.read_long_term is False
    first_read = lifecycle.step(visible=True, read_support=True)
    second_read = lifecycle.step(visible=True, read_support=True)
    assert first_read.read_long_term is True
    assert second_read.read_long_term is True
    assert second_read.episode_id == first_read.episode_id == 1


def test_per_surface_ttl_refreshes_later_entering_surfaces_independently():
    lifecycle = PerSurfaceRefreshLifecycle(
        [1, 2], absent_blocks=2, refresh_ttl_blocks=2
    )
    assert not lifecycle.step(
        visible_surface_ids=[1, 2], readable_surface_ids=[1, 2]
    ).active_surface_ids
    lifecycle.step(visible_surface_ids=[], readable_surface_ids=[])
    lifecycle.step(visible_surface_ids=[], readable_surface_ids=[])
    first = lifecycle.step(
        visible_surface_ids=[1], readable_surface_ids=[1]
    )
    assert first.newly_reentered_surface_ids == (1,)
    assert first.active_surface_ids == (1,)
    second = lifecycle.step(
        visible_surface_ids=[1, 2], readable_surface_ids=[1, 2]
    )
    assert second.newly_reentered_surface_ids == (2,)
    assert second.active_surface_ids == (1, 2)
    third = lifecycle.step(
        visible_surface_ids=[1, 2], readable_surface_ids=[1, 2]
    )
    assert third.active_surface_ids == (2,)


def test_edge_safe_masks_never_expand_outside_support():
    support = torch.zeros(1, 1, 7, 7)
    support[:, :, 1:6, 1:6] = 1
    eroded = erode_binary_coverage(support, kernel_size=3)
    expected = torch.zeros_like(support)
    expected[:, :, 2:5, 2:5] = 1
    torch.testing.assert_close(eroded, expected)
    gate = inward_feather_token_gate(
        support,
        batch=1,
        frames=1,
        token_hw=(7, 7),
        device=torch.device("cpu"),
        feather_kernel=3,
    )
    assert torch.count_nonzero(gate * (1.0 - support)) == 0
    assert torch.all(gate[:, :, 2:5, 2:5] == 1)
    assert torch.all(gate[:, :, 1:6, 1:6] > 0)


def test_rgb_warp_border_padding_avoids_black_invalid_samples():
    image = torch.tensor(
        [[[[[0.25, 0.5], [0.75, 1.0]]]]], dtype=torch.float32
    ).expand(1, 3, 1, 2, 2)
    outside = torch.full((3, 2, 2, 2), 2.0)
    zero = warp_latent(image, outside, padding_mode="zeros")
    border = warp_latent(image, outside, padding_mode="border")
    assert torch.count_nonzero(zero) == 0
    torch.testing.assert_close(border, torch.ones_like(border))


def test_edge_safe_source_protected_gate_is_inward_and_source_zero():
    coverage = torch.zeros(1, 3, 6, 6)
    coverage[:, :, 1:5, 1:5] = 1
    mask = -torch.ones(1, 3, 1, 6, 6)
    mask[:, :, :, :, :2] = 1
    context = MemoryContext(
        target_block=9,
        source_chunk=2,
        layer_payloads={},
        selected_layers=(0,),
        selected_step_indices=(0,),
        alpha=1.0,
        gate_mode="surfel_edge_safe_source_protected",
        smooth_kernel=3,
        coverage=coverage,
        reference_protection_kernel=1,
    )
    gate = context.with_query_gate(mask, (6, 6)).query_gate.reshape(
        1, 3, 6, 6
    )
    assert torch.count_nonzero(gate[:, :, :, :2]) == 0
    assert torch.count_nonzero(gate * (1.0 - coverage)) == 0
    assert torch.all(gate[:, :, 2:4, 2:4] == 1)


def test_view_adaptive_score_prefers_camera_aligned_pure_rotation():
    cells = [
        SimpleNamespace(
            xyz=np.array([0.0, 0.0, 1.0]),
            view_dirs={chunk: np.array([0.0, 0.0, -1.0])},
            chunk_weights={chunk: 1.0},
            image_center_margins={chunk: 1.0},
        )
        for chunk in (0, 1)
    ]

    class FakeIndex:
        def __init__(self):
            self.cells = cells

        def generated_only_cell_indices(self, chunk, **_):
            return np.array([int(chunk)], dtype=np.int32)

        def visible_cells(self, *_, eligible_indices, **__):
            index = int(eligible_indices[0])
            return {
                "pixels": np.array([[0, 0]], dtype=np.int32),
                "indices": np.array([index], dtype=np.int32),
            }

    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    angle = np.deg2rad(20.0)
    poses[0, :3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    ranking = score_view_adaptive_observations(
        surfel_index=FakeIndex(),
        candidate_chunks=(0, 1),
        surface_group_indices=np.array([0, 1], dtype=np.int32),
        target_chunk=3,
        query_pose=np.eye(4),
        intrinsics=np.eye(3),
        source_image_hw=(1, 1),
        target_hw=(1, 1),
        poses=poses,
        latent_length=4,
        rgb_length=4,
        frames_per_block=1,
        reference_blind_threshold=0.75,
    )
    assert [item["chunk_id"] for item in ranking] == [1, 0]
    assert ranking[0]["camera_orientation_alignment"] == pytest.approx(1.0)
    assert ranking[1]["camera_orientation_alignment"] < 0.2


def test_same_surface_view_selection_rejects_unrelated_pose_match():
    cells = [
        SimpleNamespace(
            xyz=np.array([0.0, 0.0, 1.0]),
            view_dirs={0: np.array([0.0, 0.0, -1.0])},
            chunk_weights={0: 1.0},
            image_center_margins={0: 1.0},
        ),
        SimpleNamespace(
            xyz=np.array([0.0, 0.0, 1.0]),
            view_dirs={1: np.array([0.0, 0.0, -1.0])},
            chunk_weights={1: 1.0},
            image_center_margins={1: 1.0},
        ),
    ]

    class FakeIndex:
        def __init__(self):
            self.cells = cells

        def generated_only_cell_indices(self, chunk, **_):
            return np.array([int(chunk)], dtype=np.int32)

        def visible_cells(self, *_, eligible_indices, **__):
            index = int(eligible_indices[0])
            return {
                "pixels": np.array([[0, 0]], dtype=np.int32),
                "indices": np.array([index], dtype=np.int32),
            }

    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    angle = np.deg2rad(20.0)
    poses[0, :3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    ranking = score_view_adaptive_observations(
        surfel_index=FakeIndex(),
        candidate_chunks=(0, 1),
        surface_group_indices=np.array([0], dtype=np.int32),
        target_chunk=3,
        query_pose=np.eye(4),
        intrinsics=np.eye(3),
        source_image_hw=(1, 1),
        target_hw=(1, 1),
        poses=poses,
        latent_length=4,
        rgb_length=4,
        frames_per_block=1,
        reference_blind_threshold=0.75,
        same_surface_only=True,
    )
    assert [item["chunk_id"] for item in ranking] == [0]
    assert ranking[0]["shared_anchor_surfels"] == 1
