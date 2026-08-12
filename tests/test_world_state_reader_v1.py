import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from world_memory.latent_adapter import LatentMemoryAdapter
from world_state import CameraBatch, Provenance, RotationProjector, WorldObservation
from world_state.domains import (
    build_three_domains,
    strict_source_mask,
)
from world_state.encoder_v1 import IdentityPreservingWorldEncoder
from world_state.reader_v1 import IdentityPreservingWorldReader
from world_state.reader_v1 import PatchGatedLoRA
from world_state.runtime_v1 import (
    WorldStateRuntimeV1,
    load_world_state_reader_v1,
    save_world_state_reader_v1,
)


def yaw_c2w(degrees: float) -> torch.Tensor:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [
            [cosine, 0.0, sine, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sine, 0.0, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def generated_observation():
    frames, height, width = 3, 4, 6
    latent = torch.arange(frames * 16 * height * width, dtype=torch.float32)
    latent = latent.reshape(frames, 16, height, width) / 100.0
    K = torch.tensor([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    valid = torch.ones(frames, 1, height, width, dtype=torch.bool)
    confidence = torch.ones(frames, 1, height, width)
    return WorldObservation(
        scene_id="scene",
        world_id="A",
        observation_id="M40_A",
        provenance=int(Provenance.GENERATED),
        clean_latent=latent,
        K=K.repeat(frames, 1, 1),
        c2w_W0=torch.eye(4).repeat(frames, 1, 1),
        depth=None,
        valid=valid,
        static_confidence=confidence,
        geometry_confidence=confidence,
    )


def camera(yaw=0.0):
    K = torch.tensor([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    return CameraBatch(
        K=K.repeat(1, 3, 1, 1),
        c2w_W0=yaw_c2w(yaw).repeat(1, 3, 1, 1),
    )


class WorldStateReaderV1Test(unittest.TestCase):
    def test_three_domains_are_binary_exhaustive_and_source_has_authority(self):
        packet = RotationProjector().project((generated_observation(),), camera())
        mask4 = torch.full((1, 3, 4, 4, 6), -1.0)
        mask4[..., :3, :3] = 1.0
        mask4[..., 2, 2] = 0.5
        domains = build_three_domains(
            mask4, packet, confidence_threshold=0.35, source_collar=1
        )
        self.assertTrue(domains.source.dtype == torch.bool)
        self.assertTrue((domains.source | domains.memory | domains.unknown).all())
        self.assertFalse((domains.source & domains.memory).any())
        self.assertFalse(domains.memory[..., :3, :3].any())
        self.assertTrue((domains.source_core <= domains.source).all())

    def test_mask_values_are_not_used_as_soft_authority(self):
        mask4 = torch.tensor([1.0, 0.5, -0.5]).view(1, 1, 1, 1, 3)
        mask4 = mask4.expand(1, 1, 4, 1, 3).clone()
        actual = strict_source_mask(mask4)
        self.assertEqual(actual.flatten().tolist(), [True, True, False])

    def test_encoder_keeps_full_width_center_content_separate_from_metadata(self):
        packet = RotationProjector().project((generated_observation(),), camera())
        domains = build_three_domains(
            torch.full((1, 3, 4, 4, 6), -1.0),
            packet,
            confidence_threshold=0.35,
        )
        encoder = IdentityPreservingWorldEncoder(
            LatentMemoryAdapter(model_dim=32), selector_width=8
        )
        first = encoder(packet, domains)
        packet.confidence.mul_(0.5)
        second = encoder(packet, domains)
        self.assertEqual(first.content.shape, (1, 18, 32))
        self.assertEqual(first.selector_key.shape, (1, 18, 8))
        self.assertEqual(first.memory_patch.shape, (1, 18, 1))
        self.assertTrue(torch.equal(first.content, second.content))
        self.assertFalse(torch.equal(first.selector_key, second.selector_key))

    def test_reader_is_exactly_zero_outside_hard_memory_patch(self):
        torch.manual_seed(3)
        reader = IdentityPreservingWorldReader(16, selector_width=8)
        hidden = torch.randn(1, 6, 16)
        encoded = type("Encoded", (), {})()
        encoded.content = torch.randn(1, 6, 16)
        encoded.selector_key = torch.randn(1, 6, 8)
        encoded.memory_patch = torch.tensor(
            [[[True], [False], [True], [False], [False], [True]]]
        )
        context = reader.precompute(encoded)
        actual, gate = reader(hidden, torch.randn(1, 16), context)
        outside = ~encoded.memory_patch.expand_as(hidden)
        self.assertTrue(torch.equal(actual[outside], hidden[outside]))
        self.assertTrue(torch.count_nonzero(gate[~encoded.memory_patch]) == 0)

    def test_zero_innovation_is_an_exact_noop(self):
        torch.manual_seed(4)
        reader = IdentityPreservingWorldReader(16, selector_width=8)
        hidden = torch.randn(1, 6, 16)
        with torch.no_grad():
            reader.current_projection.weight.copy_(torch.eye(16))
        encoded = type("Encoded", (), {})()
        encoded.selector_key = torch.randn(1, 6, 8)
        encoded.memory_patch = torch.ones(1, 6, 1, dtype=torch.bool)
        encoded.content = reader.hidden_norm(hidden).detach()
        context = reader.precompute(encoded)
        actual, _ = reader(hidden, torch.randn(1, 16), context)
        self.assertTrue(torch.equal(actual, hidden))

    def test_lora_update_can_be_strictly_patch_gated(self):
        torch.manual_seed(9)
        lora = PatchGatedLoRA(16, rank=2)
        with torch.no_grad():
            lora.up.weight.normal_()
        hidden = torch.randn(1, 6, 16)
        gate = torch.tensor([[[1.0], [0.0], [0.5], [0.0], [1.0], [0.0]]])
        update = lora(hidden) * gate
        self.assertTrue(torch.count_nonzero(update[gate.expand_as(update) == 0]) == 0)
        self.assertGreater(torch.count_nonzero(update[gate.expand_as(update) > 0]), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "pipeline import requires CUDA")
    def test_source_core_is_clamped_at_every_step_and_final_x0(self):
        from pipeline.causal_inference import denoise_block

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, noisy_image_or_video, **kwargs):
                self.inputs.append(noisy_image_or_video.detach().clone())
                prediction = torch.full_like(noisy_image_or_video, 2.0)
                return prediction, prediction

        class Scheduler:
            def add_noise(self, clean, noise, timestep):
                scale = timestep.reshape(-1, 1, 1, 1).to(clean.dtype)
                return clean + noise * scale

        generator = Generator()
        clean = torch.full((1, 3, 2, 2, 2), 0.25)
        fixed_noise = torch.full_like(clean, 0.1)
        core = torch.zeros(1, 3, 1, 2, 2, dtype=torch.bool)
        core[..., 0, 0] = True
        prediction, _ = denoise_block(
            generator,
            Scheduler(),
            torch.zeros_like(clean),
            {},
            [],
            denoising_steps=torch.tensor([4, 3, 2, 1]),
            full_denoise_grad=True,
            source_clean=clean,
            source_core=core,
            fixed_source_noise=fixed_noise,
        )
        for value, timestep in zip(generator.inputs, (4, 3, 2, 1)):
            expected = clean + fixed_noise * timestep
            self.assertTrue(torch.equal(value[core.expand_as(value)], expected[core.expand_as(value)]))
        self.assertTrue(torch.equal(prediction[core.expand_as(prediction)], clean[core.expand_as(clean)]))

    def test_sidecar_round_trip_excludes_frozen_adapter(self):
        class ToyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 16
                self.blocks = nn.ModuleList([ToyBlock()])

        model = ToyModel()
        runtime = WorldStateRuntimeV1(
            model,
            LatentMemoryAdapter(model_dim=16),
            selected_layers=(0,),
            selector_width=8,
            lora_rank=2,
        )
        model.add_module("world_state_runtime", runtime)
        expected = model.blocks[0].world_reader.value_projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reader_v1.safetensors"
            save_world_state_reader_v1(model, path)
            with torch.no_grad():
                model.blocks[0].world_reader.value_projection.weight.add_(1)
            load_world_state_reader_v1(model, path)
        self.assertTrue(
            torch.equal(
                model.blocks[0].world_reader.value_projection.weight, expected
            )
        )


if __name__ == "__main__":
    unittest.main()
