import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch
from torch import nn

from world_memory.latent_adapter import LatentMemoryAdapter
from world_state.encoder import WorldTokenEncoder
from world_state.local_reader import LocalWorldReader
from world_state.projector import RotationProjector
from world_state.runtime import (
    WorldStateRuntime,
    load_world_state_reader,
    save_world_state_reader,
    world_state_trainable_parameters,
)
from world_state.types import (
    CameraBatch,
    EncodedWorldTokens,
    Provenance,
    WorldObservation,
)
from training.world_teacher.dataset import ownership_masks


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


def observation(provenance=Provenance.GENERATED, *, valid=None):
    frames, height, width = 3, 4, 6
    latent = torch.arange(frames * 16 * height * width, dtype=torch.float32)
    latent = latent.reshape(frames, 16, height, width) / 100.0
    K = torch.tensor([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    if valid is None:
        valid = torch.ones(frames, 1, height, width, dtype=torch.bool)
    confidence = torch.ones(frames, 1, height, width)
    return WorldObservation(
        scene_id="scene",
        world_id="world",
        observation_id=f"obs-{int(provenance)}",
        provenance=int(provenance),
        clean_latent=latent,
        K=K.repeat(frames, 1, 1),
        c2w_W0=torch.eye(4).repeat(frames, 1, 1),
        depth=None,
        valid=valid,
        static_confidence=confidence,
        geometry_confidence=confidence,
    )


def camera(yaw: float = 0.0):
    K = torch.tensor([[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    return CameraBatch(
        K=K.repeat(1, 3, 1, 1),
        c2w_W0=yaw_c2w(yaw).repeat(1, 3, 1, 1),
    )


class WorldStateReaderTest(unittest.TestCase):
    def test_observation_is_frozen(self):
        value = observation()
        with self.assertRaises(FrozenInstanceError):
            value.world_id = "changed"

    def test_exact_identity_fast_path_preserves_latent(self):
        value = observation()
        packet = RotationProjector().project((value,), camera())
        self.assertTrue(packet.valid.all())
        self.assertTrue(torch.equal(packet.candidate_20ch[0, 0, :, 4:], value.clean_latent))
        self.assertTrue(torch.count_nonzero(packet.subpixel_offset) == 0)

    def test_rotation_visibility_is_patch_local_not_global_switch(self):
        packet = RotationProjector().project((observation(),), camera(40.0))
        valid_count = int(packet.valid.sum())
        self.assertGreater(valid_count, 0)
        self.assertLess(valid_count, packet.valid.numel())

    def test_source_authority_invalidates_only_conflicting_generated_pixels(self):
        source_valid = torch.zeros(3, 1, 4, 6, dtype=torch.bool)
        source_valid[..., :2, :3] = True
        source = observation(Provenance.SOURCE, valid=source_valid)
        generated = observation(Provenance.GENERATED)
        packet = RotationProjector().project((source, generated), camera())
        self.assertTrue(packet.valid[:, 0, ..., :2, :3].all())
        self.assertFalse(packet.valid[:, 1, ..., :2, :3].any())
        self.assertTrue(packet.valid[:, 1, ..., 2:, 3:].all())

    def test_low_static_confidence_generated_pixels_are_unknown_for_loss(self):
        value = observation()
        packet = RotationProjector().project((value,), camera())
        packet.confidence[..., :2, :3] = 0.1
        _, generated, unknown = ownership_masks(
            packet, generated_static_threshold=0.35
        )
        self.assertFalse(generated[..., :2, :3].any())
        self.assertTrue(unknown[..., :2, :3].all())
        self.assertTrue(generated[..., 2:, 3:].all())

    def test_encoder_builds_two_by_nine_plus_null_local_candidates(self):
        packet = RotationProjector().project(
            (observation(Provenance.SOURCE), observation()), camera()
        )
        encoder = WorldTokenEncoder(
            LatentMemoryAdapter(model_dim=32), world_width=16
        )
        encoded = encoder(packet)
        self.assertEqual(encoded.tokens.shape, (1, 18, 19, 16))
        self.assertEqual(encoded.valid.shape, (1, 18, 19))
        self.assertTrue(encoded.valid[..., -1].all())
        self.assertTrue(encoded.is_null[-1])

    def test_null_value_makes_all_invalid_memory_an_exact_noop(self):
        torch.manual_seed(5)
        reader = LocalWorldReader(24, world_width=16, heads=4)
        self.assertIsNone(reader.output.bias)
        tokens = torch.randn(1, 6, 3, 16)
        valid = torch.zeros(1, 6, 3, dtype=torch.bool)
        valid[..., -1] = True
        encoded = EncodedWorldTokens(
            tokens=tokens,
            valid=valid,
            attention_bias=torch.zeros(1, 6, 3),
            is_null=torch.tensor([False, False, True]),
        )
        context = reader.precompute(encoded)
        hidden = torch.randn(1, 6, 24)
        actual = reader(hidden, torch.randn(1, 24), context)
        self.assertTrue(torch.equal(actual, hidden))

    def test_runtime_structurally_bypasses_an_empty_packet(self):
        class ToyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 24
                self.blocks = nn.ModuleList([ToyBlock()])

        model = ToyModel()
        runtime = WorldStateRuntime(
            model,
            LatentMemoryAdapter(model_dim=24),
            selected_layers=(0,),
            world_width=16,
            heads=4,
        )
        packet = RotationProjector().project((observation(),), camera())
        packet.valid.zero_()
        self.assertIsNone(runtime.precompute(packet))

        visible = RotationProjector().project((observation(),), camera())
        context = runtime.precompute(visible)
        self.assertIsNotNone(context)
        self.assertEqual(tuple(context.layers), (0,))

    def test_teacher_sidecar_round_trip_excludes_frozen_content_adapter(self):
        class ToyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 24
                self.blocks = nn.ModuleList([ToyBlock()])

        model = ToyModel()
        runtime = WorldStateRuntime(
            model,
            LatentMemoryAdapter(model_dim=24),
            selected_layers=(0,),
            world_width=16,
            heads=4,
            lora_rank=2,
        )
        model.add_module("world_state_runtime", runtime)
        trainable = world_state_trainable_parameters(model, include_lora=True)
        self.assertTrue(trainable)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in runtime.encoder.content_adapter.parameters()
            )
        )
        expected = model.blocks[0].world_reader.output.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.safetensors"
            save_world_state_reader(model, path)
            with torch.no_grad():
                model.blocks[0].world_reader.output.weight.add_(1)
            load_world_state_reader(model, path)
        self.assertTrue(torch.equal(model.blocks[0].world_reader.output.weight, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "pipeline import requires CUDA")
    def test_formal_training_keeps_all_four_denoise_steps_in_grad_graph(self):
        from pipeline.causal_inference import denoise_block

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))
                self.grad_states = []

            def forward(self, noisy_image_or_video, **kwargs):
                self.grad_states.append(torch.is_grad_enabled())
                prediction = noisy_image_or_video * self.weight
                return prediction, prediction

        class Scheduler:
            def add_noise(self, prediction, noise, timestep):
                return prediction + noise * 0.0

        generator = Generator()
        prediction, _ = denoise_block(
            generator,
            Scheduler(),
            torch.ones(1, 1, 1, 1, 1),
            {},
            [],
            denoising_steps=torch.tensor([4, 3, 2, 1]),
            full_denoise_grad=True,
        )
        prediction.sum().backward()
        self.assertEqual(generator.grad_states, [True, True, True, True])
        self.assertIsNotNone(generator.weight.grad)


if __name__ == "__main__":
    unittest.main()
