"""High-bandwidth generated-memory encoder for WorldStateReader v1."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_memory.latent_adapter import LatentMemoryAdapter

from .domains import ThreeDomainMasks, generated_projection, patchify_domains
from .types import EncodedWorldContentV1, WorldReadPacket


class IdentityPreservingWorldEncoder(nn.Module):
    """Keep the frozen adapter's full 1536-D content out of metadata mixing."""

    def __init__(
        self,
        content_adapter: LatentMemoryAdapter,
        *,
        selector_width: int = 256,
        confidence_threshold: float = 0.35,
    ):
        super().__init__()
        self.model_dim = content_adapter.model_dim
        self.selector_width = int(selector_width)
        self.confidence_threshold = float(confidence_threshold)
        self.content_adapter = content_adapter.requires_grad_(False)

        # Normalization and metadata are selector-only. The value path below is
        # exactly the frozen adapter feature with no dimensional bottleneck.
        self.selector_norm = nn.LayerNorm(self.model_dim)
        self.key_projection = nn.Linear(
            self.model_dim, self.selector_width, bias=False
        )
        self.confidence_embedding = nn.Sequential(
            nn.Linear(1, self.selector_width),
            nn.SiLU(),
            nn.Linear(self.selector_width, self.selector_width),
        )
        self.pose_embedding = nn.Sequential(
            nn.Linear(7, self.selector_width),
            nn.SiLU(),
            nn.Linear(self.selector_width, self.selector_width),
        )
        self.subpixel_embedding = nn.Linear(2, self.selector_width)

    @staticmethod
    def _pool_metadata(value: torch.Tensor) -> torch.Tensor:
        # external [B,F,C,H,W] -> [B,F,H/2,W/2,C]
        internal = value.permute(0, 2, 1, 3, 4).float()
        pooled = F.avg_pool3d(
            internal, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        return pooled.permute(0, 2, 3, 4, 1)

    def forward(
        self,
        packet: WorldReadPacket,
        domains: ThreeDomainMasks,
    ) -> EncodedWorldContentV1:
        index, _ = generated_projection(
            packet, confidence_threshold=self.confidence_threshold
        )
        memory = domains.memory.to(packet.candidate_20ch.dtype)
        projected_latent = packet.candidate_20ch[:, index, :, 4:] * memory
        condition = torch.cat((memory.expand(-1, -1, 4, -1, -1), projected_latent), dim=2)
        content = self.content_adapter(
            condition.permute(0, 2, 1, 3, 4).contiguous()
        )
        batch, channels, frames, height, width = content.shape
        content = content.permute(0, 2, 3, 4, 1).reshape(
            batch, frames * height * width, channels
        )

        confidence = self._pool_metadata(packet.confidence[:, index])
        subpixel = self._pool_metadata(packet.subpixel_offset[:, index])
        pose = packet.relative_pose[:, index]
        angle = packet.view_angle[:, index]
        pose = torch.cat((pose, angle), dim=-1)[:, :, None, None]
        pose = pose.expand(-1, -1, height, width, -1)

        selector_key = self.key_projection(self.selector_norm(content))
        selector_key = selector_key + self.confidence_embedding(
            confidence.reshape(batch, -1, 1).to(selector_key.dtype)
        )
        selector_key = selector_key + self.pose_embedding(
            pose.reshape(batch, -1, 7).to(selector_key.dtype)
        )
        selector_key = selector_key + self.subpixel_embedding(
            subpixel.reshape(batch, -1, 2).to(selector_key.dtype)
        )

        _, memory_patch, _ = patchify_domains(domains)
        memory_patch = memory_patch.reshape(batch, -1, 1)
        content = content * memory_patch.to(content.dtype)
        selector_key = selector_key * memory_patch.to(selector_key.dtype)
        return EncodedWorldContentV1(
            content=content,
            selector_key=selector_key,
            memory_patch=memory_patch,
        )
