"""World candidate token encoder initialized from the frozen direct adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_memory.latent_adapter import LatentMemoryAdapter

from .types import EncodedWorldTokens, WorldReadPacket


def _shift_grid(value: torch.Tensor, dy: int, dx: int, fill_value=0):
    """Read neighbor (y+dy,x+dx) at each query grid location."""
    output = torch.full_like(value, fill_value)
    height, width = value.shape[2:4]
    query_y0, query_y1 = max(0, -dy), min(height, height - dy)
    query_x0, query_x1 = max(0, -dx), min(width, width - dx)
    source_y0, source_y1 = query_y0 + dy, query_y1 + dy
    source_x0, source_x1 = query_x0 + dx, query_x1 + dx
    if query_y1 > query_y0 and query_x1 > query_x0:
        output[:, :, query_y0:query_y1, query_x0:query_x1] = value[
            :, :, source_y0:source_y1, source_x0:source_x1
        ]
    return output


class WorldTokenEncoder(nn.Module):
    """Encode candidates independently, then expose a local 3x3 neighborhood."""

    def __init__(
        self,
        content_adapter: LatentMemoryAdapter,
        *,
        world_width: int = 512,
        neighborhood: int = 3,
    ):
        super().__init__()
        if neighborhood != 3:
            raise ValueError("Teacher v0 uses a 3x3 patch neighborhood")
        self.model_dim = content_adapter.model_dim
        self.world_width = world_width
        self.neighborhood = neighborhood
        self.content_adapter = content_adapter.requires_grad_(False)
        self.content_norm = nn.LayerNorm(self.model_dim)
        self.content_projection = nn.Linear(self.model_dim, world_width)

        self.confidence_embedding = nn.Sequential(
            nn.Linear(1, world_width), nn.SiLU(), nn.Linear(world_width, world_width)
        )
        self.pose_embedding = nn.Sequential(
            nn.Linear(7, world_width), nn.SiLU(), nn.Linear(world_width, world_width)
        )
        self.subpixel_embedding = nn.Linear(2, world_width)
        self.authority_embedding = nn.Embedding(3, world_width)
        self.provenance_embedding = nn.Embedding(2, world_width)
        self.neighbor_embedding = nn.Parameter(torch.zeros(9, world_width))
        self.null_token = nn.Parameter(torch.zeros(world_width))

        self.confidence_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.authority_bias = nn.Embedding(3, 1)
        self.provenance_bias = nn.Embedding(2, 1)
        nn.init.normal_(self.neighbor_embedding, std=0.01)
        nn.init.normal_(self.null_token, std=0.02)
        nn.init.zeros_(self.authority_bias.weight)
        nn.init.zeros_(self.provenance_bias.weight)

    def _patch_metadata(self, packet: WorldReadPacket):
        batch, candidates, frames, _, height, width = packet.valid.shape

        def pool(value, mode="avg"):
            channels = value.shape[3]
            internal = value.reshape(batch * candidates, frames, channels, height, width)
            internal = internal.permute(0, 2, 1, 3, 4).float()
            function = F.max_pool3d if mode == "max" else F.avg_pool3d
            pooled = function(internal, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            return pooled.permute(0, 2, 3, 4, 1).reshape(
                batch, candidates, frames, height // 2, width // 2, channels
            ).permute(0, 2, 3, 4, 1, 5)

        valid = pool(packet.valid, mode="max")[..., 0] > 0
        confidence = pool(packet.confidence)[..., :1]
        authority = pool(packet.authority.float(), mode="max")[..., 0].long().clamp(0, 2)
        subpixel = pool(packet.subpixel_offset)
        return valid, confidence, authority, subpixel

    def forward(self, packet: WorldReadPacket) -> EncodedWorldTokens:
        condition = packet.candidate_20ch
        batch, candidates, frames, channels, height, width = condition.shape
        internal = condition.reshape(batch * candidates, frames, channels, height, width)
        internal = internal.permute(0, 2, 1, 3, 4).contiguous()
        content = self.content_adapter(internal)
        patch_height, patch_width = content.shape[-2:]
        content = content.permute(0, 2, 3, 4, 1).reshape(
            batch, candidates, frames, patch_height, patch_width, self.model_dim
        ).permute(0, 2, 3, 4, 1, 5)
        content = self.content_projection(self.content_norm(content))

        valid, confidence, authority, subpixel = self._patch_metadata(packet)
        provenance = packet.provenance[:, None, None, None, :].expand(
            -1, frames, patch_height, patch_width, -1
        )
        relative_pose = packet.relative_pose.permute(0, 2, 1, 3)[:, :, None, None]
        relative_pose = relative_pose.expand(-1, -1, patch_height, patch_width, -1, -1)
        view_angle = packet.view_angle.permute(0, 2, 1, 3)[:, :, None, None]
        view_angle = view_angle.expand(-1, -1, patch_height, patch_width, -1, -1)

        tokens = content
        tokens = tokens + self.confidence_embedding(confidence.to(tokens.dtype))
        tokens = tokens + self.pose_embedding(
            torch.cat((relative_pose, view_angle), dim=-1).to(tokens.dtype)
        )
        tokens = tokens + self.subpixel_embedding(subpixel.to(tokens.dtype))
        tokens = tokens + self.authority_embedding(authority)
        tokens = tokens + self.provenance_embedding(provenance)

        bias = self.confidence_bias_scale * torch.log(confidence[..., 0].clamp_min(1e-4))
        bias = bias + self.authority_bias(authority)[..., 0]
        bias = bias + self.provenance_bias(provenance)[..., 0]

        neighbor_tokens, neighbor_valid, neighbor_bias = [], [], []
        neighbor_index = 0
        for candidate in range(candidates):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    shifted_token = _shift_grid(tokens[..., candidate, :], dy, dx)
                    shifted_token = shifted_token + self.neighbor_embedding[neighbor_index]
                    neighbor_tokens.append(shifted_token)
                    neighbor_valid.append(_shift_grid(valid[..., candidate], dy, dx, False))
                    neighbor_bias.append(_shift_grid(bias[..., candidate], dy, dx))
                    neighbor_index += 1
            neighbor_index = 0

        tokens = torch.stack(neighbor_tokens, dim=4)
        valid = torch.stack(neighbor_valid, dim=4)
        bias = torch.stack(neighbor_bias, dim=4)
        sequence = frames * patch_height * patch_width
        tokens = tokens.reshape(batch, sequence, candidates * 9, self.world_width)
        valid = valid.reshape(batch, sequence, candidates * 9)
        bias = bias.reshape(batch, sequence, candidates * 9)

        null = self.null_token.view(1, 1, 1, -1).expand(batch, sequence, 1, -1)
        tokens = torch.cat((tokens, null), dim=2)
        valid = torch.cat(
            (valid, torch.ones(batch, sequence, 1, dtype=torch.bool, device=valid.device)),
            dim=2,
        )
        bias = torch.cat((bias, torch.zeros_like(bias[:, :, :1])), dim=2)
        is_null = torch.zeros(tokens.shape[2], dtype=torch.bool, device=tokens.device)
        is_null[-1] = True
        return EncodedWorldTokens(
            tokens=tokens,
            valid=valid,
            attention_bias=bias,
            is_null=is_null,
        )
