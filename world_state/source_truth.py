"""Conservative SourceTruth construction for static source observations."""

import torch

from .types import Provenance, WorldObservation


def conservative_static_confidence(
    source_latent: torch.Tensor,
    *,
    quantile: float = 0.55,
) -> torch.Tensor:
    """Estimate a conservative static mask from temporal latent variation.

    This deliberately rejects high-variation regions; it never promotes a render
    validity mask to source authority.
    """
    if source_latent.ndim != 4 or source_latent.shape[1] != 16:
        raise ValueError("source_latent must have shape [F,16,H,W]")
    centered = source_latent.float() - source_latent.float().median(dim=0).values
    variation = centered.abs().mean(dim=(0, 1), keepdim=True)
    threshold = torch.quantile(variation.flatten(), quantile)
    confidence = torch.exp(-variation / threshold.clamp_min(1e-6))
    return confidence.expand(source_latent.shape[0], 1, -1, -1).contiguous()


def make_source_observation(
    *,
    scene_id: str,
    world_id: str,
    observation_id: str,
    clean_latent: torch.Tensor,
    K: torch.Tensor,
    c2w_W0: torch.Tensor,
    static_confidence: torch.Tensor,
    geometry_confidence: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    static_threshold: float = 0.65,
) -> WorldObservation:
    if geometry_confidence is None:
        geometry_confidence = torch.ones_like(static_confidence)
    valid = (static_confidence >= static_threshold) & (geometry_confidence > 0)
    return WorldObservation(
        scene_id=scene_id,
        world_id=world_id,
        observation_id=observation_id,
        provenance=int(Provenance.SOURCE),
        clean_latent=clean_latent,
        K=K,
        c2w_W0=c2w_W0,
        depth=depth,
        valid=valid,
        static_confidence=static_confidence,
        geometry_confidence=geometry_confidence,
    )
