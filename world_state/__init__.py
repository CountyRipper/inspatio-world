"""Read-only WorldState Teacher v0 components."""

from .fixed_bank import FixedWorldBank
from .projector import RotationProjector
from .runtime import (
    WorldStateRuntime,
    attach_world_state_reader,
    load_world_state_reader,
    save_world_state_reader,
    world_state_trainable_parameters,
)
from .types import (
    Authority,
    CameraBatch,
    Provenance,
    WorldObservation,
    WorldReadPacket,
)
from .domains import (
    ThreeDomainMasks,
    build_exact_shared_domains,
    build_three_domains,
    erode_source_mask,
    patchify_domains,
    strict_source_mask,
)
from .runtime_v1 import (
    WorldStateRuntimeV1,
    attach_world_state_reader_v1,
    load_world_state_reader_v1,
    save_world_state_reader_v1,
    world_state_v1_trainable_parameters,
)

__all__ = [
    "Authority",
    "CameraBatch",
    "FixedWorldBank",
    "Provenance",
    "RotationProjector",
    "WorldObservation",
    "WorldReadPacket",
    "WorldStateRuntime",
    "WorldStateRuntimeV1",
    "attach_world_state_reader",
    "load_world_state_reader",
    "save_world_state_reader",
    "world_state_trainable_parameters",
    "ThreeDomainMasks",
    "attach_world_state_reader_v1",
    "build_exact_shared_domains",
    "build_three_domains",
    "erode_source_mask",
    "load_world_state_reader_v1",
    "patchify_domains",
    "save_world_state_reader_v1",
    "strict_source_mask",
    "world_state_v1_trainable_parameters",
]
