"""Controlled trajectory API retained under the v0.4 package namespace."""

from mapkv_proto.trajectory_builder import (
    build_control_phases,
    build_exact_c2w,
    build_source_protected_revisit_phases,
    build_yaw_samples,
    validate_exact_case,
)

__all__ = [
    "build_control_phases",
    "build_exact_c2w",
    "build_source_protected_revisit_phases",
    "build_yaw_samples",
    "validate_exact_case",
]
