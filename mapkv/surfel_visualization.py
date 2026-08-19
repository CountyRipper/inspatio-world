from __future__ import annotations

import argparse
from pathlib import Path

from .surfel_index import (
    DEFAULT_DISPLAY_Z_FLIPPED,
    SurfelIndex,
    write_center_preview,
    write_oriented_disk_preview,
)


def visualize_existing(
    index_path: str | Path,
    output_dir: str | Path,
    *,
    display_z_flipped: bool = DEFAULT_DISPLAY_Z_FLIPPED,
) -> dict:
    """Regenerate center/disk views without rebuilding or changing geometry."""
    index_path = Path(index_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = SurfelIndex.load(index_path)
    disk_path = output_dir / "surfel_disk_preview.png"
    write_oriented_disk_preview(
        index, disk_path, display_z_flipped=display_z_flipped
    )
    center_path = output_dir / "surfel_center_preview.png"
    write_center_preview(
        index, center_path, display_z_flipped=display_z_flipped
    )
    write_center_preview(
        index,
        output_dir / "surfel_preview.png",
        display_z_flipped=display_z_flipped,
    )
    return {
        "index": str(index_path),
        "num_cells": len(index.cells),
        "center_preview": str(center_path) if center_path.exists() else None,
        "disk_preview": str(disk_path),
        "geometry_rebuilt": False,
        "display_z_flipped": bool(display_z_flipped),
        "geometry_coordinates_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render center/disk diagnostics from an existing surfel index"
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_display_z_flip", action="store_true")
    args = parser.parse_args()
    result = visualize_existing(
        args.index,
        args.output_dir,
        display_z_flipped=not args.no_display_z_flip,
    )
    print(result)


if __name__ == "__main__":
    main()
