from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .surfel_index import SurfelIndex, write_oriented_disk_preview


def visualize_existing(index_path: str | Path, output_dir: str | Path) -> dict:
    """Add disk visualization to an existing index without rebuilding geometry."""
    index_path = Path(index_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = SurfelIndex.load(index_path)
    disk_path = output_dir / "surfel_disk_preview.png"
    write_oriented_disk_preview(index, disk_path)
    legacy_center = output_dir / "surfel_preview.png"
    center_path = output_dir / "surfel_center_preview.png"
    if legacy_center.exists() and not center_path.exists():
        shutil.copy2(legacy_center, center_path)
    return {
        "index": str(index_path),
        "num_cells": len(index.cells),
        "center_preview": str(center_path) if center_path.exists() else None,
        "disk_preview": str(disk_path),
        "geometry_rebuilt": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render center/disk diagnostics from an existing surfel index"
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = visualize_existing(args.index, args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
