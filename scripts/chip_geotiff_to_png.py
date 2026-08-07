"""Chip a large GeoTIFF into RGB PNG tiles.

Edit the CONFIG block below, then run:

    python scripts/chip_geotiff_to_png.py

CLI flags are optional and override CONFIG when provided.
--output / CONFIG['output'] is used as-is (no forced test/images suffix).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from osgeo import gdal

from building_footprint_segmentation.utils.script_config import apply_cli_overrides

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# CONFIG — edit these paths / settings directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "input": r"D:\sadoun\Devs\TestAreaSteyr\Tif\TestAreaSteyr.tif",
    # Folder that will contain the PNG tiles (used as-is)
    "output": str(PROJECT_ROOT / "data" / "steyr_512" / "images"),
    "tile_size": 512,
    "min_size": 256,
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tile-size", dest="tile_size", type=int, default=None)
    parser.add_argument("--min-size", dest="min_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())

    input_path = Path(settings["input"])
    output_dir = Path(settings["output"])
    tile = int(settings["tile_size"])
    min_size = int(settings["min_size"])

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    ds = gdal.Open(str(input_path))
    if ds is None:
        raise RuntimeError(f"Could not open: {input_path}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    bands = ds.RasterCount
    print(f"Opened: {input_path}")
    print(f"Size: {width} x {height}, bands: {bands}")
    print(f"Output: {output_dir.resolve()}")

    if bands < 3:
        raise RuntimeError(f"Need at least 3 bands for RGB, found {bands}")

    written = 0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            win_w = min(tile, width - x)
            win_h = min(tile, height - y)
            if win_w < min_size or win_h < min_size:
                continue

            out_path = output_dir / f"tile_{x}_{y}.png"
            gdal.Translate(
                str(out_path),
                ds,
                format="PNG",
                srcWin=[x, y, win_w, win_h],
                bandList=[1, 2, 3],
            )
            written += 1
            print(f"Saved: {out_path} ({win_w}x{win_h})")

    print(f"Done. Wrote {written} tiles to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
