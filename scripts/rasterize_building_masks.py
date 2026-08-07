"""Rasterize building footprints onto tile masks aligned with chipped PNGs.

Edit the CONFIG block below, then run:

    python scripts/rasterize_building_masks.py

CLI flags are optional and override CONFIG when provided.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal

from building_footprint_segmentation.utils.script_config import apply_cli_overrides

gdal.UseExceptions()

TILE_NAME_RE = re.compile(r"^tile_(\d+)_(\d+)$")

# ---------------------------------------------------------------------------
# CONFIG — edit these paths / settings directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "geotiff": r"D:\sadoun\Devs\TestAreaSteyr\Tif\TestAreaSteyr.tif",
    "shapefile": r"D:\sadoun\Devs\TestAreaSteyr\Buildings\Original_DATA_2026.shp",
    # Folder that actually contains tile PNGs
    "images": str(PROJECT_ROOT / "data" / "steyr_512" / "images"),
    # Folder for label masks (used as-is)
    "labels": str(PROJECT_ROOT / "data" / "steyr_512" / "labels"),
}
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geotiff", default=None)
    parser.add_argument("--shapefile", default=None)
    parser.add_argument("--images", default=None)
    parser.add_argument("--labels", default=None)
    return parser.parse_args()


def parse_tile_offsets(stem: str) -> tuple[int, int]:
    match = TILE_NAME_RE.match(stem)
    if not match:
        raise ValueError(f"Expected tile name like tile_1024_2048.png, got: {stem}")
    return int(match.group(1)), int(match.group(2))


def rasterize_tile_mask(
    geotiff_path: Path,
    shapefile_path: Path,
    image_path: Path,
) -> np.ndarray:
    x_off, y_off = parse_tile_offsets(image_path.stem)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    source_ds = gdal.Open(str(geotiff_path))
    if source_ds is None:
        raise RuntimeError(f"Could not open GeoTIFF: {geotiff_path}")

    geo_transform = source_ds.GetGeoTransform()
    projection = source_ds.GetProjection()

    tile_transform = (
        geo_transform[0] + x_off * geo_transform[1] + y_off * geo_transform[2],
        geo_transform[1],
        geo_transform[2],
        geo_transform[3] + x_off * geo_transform[4] + y_off * geo_transform[5],
        geo_transform[4],
        geo_transform[5],
    )

    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(tile_transform)
    mask_ds.SetProjection(projection)
    mask_ds.GetRasterBand(1).Fill(0)

    gdal.Rasterize(
        mask_ds,
        str(shapefile_path),
        burnValues=[255],
        allTouched=True,
    )

    mask = mask_ds.GetRasterBand(1).ReadAsArray()
    return (mask > 0).astype(np.uint8) * 255


def main() -> None:
    settings = apply_cli_overrides(CONFIG, parse_args())

    geotiff_path = Path(settings["geotiff"])
    shapefile_path = Path(settings["shapefile"])
    images_dir = Path(settings["images"])
    labels_dir = Path(settings["labels"])

    if not geotiff_path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("tile_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No tile_*.png files in {images_dir}")

    print(f"GeoTIFF: {geotiff_path}")
    print(f"Shapefile: {shapefile_path}")
    print(f"Images: {images_dir} ({len(image_paths)} tiles)")
    print(f"Labels: {labels_dir}")

    written = 0
    with_buildings = 0

    for image_path in image_paths:
        mask = rasterize_tile_mask(geotiff_path, shapefile_path, image_path)
        label_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        label_rgb[mask > 0] = 255

        label_path = labels_dir / image_path.name
        cv2.imwrite(str(label_path), label_rgb)

        if mask.any():
            with_buildings += 1
        written += 1
        print(f"Saved: {label_path}")

    print(
        f"Done. Wrote {written} masks ({with_buildings} with buildings) "
        f"to {labels_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
