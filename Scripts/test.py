"""
Cell Size Measurement Script
=============================
Measures cell size by counting non-black pixels in grayscale images
on a black background. Outputs a sorted CSV with area and estimated volume.

Usage:
    python measure_cells.py
    python measure_cells.py --folder "D:\MMA_LabelledData\Sliced\Monocyte_without_RBC"
    python measure_cells.py --threshold 15 --no-volume

Requirements:
    pip install Pillow numpy pandas
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def count_nonblack_pixels(image_path: Path, threshold: int = 10) -> dict:
    """
    Count non-black pixels in a grayscale image.

    Args:
        image_path: Path to the image file.
        threshold:  Pixel values <= threshold are treated as black (background).
                    Increase if you see noise being counted; decrease if cell
                    edges are being clipped.  Default: 10.

    Returns:
        dict with measurement results, or None on error.
    """
    try:
        img = Image.open(image_path).convert("L")   # ensure grayscale
        arr = np.array(img, dtype=np.uint8)

        # --- measurements ---
        total_pixels  = arr.size
        cell_pixels   = int(np.sum(arr > threshold))   # non-black = cell
        bg_pixels     = total_pixels - cell_pixels

        # Bounding box of the cell (tight crop around non-black region)
        rows = np.any(arr > threshold, axis=1)
        cols = np.any(arr > threshold, axis=0)
        if rows.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_h = int(rmax - rmin + 1)
            bbox_w = int(cmax - cmin + 1)
        else:
            bbox_h = bbox_w = 0

        # Estimated radius and volume (assumes roughly spherical / circular cell)
        # Area = π r²  →  r = sqrt(area / π)
        radius_px = math.sqrt(cell_pixels / math.pi) if cell_pixels > 0 else 0.0
        # Volume of sphere: (4/3) π r³
        volume_px3 = (4 / 3) * math.pi * (radius_px ** 3)

        # Mean intensity of cell pixels (proxy for staining density)
        mean_intensity = float(arr[arr > threshold].mean()) if cell_pixels > 0 else 0.0

        return {
            "filename"      : image_path.name,
            "width_px"      : arr.shape[1],
            "height_px"     : arr.shape[0],
            "cell_area_px"  : cell_pixels,
            "bg_area_px"    : bg_pixels,
            "bbox_w_px"     : bbox_w,
            "bbox_h_px"     : bbox_h,
            "est_radius_px" : round(radius_px,  2),
            "est_volume_px3": round(volume_px3, 2),
            "mean_intensity": round(mean_intensity, 2),
        }

    except Exception as exc:
        print(f"  [WARN] Could not process {image_path.name}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Measure cell size from images on a black background.")
    parser.add_argument(
        "--folder", "-f",
        default=r"D:\MMA_LabelledData\Sliced\Monocyte_without_RBC",
        help="Path to the folder containing cell images."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int, default=10,
        help="Pixel intensity threshold below which a pixel is treated as background (0-255). Default: 10."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output CSV. Defaults to <folder>/cell_sizes.csv"
    )
    parser.add_argument(
        "--sort-by", "-s",
        choices=["cell_area_px", "est_volume_px3", "est_radius_px", "filename"],
        default="cell_area_px",
        help="Column to sort results by. Default: cell_area_px"
    )
    parser.add_argument(
        "--descending", "-d",
        action="store_true",
        help="Sort in descending order (largest first). Default: ascending."
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])

    if not image_paths:
        print(f"No image files found in: {folder}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in: {folder}")
    print(f"Using intensity threshold: {args.threshold}  (pixels <= {args.threshold} = background)\n")

    results = []
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i:>4}/{len(image_paths)}] {path.name}", end="\r")
        row = count_nonblack_pixels(path, threshold=args.threshold)
        if row:
            results.append(row)

    print(f"\nProcessed {len(results)} images successfully.")

    if not results:
        print("No results to save.")
        sys.exit(1)

    df = pd.DataFrame(results)
    df.sort_values(by=args.sort_by, ascending=not args.descending, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1   # rank starts at 1
    df.index.name = "rank"

    output_path = Path(args.output) if args.output else folder / "cell_sizes.csv"
    df.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")

    # Quick summary
    print("\n--- Summary ---")
    print(f"  Images measured : {len(df)}")
    print(f"  Area  — min: {df['cell_area_px'].min():,}  max: {df['cell_area_px'].max():,}  mean: {df['cell_area_px'].mean():,.1f} px")
    print(f"  Volume— min: {df['est_volume_px3'].min():,.1f}  max: {df['est_volume_px3'].max():,.1f}  mean: {df['est_volume_px3'].mean():,.1f} px³")
    print(f"\nTop 5 largest cells (by area):")
    print(df[["filename", "cell_area_px", "est_radius_px", "est_volume_px3"]].head(5).to_string())


if __name__ == "__main__":
    main()