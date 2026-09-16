"""
extract_cells_by_class.py

Given a predictions CSV (tab-separated, with columns including 'class' and
'parent'), find every parent tile whose rows contain a chosen class, locate
that parent's Cellpose segmentation file (<parent>_seg.npy), and run
extract_single_cells() on it. Output for each parent goes into its own
subfolder named after the parent.

Usage:
    python extract_cells_by_class.py \
        --csv predictions.csv \
        --seg-dir /path/to/seg_files \
        --output-dir /path/to/output \
        --class "UNclustered Monocyte" \
        --pad 20 --bg-size 256
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, measure
from PIL import Image


def extract_single_cells(seg_file, output_dir, pad=20, bg_size=256):
    """
    Extract single cells from segmentation with maximum image quality
    (unchanged from original implementation).
    """
    data = np.load(seg_file, allow_pickle=True).item()
    masks = data['masks']
    img = io.imread(data['filename'])

    imgname = Path(data['filename']).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing image: {imgname}")
    print(f"Image dtype: {img.dtype}, shape: {img.shape}\n")

    if masks.ndim == 3:
        masks = masks[0]

    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[0] < 4:
            img = img[0]

    props = measure.regionprops(masks)
    print(f"Found {len(props)} cells\n")

    for i, prop in enumerate(props, start=1):
        bbox = prop.bbox
        if len(bbox) == 6:
            min_z, minr, minc, max_z, maxr, maxc = bbox
        else:
            minr, minc, maxr, maxc = bbox

        minr = max(minr - pad, 0)
        minc = max(minc - pad, 0)
        maxr = min(maxr + pad, masks.shape[0])
        maxc = min(maxc + pad, masks.shape[1])

        crop_img = img[minr:maxr, minc:maxc].copy()
        crop_mask = masks[minr:maxr, minc:maxc] == prop.label

        if crop_img.dtype == np.uint16:
            crop_img = (crop_img.astype(np.float32) / crop_img.max() * 255).astype(np.uint8)
        elif crop_img.dtype != np.uint8:
            crop_img = ((crop_img.astype(np.float32) - crop_img.min()) /
                        (crop_img.max() - crop_img.min() + 1e-10) * 255).astype(np.uint8)

        if crop_img.ndim == 2:
            crop_rgb = np.stack([crop_img] * 3, axis=-1)
        else:
            crop_rgb = crop_img.copy()

        for c in range(crop_rgb.shape[-1]):
            crop_rgb[~crop_mask, c] = 0

        canvas = np.zeros((bg_size, bg_size, 3), dtype=np.uint8)

        y_offset = (bg_size - crop_rgb.shape[0]) // 2
        x_offset = (bg_size - crop_rgb.shape[1]) // 2

        y_min = max(0, y_offset)
        y_max = min(bg_size, y_offset + crop_rgb.shape[0])
        x_min = max(0, x_offset)
        x_max = min(bg_size, x_offset + crop_rgb.shape[1])

        crop_y_min = max(0, -y_offset)
        crop_y_max = crop_y_min + (y_max - y_min)
        crop_x_min = max(0, -x_offset)
        crop_x_max = crop_x_min + (x_max - x_min)

        canvas[y_min:y_max, x_min:x_max] = crop_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        pil_img = Image.fromarray(canvas, mode='RGB')
        out_path = output_dir / f"{imgname}_cell_{i:04d}.png"
        pil_img.save(out_path)

        if i % 10 == 0 or i == len(props):
            print(f"Saved {i}/{len(props)} cells")

    print(f"\nCompleted! {len(props)} cells extracted to {output_dir}")


def find_seg_file(parent, seg_dir):
    """
    Locate the Cellpose segmentation file for a given parent tile name.
    Tries an exact '<parent>_seg.npy' match first (recursively), then
    falls back to any file containing the parent name.
    """
    seg_dir = Path(seg_dir)

    exact = list(seg_dir.rglob(f"{parent}_seg.npy"))
    if exact:
        return exact[0]

    fallback = list(seg_dir.rglob(f"*{parent}*_seg.npy"))
    if fallback:
        return fallback[0]

    return None


def process_class(csv_path, seg_dir, output_base_dir, chosen_class,
                   pad=20, bg_size=256, class_col="class", parent_col="parent"):
    """
    Filter the CSV to rows matching chosen_class, find each unique parent's
    seg file, and run extract_single_cells() into output_base_dir/<parent>/.
    """
    # Auto-detect delimiter (comma, tab, etc.) rather than assuming tab-separated
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if class_col not in df.columns or parent_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{class_col}' and '{parent_col}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    matches = df[df[class_col] == chosen_class]
    if matches.empty:
        print(f"No rows found with {class_col} == '{chosen_class}'")
        return

    parents = matches[parent_col].unique()
    print(f"Found {len(parents)} parent tile(s) containing class '{chosen_class}': "
          f"{list(parents)}\n")

    output_base_dir = Path(output_base_dir)

    for parent in parents:
        seg_file = find_seg_file(parent, seg_dir)
        if seg_file is None:
            print(f"[WARN] No seg file found for parent '{parent}' in {seg_dir}, skipping.")
            continue

        parent_output_dir = output_base_dir / parent
        print(f"--- Extracting cells for parent '{parent}' using {seg_file} ---")
        extract_single_cells(seg_file, parent_output_dir, pad=pad, bg_size=bg_size)


def main():
    parser = argparse.ArgumentParser(
        description="Extract cells from seg files whose parent tile contains a chosen class."
    )
    parser.add_argument("--csv", required=True, help="Path to the predictions CSV (tab-separated).")
    parser.add_argument("--seg-dir", required=True, help="Directory containing *_seg.npy files.")
    parser.add_argument("--output-dir", required=True, help="Base directory for output subfolders.")
    parser.add_argument("--class", dest="chosen_class", required=True,
                         help="Class value to filter on, e.g. 'UNclustered Monocyte'.")
    parser.add_argument("--pad", type=int, default=20, help="Padding around each cell in pixels.")
    parser.add_argument("--bg-size", type=int, default=256, help="Output canvas size in pixels.")
    args = parser.parse_args()

    process_class(
        csv_path=args.csv,
        seg_dir=args.seg_dir,
        output_base_dir=args.output_dir,
        chosen_class=args.chosen_class,
        pad=args.pad,
        bg_size=args.bg_size,
    )


if __name__ == "__main__":
    main()