import numpy as np
from skimage import io, measure
import os
from PIL import Image
import cv2
from pathlib import Path

# ---------------------------------------------------------
# Your addContext() function (unchanged)
# ---------------------------------------------------------
def addContext(seg_file, single_cell_dir, output_dir, bg_size=256):
    """
    Creates a side-by-side image:
        LEFT  = processed single-cell image (256x256)
        RIGHT = plain crop from original image, centered on the cell centroid.
                Crop is up to bg_size x bg_size, clipped at image edges.
    """

    seg_file = Path(seg_file)
    single_cell_dir = Path(single_cell_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation + original image
    data = np.load(seg_file, allow_pickle=True).item()
    masks = data["masks"]
    img = io.imread(data["filename"])
    imgname = Path(data["filename"]).stem

    # Handle 3D masks
    if masks.ndim == 3:
        masks = masks[0]

    # Handle 3D images
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[0] < 4:
            img = img[0]

    props = measure.regionprops(masks)
    print(f"Generating context crops for {imgname} ({len(props)} cells)")

    H_img, W_img = img.shape[:2]
    half = bg_size // 2

    for i, prop in enumerate(props, start=1):
        # Processed single-cell image
        single_cell_path = single_cell_dir / f"{imgname}_cell_{i:04d}.png"
        if not single_cell_path.exists():
            print(f"Missing processed cell image: {single_cell_path}")
            continue

        processed_cell = Image.open(single_cell_path).convert("RGB")

        # --- centroid-based crop from original image ---
        rc, cc = prop.centroid  # (row, col) in float
        rc = int(round(rc))
        cc = int(round(cc))

        r_min = max(rc - half, 0)
        r_max = min(rc + half, H_img)
        c_min = max(cc - half, 0)
        c_max = min(cc + half, W_img)

        crop = img[r_min:r_max, c_min:c_max].copy()

        # Normalize to uint8 for display
        if crop.dtype == np.uint16:
            crop = (crop.astype(np.float32) / (crop.max() + 1e-10) * 255).astype(np.uint8)
        elif crop.dtype != np.uint8:
            crop = ((crop.astype(np.float32) - crop.min()) /
                    (crop.max() - crop.min() + 1e-10) * 255).astype(np.uint8)

        # Grayscale → RGB
        if crop.ndim == 2:
            crop_rgb = np.stack([crop] * 3, axis=-1)
        else:
            crop_rgb = crop.copy()

        context_img = Image.fromarray(crop_rgb)

        # --- composite: processed (left) + context (right) ---
        H = max(processed_cell.height, context_img.height)
        W = processed_cell.width + context_img.width

        combined = Image.new("RGB", (W, H), color=(0, 0, 0))
        combined.paste(processed_cell, (0, 0))
        combined.paste(context_img, (processed_cell.width, 0))

        out_path = output_dir / f"{imgname}_cell_{i:04d}_context.png"
        combined.save(out_path)

        if i % 10 == 0 or i == len(props):
            print(f"Saved {i}/{len(props)} context images")

    print(f"Completed context generation for {imgname}\n")


# ---------------------------------------------------------
# Batch runner that preserves folder structure
# ---------------------------------------------------------
def batch_add_context(seg_root, single_cell_root, output_root):
    seg_root = Path(seg_root)
    single_cell_root = Path(single_cell_root)
    output_root = Path(output_root)

    seg_files = list(seg_root.rglob("*_seg.npy"))
    print(f"Found {len(seg_files)} segmentation files\n")

    for seg_file in seg_files:
        print(f"Processing: {seg_file}")

        # Compute relative path inside seg_root
        rel_path = seg_file.relative_to(seg_root)

        # Example:
        # seg_root/Slide1/_seg.npy  →  Slide1/_seg.npy
        # We want to preserve "Slide1"

        slide_folder = rel_path.parent  # e.g., Slide1

        # Single-cell folder for this slide
        single_cell_dir = single_cell_root / slide_folder

        if not single_cell_dir.exists():
            print(f"❌ No single-cell folder found for: {single_cell_dir}")
            continue

        # Output folder mirrors the same structure
        output_dir = output_root / slide_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run context generation
        addContext(seg_file, single_cell_dir, output_dir)


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
seg_root = "D:/MMA_batch3_CellposeCustom/Segmented_Images"
single_cell_root = "D:/MMA_batch3_CellposeCustom/Single cells/contrast_1.0"
output_root = "D:/MMA_batch3_CellposeCustom/ContextCells"

batch_add_context(seg_root, single_cell_root, output_root)



