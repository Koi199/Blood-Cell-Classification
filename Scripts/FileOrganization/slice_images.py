"""
slice_images.py
───────────────
Iterates over all subfolders of the source directory, crops each image
to the left half, and saves to a mirrored subfolder structure in the
destination directory.

Used to prepare unlabelled images for active learning — run inference
on these sliced images to find uncertain cases for labelling.

Source:      D:/MMA_batch1/contrast_1.0/         (subfolders per well position)
Destination: D:/MMA_batch1/contrast_1.0_Sliced/  (mirrored structure)
"""

import os
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "source_dir":     "D:/MMA_batch1/contrast_1.0",
    "dest_dir":       "D:/MMA_batch1/contrast_1.0_Sliced",
    "img_extensions": ('.png', '.jpg', '.jpeg', '.tiff', '.bmp'),
    "dry_run":        False,  # set True to preview without writing files
}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    source_dir = Path(CONFIG["source_dir"])
    dest_dir   = Path(CONFIG["dest_dir"])
    dry_run    = CONFIG["dry_run"]

    if dry_run:
        print("── DRY RUN — no files will be written ──\n")

    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_dir}")
    print()

    total_sliced  = 0
    total_skipped = 0
    total_errors  = 0

    for subfolder in sorted(source_dir.rglob("*")):
        if not subfolder.is_dir():
            continue

        images = [
            f for f in subfolder.iterdir()
            if f.is_file() and f.suffix.lower() in CONFIG["img_extensions"]
        ]

        if not images:
            continue

        relative_path  = subfolder.relative_to(source_dir)
        dest_subfolder = dest_dir / relative_path

        sliced, skipped, errors = 0, 0, 0

        for img_path in images:
            dst_path = dest_subfolder / img_path.name

            # Skip if already processed
            if dst_path.exists():
                skipped += 1
                continue

            try:
                with Image.open(img_path) as img:
                    #width, height = img.size
                    left_half = img.crop((0, 0, 256, 256))

                    if not dry_run:
                        os.makedirs(dest_subfolder, exist_ok=True)
                        left_half.save(dst_path)

                    sliced += 1

            except Exception as e:
                print(f"  Error: {img_path.name} — {e}")
                errors += 1

        total_sliced  += sliced
        total_skipped += skipped
        total_errors  += errors

        if sliced > 0 or errors > 0:
            print(f"  [{relative_path}]  sliced={sliced}  skipped={skipped}  errors={errors}")

    print(f"\n── Summary ──")
    print(f"  Sliced:  {total_sliced}")
    print(f"  Skipped: {total_skipped} (already existed)")
    print(f"  Errors:  {total_errors}")
    print(f"\n  Output: {dest_dir}")

    if dry_run:
        print(f"\n  DRY RUN complete — set dry_run=False to write files")