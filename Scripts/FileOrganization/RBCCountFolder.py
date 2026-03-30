"""
sort_rbc_count.py
─────────────────
Reads a Label Studio export where choice = 0, 1, 2, or 3 (RBC count)
and copies pre-sliced images into corresponding subfolders.

Source:      C:/Users/Kyle/Monocyte_with_RBC/
Destination: D:/MMA_LabelledData/Unclustered_RBCCount/
             ├── RBC_0/
             ├── RBC_1/
             ├── RBC_2/
             └── RBC_3/

Images are copied (not moved) so originals are preserved.
Filenames are kept as-is since images are already sliced and named.
"""

import os
import re
import shutil
import pandas as pd
from urllib.parse import unquote

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "label_export":  "C:/repos/Blood-Cell-Classification/LabelledData/LabelExport_20260317_CountingRBC.csv",
    "source_dir":    "C:/Users/Kyle/Monocyte_with_RBC",
    "dest_dir":      "D:/MMA_LabelledData/Unclustered_RBCCount",
}

# ─────────────────────────────────────────────
# FOLDER MAP — choice value → subfolder name
# ─────────────────────────────────────────────
FOLDER_MAP = {
    "0": "RBC_0",
    "1": "RBC_1",
    "2": "RBC_2",
    "3": "RBC_3",
    # also handle integer values in case CSV parses as int
    0:   "RBC_0",
    1:   "RBC_1",
    2:   "RBC_2",
    3:   "RBC_3",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_filename(image_path):
    """
    Extract just the filename from a Label Studio image path.

    Input:  /data/local-files/?d=Users%5CKyle%5CMonocyte_with_RBC%5Ctile_x001_y001_cell_0001_slide1_6.png
    Output: tile_x001_y001_cell_0001_slide1_6.png
    """
    match = re.search(r'\?d=(.+)$', image_path)
    if not match:
        # fallback — try splitting on last slash
        return image_path.strip().split("/")[-1].split("\\")[-1]

    decoded  = unquote(match.group(1)).replace("\\", "/")
    filename = decoded.split("/")[-1]
    return filename


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    df = pd.read_csv(CONFIG["label_export"])

    print(f"── RBC Count Sorter ──")
    print(f"  Export rows: {len(df)}")
    print(f"  Source dir:  {CONFIG['source_dir']}")
    print(f"  Dest dir:    {CONFIG['dest_dir']}")
    print()

    # Create destination subfolders
    for subfolder in FOLDER_MAP.values():
        os.makedirs(os.path.join(CONFIG["dest_dir"], subfolder), exist_ok=True)

    # ── Process each row ──
    copied    = 0
    skipped   = []
    not_found = []

    for _, row in df.iterrows():
        choice   = row["choice"]
        filename = extract_filename(str(row["image"]))

        # Validate choice
        if choice not in FOLDER_MAP:
            skipped.append((filename, choice))
            continue

        src_path = os.path.join(CONFIG["source_dir"], filename)
        dst_path = os.path.join(CONFIG["dest_dir"], FOLDER_MAP[choice], filename)

        # Skip if already copied
        if os.path.exists(dst_path):
            copied += 1
            continue

        if not os.path.exists(src_path):
            not_found.append(filename)
            continue

        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"  Error copying {filename}: {e}")

    # ── Summary ──
    print(f"── Summary ──")
    print(f"  Copied:    {copied}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Skipped (unrecognised choice): {len(skipped)}")

    if not_found:
        print(f"\n  Files not found in source dir (first 10):")
        for f in not_found[:10]:
            print(f"    {f}")
        print(f"  → Check that source_dir is correct and filenames match.")

    if skipped:
        print(f"\n  Unrecognised choice values:")
        for filename, choice in skipped[:10]:
            print(f"    {filename}: '{choice}'")

    # ── Count per folder ──
    print(f"\n── Files per RBC count folder ──")
    for subfolder in sorted(set(FOLDER_MAP.values())):
        folder_path = os.path.join(CONFIG["dest_dir"], subfolder)
        count = len([f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {subfolder}: {count} images")