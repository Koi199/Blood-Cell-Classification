"""
find_mislabelled_ids.py
───────────────────────
Cross-references the relabelling export against the master label export
to find the Label Studio IDs of mislabelled cells so they can be corrected
in the main labelling system.

Relabel export:
    choice = "1"         → was correctly labelled, ignore
    choice = "Clustered" → was mislabelled as MCwRBC, needs fixing in master

Master export:
    choice = "Monocyte with RBC" → current (wrong) label
    id     = Label Studio task ID to correct

Join key:
    Both exports contain the cell filename stem (tile_x001_y001_cell_XXXX)
    and slide info. We extract these from both sides and match on them.
"""

import re
import pandas as pd
from urllib.parse import unquote

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "relabel_csv": "C:/repos/Blood-Cell-Classification/LabelledData/LabelExport_20260316_unclusteredfix.csv",
    "master_csv":  "C:/repos/Blood-Cell-Classification/LabelledData/LabelExport_20260316.csv",
    "output_csv":  "C:/repos/Blood-Cell-Classification/LabelledData/ids_to_fix.csv",
}

# ─────────────────────────────────────────────
# KEY EXTRACTION
# ─────────────────────────────────────────────
def extract_key_from_relabel(image_path):
    """
    Extract cell stem + slide key from relabel export image path.

    Input:
        /data/local-files/?d=Users%5CKyle%5CMonocyte_with_RBC%5Ctile_x001_y001_cell_0001_slide1_6.png

    Output:
        tile_x001_y001_cell_0001_slide1_6   ← full filename stem
    """
    match = re.search(r'\?d=(.+)$', image_path)
    if not match:
        return None
    decoded  = unquote(match.group(1)).replace("\\", "/")
    filename = decoded.split("/")[-1]                       # tile_x001_y001_cell_0001_slide1_6.png
    stem     = filename.rsplit(".", 1)[0]                   # tile_x001_y001_cell_0001_slide1_6
    return stem


def extract_key_from_master(image_path):
    """
    Extract cell stem + slide key from master export image path.

    Input:
        /data/local-files/?d=Users%5CKyle%5CMMA_batch1%5Ccontrast_1.0%5CSlide%201-6%5Ctile_x001_y001_cell_0001.png

    The slide folder is "Slide 1-6" → normalised to "slide1_6"
    The cell stem is "tile_x001_y001_cell_0001"
    Combined key: "tile_x001_y001_cell_0001_slide1_6"
    """
    match = re.search(r'\?d=(.+)$', image_path)
    if not match:
        return None
    decoded = unquote(match.group(1)).replace("\\", "/")
    parts   = decoded.split("/")

    # Find Slide folder and filename
    slide_part = None
    cell_stem  = None

    for i, part in enumerate(parts):
        if part.startswith("Slide "):
            # "Slide 1-6" → "slide1_6"
            slide_part = part.replace("Slide ", "slide").replace("-", "_").replace(" ", "")
        if part.endswith(".png") or part.endswith(".jpg") or part.endswith(".jpeg"):
            cell_stem = part.rsplit(".", 1)[0]  # strip extension

    if slide_part and cell_stem:
        return f"{cell_stem}_{slide_part}"  # tile_x001_y001_cell_0001_slide1_6

    return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load exports ──
    df_relabel = pd.read_csv(CONFIG["relabel_csv"])
    df_master  = pd.read_csv(CONFIG["master_csv"])

    print(f"── Cross-Reference Mislabelled IDs ──")
    print(f"  Relabel export rows: {len(df_relabel)}")
    print(f"  Master export rows:  {len(df_master)}")

    # ── Filter relabel to only the mislabelled ones ──
    df_mislabelled = df_relabel[df_relabel["choice"] == "Clustered"].copy()
    print(f"  Mislabelled (choice=Clustered): {len(df_mislabelled)}")

    if df_mislabelled.empty:
        print("  No mislabelled rows found — check that 'choice' column contains 'Clustered'")
        exit()

    # ── Extract join keys from relabel export ──
    df_mislabelled["join_key"] = df_mislabelled["image"].apply(extract_key_from_relabel)

    # ── Extract join keys from master export ──
    df_master["join_key"] = df_master["image"].apply(extract_key_from_master)

    # ── Diagnose key extraction ──
    relabel_null = df_mislabelled["join_key"].isna().sum()
    master_null  = df_master["join_key"].isna().sum()
    if relabel_null > 0:
        print(f"\n  Warning: {relabel_null} relabel rows could not extract join key")
        print("  Sample unmatched relabel paths:")
        for p in df_mislabelled[df_mislabelled["join_key"].isna()]["image"].head(3):
            print(f"    {p}")
    if master_null > 0:
        print(f"\n  Warning: {master_null} master rows could not extract join key")

    # ── Join on key ──
    df_mislabelled_clean = df_mislabelled[df_mislabelled["join_key"].notna()]
    df_master_clean      = df_master[df_master["join_key"].notna()]

    df_merged = df_mislabelled_clean.merge(
        df_master_clean[["id", "join_key", "choice", "image"]],
        on="join_key",
        how="left",
        suffixes=("_relabel", "_master")
    )

    # ── Report matches ──
    matched   = df_merged["id_master"].notna().sum()
    unmatched = df_merged["id_master"].isna().sum()
    print(f"\n  Matched to master:   {matched}")
    print(f"  Not found in master: {unmatched}")

    if unmatched > 0:
        print(f"\n  Unmatched join keys (not in master export):")
        for key in df_merged[df_merged["id_master"].isna()]["join_key"].head(5):
            print(f"    {key}")
        print(f"  → These may be from slides not yet in the master export.")
        print(f"    Re-export master from Label Studio and rerun.")

    # ── Build output ──
    df_output = df_merged[df_merged["id_master"].notna()].copy()
    df_output = df_output[[
        "id_master",        # Label Studio ID to fix
        "join_key",         # cell identifier for reference
        "image_master",     # original path in master export
        "choice_master",    # current label in master (should be "Monocyte with RBC")
        "choice_relabel",   # new correct label (should be "Clustered")
    ]].rename(columns={
        "id_master":      "label_studio_id",
        "image_master":   "master_image_path",
        "choice_master":  "current_label",
        "choice_relabel": "correct_label",
    })

    df_output["action"] = "Change label to: Clustered cell"
    df_output = df_output.sort_values("label_studio_id")

    df_output.to_csv(CONFIG["output_csv"], index=False)

    print(f"\n── Output ──")
    print(f"  {len(df_output)} IDs to fix saved to: {CONFIG['output_csv']}")
    print(f"\n  Preview:")
    print(df_output[["label_studio_id", "current_label", "correct_label"]].head(10).to_string(index=False))