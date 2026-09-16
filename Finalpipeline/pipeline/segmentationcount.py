"""
rbc_segmentation_pipeline.py
─────────────────────────────
Extracts images classified as Has_RBC from the cascade results,
splits them into clustered and unclustered buckets, and runs
Cellpose segmentation on each separately to count RBCs.

The .npy files saved here will be compiled into final counts
by a separate function downstream.

Usage:
    from rbc_segmentation_pipeline import extract_has_rbc, run_rbc_segmentation

    clustered_paths, unclustered_paths = extract_has_rbc(results)
    run_rbc_segmentation(clustered_paths,   save_dir="path/to/clustered_seg",   log_fn=print)
    run_rbc_segmentation(unclustered_paths, save_dir="path/to/unclustered_seg", log_fn=print)
"""

from pathlib import Path
import numpy as np
import cv2
from skimage import io


# ─────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────
def extract_has_rbc_ram(results, cells, log_fn=print):
    clustered = []
    unclustered = []
    skipped = 0

    # Build lookup: (parent, index) → cell dict
    lookup = {(c["parent"], c["index"]): c for c in cells}

    for r in results:
        last = r["path"][-1]
        pred = last["pred"]
        model = last["model"]

        if pred != 1:
            skipped += 1
            continue

        key = (r["parent"], r["index"])
        if key not in lookup:
            skipped += 1
            continue

        cell = lookup[key]

        if model == "Cluster_RBCCount":
            clustered.append(cell)
        elif model == "Unclustered_RBCCount":
            unclustered.append(cell)
        else:
            skipped += 1

    log_fn(f"\n── Has_RBC Extraction (RAM) ──")
    log_fn(f"  Clustered   Has_RBC: {len(clustered)}")
    log_fn(f"  Unclustered Has_RBC: {len(unclustered)}")
    log_fn(f"  Skipped: {skipped}")

    return clustered, unclustered

# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────
def run_rbc_segmentation_ram(
    cells: list[dict],
    model_path: str = "cpsam",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    batch_size: int = 8,
    log_fn=print,
):
    """
    Run Cellpose segmentation directly on in‑RAM cell images
    to count RBCs inside each monocyte.

    Args:
        cells: list of RAM cell dicts from extract_single_cells()
               each containing "image": np.ndarray (H,W,3)
        model_path: Cellpose model
        diameter: RBC diameter
        flow_threshold: Cellpose flow threshold
        cellprob_threshold: Cellpose cell probability threshold
        batch_size: Cellpose batch size
    """

    from cellpose import models, core

    if not core.use_gpu():
        raise RuntimeError("GPU required for Cellpose inference.")

    if len(cells) == 0:
        log_fn("No Has_RBC cells to segment.")
        return []

    # Extract NumPy images
    imgs = [c["image"] for c in cells]

    log_fn(f"\nRunning Cellpose on {len(imgs)} RAM images...")

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    try:
        masks, flows, _ = model.eval(
            imgs,
            batch_size=batch_size,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    except Exception as e:
        log_fn(f"❌ Cellpose inference failed: {e}")
        return []

    # Count RBCs
    results = []
    for cell, mask in zip(cells, masks):
        # RBC count = number of unique labels minus background
        rbc_count = len(np.unique(mask)) - 1

        results.append({
            "parent": cell["parent"],
            "index": cell["index"],
            "rbc_count": int(rbc_count),
        })

    log_fn(f"✔ RBC segmentation complete — {len(results)} cells processed")

    return results


# ─────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────
def run_full_rbc_segmentation_pipeline_ram(
    results: list[dict],
    cells: list[dict],
    model_path: str = "cpsam",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    batch_size: int = 8,
    log_fn=print,
) -> dict:
    """
    Full RAM-based RBC segmentation pipeline.

    Steps:
      1. Extract Has_RBC cells from cascade results (RAM objects)
      2. Run Cellpose on clustered and unclustered cells separately
      3. Count RBCs directly from Cellpose masks (RAM)
      4. Return structured results (no disk writes)
    """

    # 1. Extract Has_RBC cells (RAM)
    clustered_cells, unclustered_cells = extract_has_rbc_ram(
        results=results,
        cells=cells,
        log_fn=log_fn
    )

    log_fn(f"\n── Segmenting Clustered Has_RBC ({len(clustered_cells)} cells) ──")
    clustered_counts = run_rbc_segmentation_ram(
        cells=clustered_cells,
        model_path=model_path,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=batch_size,
        log_fn=log_fn,
    )

    log_fn(f"\n── Segmenting Unclustered Has_RBC ({len(unclustered_cells)} cells) ──")
    unclustered_counts = run_rbc_segmentation_ram(
        cells=unclustered_cells,
        model_path=model_path,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=batch_size,
        log_fn=log_fn,
    )

    # 2. Compute totals
    total_rbcs = (
        sum(c["rbc_count"] for c in clustered_counts) +
        sum(c["rbc_count"] for c in unclustered_counts)
    )

    log_fn(f"\n✅ Full RAM RBC segmentation pipeline complete.")
    log_fn(f"  Clustered   cells processed : {len(clustered_counts)}")
    log_fn(f"  Unclustered cells processed : {len(unclustered_counts)}")
    log_fn(f"  Total RBCs detected        : {total_rbcs}")

    return {
        "clustered_counts":   clustered_counts,
        "unclustered_counts": unclustered_counts,
        "total_rbcs":         total_rbcs,
    }

