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
def extract_has_rbc(
    results: list[dict],
    log_fn=print,
) -> tuple[list[str], list[str]]:
    """
    Comb through cascade results and extract image paths where the
    final RBC node predicted Has_RBC (pred=1).

    Args:
        results:  List of result dicts from run_classification().
        log_fn:   Callable for logging.

    Returns:
        (clustered_paths, unclustered_paths): two lists of file path strings.
    """
    clustered_paths   = []
    unclustered_paths = []
    skipped           = 0

    for r in results:
        path      = r.get("path", [])
        file      = r.get("file", "")
        last_node = path[-1] if path else None

        if last_node is None:
            skipped += 1
            continue

        model_name = last_node["model"]
        pred       = last_node["pred"]

        if pred != 1:
            # No_RBC (pred=0) or RBC_alone (pred=2) — skip for now
            skipped += 1
            continue

        if model_name == "Cluster_RBCCount":
            clustered_paths.append(file)
        elif model_name == "Unclustered_RBCCount":
            unclustered_paths.append(file)
        else:
            # Stopped at an earlier node — not relevant here
            skipped += 1

    log_fn(f"\n── Has_RBC Extraction ──")
    log_fn(f"  Clustered   Has_RBC: {len(clustered_paths)}")
    log_fn(f"  Unclustered Has_RBC: {len(unclustered_paths)}")
    log_fn(f"  Skipped (No_RBC / RBC_alone / early stop): {skipped}")
    log_fn(f"  Total processed: {len(results)}")

    return clustered_paths, unclustered_paths

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
def run_rbc_segmentation(
    image_paths: list[str],
    save_dir: str,
    model_path: str = "cpsam",
    diameter: float = 10,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    batch_size: int = 8,
    log_fn=print,
) -> list[str]:
    """
    Run Cellpose segmentation on a list of Has_RBC images to count
    individual RBCs inside each monocyte.

    Args:
        image_paths:         List of image file paths to segment.
        save_dir:            Directory to save .npy segmentation files.
        model_path:          Cellpose model path or name (default: "cpsam").
        diameter:            Expected RBC diameter in pixels. Set carefully —
                             RBCs inside monocytes are smaller than whole cells.
        flow_threshold:      Cellpose flow threshold.
        cellprob_threshold:  Cellpose cell probability threshold.
        batch_size:          Cellpose inference batch size.
        log_fn:              Callable for logging.

    Returns:
        List of saved .npy file paths.
    """
    from cellpose import models, core, io

    io.logger_setup()

    if not core.use_gpu():
        raise RuntimeError("No GPU detected. A GPU is required for Cellpose inference.")

    if not image_paths:
        log_fn("  No images to segment, skipping.")
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    # ── Load images ──
    log_fn(f"\n  Loading {len(image_paths)} images for RBC segmentation...")
    imgs, valid_paths = [], []

    for path in image_paths:
        p = Path(path)
        try:
            img = io.imread(str(p))
            imgs.append(img)
            valid_paths.append(p)
        except Exception as e:
            log_fn(f"  ❌ Failed to load {p.name}: {e}")
            continue

    if not imgs:
        log_fn("  No valid images loaded. Aborting.")
        return []

    # ── Inference ──
    log_fn(f"  Running Cellpose on {len(imgs)} images...")
    try:
        masks, flows, _ = model.eval(
            imgs,
            batch_size=batch_size,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            diameter=diameter,
        )
    except Exception as e:
        log_fn(f"  ❌ Cellpose inference failed: {e}")
        raise

    # ── Save results ──
    saved_files = []

    for i, p in enumerate(valid_paths):
        try:
            mask = masks[i]
            flow = flows[i]

            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]

            n_rbcs = len(np.unique(masks[i])) - 1  # exclude background

            seg_data = {
                "masks":    mask,
                "flows":    flow,
                "filename": str(p.resolve()),
                "diameter": diameter,
                "rbc_count": n_rbcs,
            }

            npy_path = save_dir / f"{p.stem}_rbc_seg.npy"
            np.save(npy_path, seg_data, allow_pickle=True)

            log_fn(f"  ✔ {p.name} → {n_rbcs} RBCs detected")
            saved_files.append(str(npy_path))

        except Exception as e:
            log_fn(f"  ❌ Failed saving {p.name}: {e}")
            continue

    log_fn(f"\n  ✅ RBC segmentation done — {len(saved_files)}/{len(valid_paths)} saved to {save_dir}")
    return saved_files

def run_rbc_segmentation_ram(
    cells: list[dict],
    model_path: str = "cpsam",
    diameter: float = 10,
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
            diameter=diameter,
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
def run_full_rbc_segmentation_pipeline(
    results: list[dict],
    clustered_save_dir: str,
    unclustered_save_dir: str,
    model_path: str = "cpsam",
    diameter: float = 10,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    batch_size: int = 8,
    log_fn=print,
) -> dict:
    """
    Full pipeline: extract Has_RBC images from cascade results,
    run segmentation separately for clustered and unclustered,
    then compile RBC counts from the saved .npy files.

    Args:
        results:              List of result dicts from run_classification().
        clustered_save_dir:   Directory to save clustered RBC .npy files.
        unclustered_save_dir: Directory to save unclustered RBC .npy files.
        model_path:           Cellpose model path or name.
        diameter:             Expected RBC diameter in pixels.
        flow_threshold:       Cellpose flow threshold.
        cellprob_threshold:   Cellpose cell probability threshold.
        batch_size:           Cellpose inference batch size.
        log_fn:               Callable for logging.

    Returns:
        dict with keys:
            "clustered_saved":    list of saved .npy paths for clustered
            "unclustered_saved":  list of saved .npy paths for unclustered
            "clustered_counts":   count_rbcs_from_segmentation output for clustered
            "unclustered_counts": count_rbcs_from_segmentation output for unclustered
            "total_rbcs":         combined RBC count across both buckets
    """
    from pipeline.metrics import count_rbcs_from_segmentation

    clustered_paths, unclustered_paths = extract_has_rbc_ram(results, log_fn=log_fn)

    log_fn(f"\n── Segmenting Clustered Has_RBC ({len(clustered_paths)} images) ──")
    clustered_saved = run_rbc_segmentation(
        clustered_paths,
        save_dir=clustered_save_dir,
        model_path=model_path,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=batch_size,
        log_fn=log_fn,
    )

    log_fn(f"\n── Segmenting Unclustered Has_RBC ({len(unclustered_paths)} images) ──")
    unclustered_saved = run_rbc_segmentation(
        unclustered_paths,
        save_dir=unclustered_save_dir,
        model_path=model_path,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=batch_size,
        log_fn=log_fn,
    )

    # ── Compile counts from saved .npy files ──
    clustered_counts   = count_rbcs_from_segmentation(clustered_save_dir,   log_fn=log_fn)
    unclustered_counts = count_rbcs_from_segmentation(unclustered_save_dir, log_fn=log_fn)
    total_rbcs         = clustered_counts["total_rbcs"] + unclustered_counts["total_rbcs"]

    log_fn(f"\n✅ Full RBC segmentation pipeline complete.")
    log_fn(f"  Clustered   .npy files : {len(clustered_saved)}")
    log_fn(f"  Unclustered .npy files : {len(unclustered_saved)}")
    log_fn(f"  Total RBCs detected    : {total_rbcs}")

    return {
        "clustered_saved":    clustered_saved,
        "unclustered_saved":  unclustered_saved,
        "clustered_counts":   clustered_counts,
        "unclustered_counts": unclustered_counts,
        "total_rbcs":         total_rbcs,
    }

def run_full_rbc_segmentation_pipeline_ram(
    results: list[dict],
    cells: list[dict],
    model_path: str = "cpsam",
    diameter: float = 10,
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
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=batch_size,
        log_fn=log_fn,
    )

    log_fn(f"\n── Segmenting Unclustered Has_RBC ({len(unclustered_cells)} cells) ──")
    unclustered_counts = run_rbc_segmentation_ram(
        cells=unclustered_cells,
        model_path=model_path,
        diameter=diameter,
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
