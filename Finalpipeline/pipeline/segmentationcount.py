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


# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────
def run_rbc_segmentation(
    image_paths: list[str],
    save_dir: str,
    model_path: str = "cpsam",
    diameter: float = 15,
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


# ─────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────
def run_full_rbc_segmentation_pipeline(
    results: list[dict],
    clustered_save_dir: str,
    unclustered_save_dir: str,
    model_path: str = "cpsam",
    diameter: float = 15,
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

    clustered_paths, unclustered_paths = extract_has_rbc(results, log_fn=log_fn)

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