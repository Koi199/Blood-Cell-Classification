from pathlib import Path
import numpy as np
import cv2


def is_black_image(img, threshold=0.30):
    if img.ndim == 3:
        img_2d = img[0] if img.shape[0] < 4 else img
    else:
        img_2d = img

    if img_2d.mean() < 5:
        return True
    return (img_2d < 10).sum() / img_2d.size > threshold


def normalize99(img):
    p1, p99 = np.percentile(img, [1, 99])
    return (img - p1) / (p99 - p1 + 1e-20)


def random_colors(n):
    np.random.seed(42)
    return np.random.rand(n, 3)


def visualize_segmentation(img, masks, filename, output_dir, alpha=0.5):
    import matplotlib.pyplot as plt

    masks_2d = masks[0] if masks.ndim == 3 else masks.squeeze()

    if img.ndim == 4:
        img_2d = img[0]
    elif img.ndim == 3:
        img_2d = img[0] if img.shape[0] < 4 else img
    else:
        img_2d = img

    if img_2d.dtype == np.uint16:
        img_display = img_2d / img_2d.max()
    elif img_2d.dtype == np.uint8:
        img_display = img_2d / 255.0
    else:
        img_display = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-10)

    if img_display.ndim == 2:
        img_display = np.stack([img_display] * 3, axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(img_display, cmap="gray");  axes[0].set_title("Original");      axes[0].axis("off")
    axes[1].imshow(masks_2d, cmap="nipy_spectral")
    n_cells = len(np.unique(masks_2d)) - 1
    axes[1].set_title(f"Masks ({n_cells} cells)"); axes[1].axis("off")

    overlay = img_display.copy()
    cell_ids = np.unique(masks_2d)[1:]
    colors = random_colors(len(cell_ids))
    for idx, cell_id in enumerate(cell_ids):
        cell_mask = masks_2d == cell_id
        overlay[cell_mask] = (1 - alpha) * overlay[cell_mask] + alpha * colors[idx]

    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()

    save_path = Path(output_dir) / f"{filename}_overlay.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path, n_cells


def run_segmentation(
    image_paths: list[str],
    save_base_dir: str,
    overlay_base_dir: str,
    model_path: str = "D:/MMA_batch1/TrainedCellpose/models/MMA_trainv3",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    tile_norm_blocksize: int = 0,
    black_threshold: float = 0.30,
    batch_size: int = 32,
    log_fn=print,
):
    """
    Run Cellpose segmentation on a list of image file paths.

    Args:
        image_paths:         List of absolute image file paths (from UI upload).
        save_base_dir:       Directory to save .npy segmentation files.
        overlay_base_dir:    Directory to save overlay .png visualizations.
        model_path:          Path to pretrained Cellpose model.
        flow_threshold:      Cellpose flow threshold.
        cellprob_threshold:  Cellpose cell probability threshold.
        tile_norm_blocksize: Tile normalization block size.
        black_threshold:     Fraction of black pixels to skip an image.
        batch_size:          Cellpose inference batch size.
        log_fn:              Callable for logging — pass worker.log.emit for Qt signal.

    Returns:
        dict with keys: total_images, total_cells, saved_files
    """
    from cellpose import models, core, io
    from cellpose.transforms import resize_image

    io.logger_setup()

    if not core.use_gpu():
        raise RuntimeError("No GPU detected. A GPU is required for Cellpose inference.")

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    save_base_dir = Path(save_base_dir)
    overlay_base_dir = Path(overlay_base_dir)
    save_base_dir.mkdir(parents=True, exist_ok=True)
    overlay_base_dir.mkdir(parents=True, exist_ok=True)

    # ── Load images ────────────────────────────────────────────
    log_fn(f"Loading {len(image_paths)} images...")
    imgs, valid_paths = [], []

    for path in image_paths:
        p = Path(path)
        img = io.imread(str(p))

        if is_black_image(img, threshold=black_threshold):
            log_fn(f"  ✗ SKIPPED (blank): {p.name}")
            continue

        log_fn(f"  ✓ Accepted: {p.name}")
        imgs.append(img)
        valid_paths.append(p)

    if not imgs:
        log_fn("No valid images after filtering. Aborting.")
        return {"total_images": 0, "total_cells": 0, "saved_files": []}

    log_fn(f"Running Cellpose on {len(imgs)} images...")

    # ── Inference ──────────────────────────────────────────────
    try:
        masks, flows, _ = model.eval(
            imgs,
            batch_size=batch_size,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize},
        )
    except Exception as e:
        log_fn(f"❌ Cellpose inference failed: {e}")
        raise

    # ── Save results ───────────────────────────────────────────
    total_cells = 0
    saved_files = []

    for i, f in enumerate(valid_paths):
        try:
            img  = imgs[i]
            mask = masks[i]
            flow = flows[i]

            if img.ndim  == 2: img  = img[np.newaxis, ...]
            if mask.ndim == 2: mask = mask[np.newaxis, ...]

            Ly, Lx = img.shape[-2:]

            raw_flows = [
                flow[0].copy(),
                flow[1].copy(),
                (np.clip(normalize99(flow[2].copy()), 0, 1) * 255).astype("uint8"),
                flow[2].copy(),
            ]
            resized_flows = [
                resize_image(fl, Ly=Ly, Lx=Lx,
                             no_channels=(fl.ndim == 2),
                             interpolation=cv2.INTER_NEAREST)
                if fl.shape[-2:] != (Ly, Lx) else fl
                for fl in raw_flows
            ]

            seg_data = {
                "masks":    mask,
                "flows":    resized_flows,
                "filename": str(f.resolve()),
                "diameter": None,
                "ismanual": np.zeros(len(np.unique(mask)) - 1, dtype=bool),
            }

            npy_path = save_base_dir / f"{f.stem}_seg.npy"
            np.save(npy_path, seg_data, allow_pickle=True)

            _, n_cells = visualize_segmentation(img, mask, f.stem, overlay_base_dir, alpha=0.4)

            log_fn(f"  ✔ {f.name} → {n_cells} cells")
            total_cells += n_cells
            saved_files.append(str(npy_path))

        except Exception as e:
            log_fn(f"  ❌ Failed on {f.name}: {e}")
            continue

    log_fn(f"\n✅ Done — {len(saved_files)} images, {total_cells} cells total.")
    return {
        "total_images": len(saved_files),
        "total_cells":  total_cells,
        "saved_files":  saved_files,
    }