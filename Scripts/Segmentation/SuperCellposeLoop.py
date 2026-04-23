def run_segmentation(base_dir="D:/MMA_batch1/Raw Image",
                     save_base_dir="D:/MMA_batch3_CellposeCustom/Segmented_Images",
                     overlay_base_dir="D:/MMA_batch3_CellposeCustom/Overlay"): # fucked up here, did not update the overlay save directory :(
    """
    Batch segmentation of microscopy images using Cellpose-SAM.
    Includes:
    - Black image filtering
    - Safe try/except around Cellpose
    - Safe saving
    - Overlay generation
    """
 
    import numpy as np
    from cellpose import models, core, io
    from cellpose.transforms import resize_image
    from pathlib import Path
    from tqdm import trange
    import matplotlib.pyplot as plt
    from natsort import natsorted
    import cv2
    from PIL import ImageStat

    io.logger_setup()

    # ============================================================
    # GPU CHECK
    # ============================================================
    if core.use_gpu() == False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(
        gpu=True,
        pretrained_model="D:/MMA_batch1/TrainedCellposev2/models/MMA_trainv5"
    )

    # ============================================================
    # PATH SETUP
    # ============================================================
    base_dir = Path(base_dir)
    save_base_dir = Path(save_base_dir)
    overlay_base_dir = Path(overlay_base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    large_image_dirs = list(base_dir.rglob("Large Image"))
    print(f"Found {len(large_image_dirs)} 'Large Image' folders")

    if len(large_image_dirs) == 0:
        raise FileNotFoundError("No 'Large Image' folders found")

    # ============================================================
    # PARAMETERS
    # ============================================================
    image_ext = ".tif"
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    black_threshold = 0.30  # Skip images with >30% black pixels
    diameter = 70

    # ============================================================
    # BLACK IMAGE CHECK
    # ============================================================
    def is_black_image(img, threshold=0.30):
        """
        Returns True if > threshold fraction of pixels are near-black.
        """
        # Convert to 2D if needed
        if img.ndim == 3:
            if img.shape[0] == 1:
                img_2d = img[0]
            elif img.shape[0] < 4:
                img_2d = img[0]
            else:
                img_2d = img
        else:
            img_2d = img

        # Fast mean check
        if img_2d.mean() < 5:
            return True

        # Pixel threshold check
        black_pixels = (img_2d < 10).sum() / img_2d.size
        return black_pixels > threshold

    # ============================================================
    # VISUALIZATION
    # ============================================================
    def normalize99(img):
        p1, p99 = np.percentile(img, [1, 99])
        return (img - p1) / (p99 - p1 + 1e-20)

    def random_colors(n):
        np.random.seed(42)
        return np.random.rand(n, 3)

    def visualize_segmentation(img, masks, filename, output_dir, alpha=0.5):
        if masks.ndim == 3:
            masks_2d = masks[0]
        else:
            masks_2d = masks.squeeze()

        if img.ndim == 4:
            img_2d = img[0]
        elif img.ndim == 3:
            if img.shape[0] == 1:
                img_2d = img[0]
            elif img.shape[0] < 4:
                img_2d = img[0]
            else:
                img_2d = img
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

        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(masks_2d, cmap='nipy_spectral')
        n_cells = len(np.unique(masks_2d)) - 1
        axes[1].set_title(f'Segmentation Masks ({n_cells} cells)')
        axes[1].axis('off')

        overlay = img_display.copy()
        cell_ids = np.unique(masks_2d)[1:]
        colors = random_colors(len(cell_ids))

        for idx, cell_id in enumerate(cell_ids):
            cell_mask = masks_2d == cell_id
            overlay[cell_mask] = (1 - alpha) * overlay[cell_mask] + alpha * colors[idx]

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = output_dir / f"{filename}_overlay.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path, n_cells

    # ============================================================
    # MAIN LOOP
    # ============================================================
    total_images = 0
    total_cells = 0

    for large_img_dir in large_image_dirs:
        print("\n" + "="*60)
        print(f"Processing: {large_img_dir}")
        print("="*60)

        files = natsorted([
            f for f in large_img_dir.glob("*" + image_ext)
            if "_masks" not in f.name and "_flows" not in f.name
        ])

        if len(files) == 0:
            print(f"No image files found in {large_img_dir}")
            continue

        print(f"Found {len(files)} images")

        folder_name = large_img_dir.parent.name
        save_dir = save_base_dir / folder_name
        overlay_dir = overlay_base_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        print("Loading images...")
        imgs = [io.imread(files[i]) for i in trange(len(files))]

        # ============================================================
        # FILTER BLACK IMAGES
        # ============================================================
        print("Filtering images by black pixel content...")
        filtered_imgs = []
        filtered_files = []

        for img, file in zip(imgs, files):
            if is_black_image(img, threshold=black_threshold):
                print(f"  ✗ {file.name} - SKIPPED (black/blank image)")
                continue

            print(f"  ✓ {file.name} - accepted")
            filtered_imgs.append(img)
            filtered_files.append(file)

        print(f"Kept {len(filtered_imgs)}/{len(files)} images\n")

        if len(filtered_imgs) == 0:
            print("No valid images in this folder, skipping.")
            continue

        # ============================================================
        # SAFE CELLPOSE INFERENCE
        # ============================================================
        print("Running cellpose-SAM...")

        try:
            masks, flows, styles = model.eval(
                filtered_imgs,
                batch_size=32,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize={"tile_norm_blocksize": tile_norm_blocksize}, 
                diameter=diameter
            )

        except Exception as e:
            print("\n❌ Cellpose failed on this folder.")
            print(f"Error: {e}")
            print("Skipping folder...\n")
            continue

        # ============================================================
        # SAVE RESULTS
        # ============================================================
        print("Saving segmentations and generating visualizations...")

        for i, f in enumerate(filtered_files):
            try:
                f = Path(f)
                img = imgs[i]
                mask = masks[i]
                flow = flows[i]

                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                if mask.ndim == 2:
                    mask = mask[np.newaxis, ...]

                Ly, Lx = img.shape[-2:]

                flows_new = []
                flows_new.append(flow[0].copy())
                flows_new.append(flow[1].copy())
                flows_new.append((np.clip(normalize99(flow[2].copy()), 0, 1) * 255).astype("uint8"))
                flows_new.append(flow[2].copy())

                resized_flows = []
                for fl in flows_new:
                    if fl.shape[-2:] != (Ly, Lx):
                        resized = resize_image(
                            fl,
                            Ly=Ly,
                            Lx=Lx,
                            no_channels=(fl.ndim == 2),
                            interpolation=cv2.INTER_NEAREST
                        )
                        resized_flows.append(resized)
                    else:
                        resized_flows.append(fl)

                seg_data = {
                    'masks': mask,
                    'flows': resized_flows,
                    'filename': str(f.resolve()),
                    'diameter': None,
                    'ismanual': np.zeros(len(np.unique(mask)) - 1, dtype=bool),
                }

                save_path = save_dir / f"{f.stem}_seg.npy"
                np.save(save_path, seg_data, allow_pickle=True)

                overlay_path, n_cells = visualize_segmentation(
                    img, mask, f.stem, overlay_dir, alpha=0.4
                )

                print(f"Saved: {f.stem}")
                print(f"  Cells found: {n_cells}")

                total_images += 1
                total_cells += n_cells

            except Exception as e:
                print(f"❌ Failed to save/visualize {f.name}: {e}")
                continue

        print(f"\nFolder complete: {len(filtered_imgs)} images processed")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("BATCH SEGMENTATION COMPLETE!")
    print(f"Total images processed: {total_images}")
    print(f"Total cells found: {total_cells}")
    print(f"Segmentation files saved to: {save_base_dir}")
    print(f"Overlay visualizations saved to: {overlay_base_dir}")
    print("="*60)


if __name__ == "__main__":
    run_segmentation()
