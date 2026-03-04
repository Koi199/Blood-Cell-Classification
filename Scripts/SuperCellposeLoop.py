def run_segmentation(base_dir="D:/Kyle 2025/MMA",
                     save_base_dir="D:/Kyle 2025/Repos/Blood-Cell-Classification/Segmented_Images",
                     overlay_base_dir="D:/Kyle 2025/Repos/Blood-Cell-Classification"):
    """
    This script performs batch segmentation of microscopy images using the Cellpose-SAM model.
    It processes all images in "Large Images" folders under a specified base directory,
    filters out images with excessive black pixels, runs segmentation, saves the results,
    and generates overlay visualizations.
    """

    import numpy as np
    from cellpose import models, core, io, plot
    from cellpose.transforms import resize_image
    from pathlib import Path
    from tqdm import trange
    import matplotlib.pyplot as plt
    from natsort import natsorted
    import cv2

    io.logger_setup()

    # ============ SEGMENTATION PARAMETERS ============
    if core.use_gpu() == False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)

    # Base directory containing all the slide folders
    base_dir = Path("D:/Kyle 2025/MMA")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    # Find all "Large Images" folders recursively
    large_image_dirs = list(base_dir.rglob("Large Image"))
    print(f"Found {len(large_image_dirs)} 'Large Image' folders")

    if len(large_image_dirs) == 0:
        raise FileNotFoundError("No 'Large Image' folders found")

    image_ext = ".tif"
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    black_threshold = 0.30  # Skip images with >30% black pixels

    # save_base_dir = Path("D:/Kyle 2025/Repos/Blood-Cell-Classification/Segmented_Images")
    # overlay_base_dir = Path("D:/Kyle 2025/Repos/Blood-Cell-Classification/Overlays")

    def normalize99(img):
        """Normalize image to 0-1 range using 1st and 99th percentiles"""
        p1, p99 = np.percentile(img, [1, 99])
        return (img - p1) / (p99 - p1 + 1e-20)

    def random_colors(n):
        """Generate n random colors for visualization"""
        np.random.seed(42)
        colors = np.random.rand(n, 3)
        return colors

    def visualize_segmentation(img, masks, filename, output_dir, alpha=0.5):
        """
        Create and save overlay visualization of masks on original image
        """
        # Handle 3D data - take first z-slice if needed
        if masks.ndim == 3:
            masks_2d = masks[0]
        else:
            masks_2d = masks.squeeze()
        
        # Handle image dimensions - squeeze single z dimension
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
        
        # Normalize image for display
        if img_2d.dtype == np.uint16:
            img_display = img_2d / img_2d.max()
        elif img_2d.dtype == np.uint8:
            img_display = img_2d / 255.0
        else:
            img_display = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-10)
        
        # Convert grayscale to RGB if needed
        if img_display.ndim == 2:
            img_display = np.stack([img_display] * 3, axis=-1)
        
        # Create figure
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

    # Process each "Large Images" directory
    total_images = 0
    total_cells = 0

    for large_img_dir in large_image_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {large_img_dir}")
        print('='*60)
        
        # Get image files
        files = natsorted([f for f in large_img_dir.glob("*"+image_ext) 
                        if "_masks" not in f.name and "_flows" not in f.name])
        
        if len(files) == 0:
            print(f"No image files found in {large_img_dir}")
            continue
        
        print(f"Found {len(files)} images")
        
        # Create output directories with descriptive names
        folder_name = large_img_dir.parent.name
        save_dir = save_base_dir / folder_name
        overlay_dir = overlay_base_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all images
        print("Loading images...")
        imgs = [io.imread(files[i]) for i in trange(len(files))]
        
        # Filter out images with >30% black pixels
        print("Filtering images by black pixel content...")
        filtered_imgs = []
        filtered_files = []
        
        for img, file in zip(imgs, files):
            # Handle 3D images
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img_2d = img[0]
                elif img.shape[0] < 4:
                    img_2d = img[0]
                else:
                    img_2d = img
            else:
                img_2d = img
            
            # Calculate percentage of black pixels (value < 10)
            black_pixels = np.sum(img_2d < 10) / img_2d.size
            
            if black_pixels <= black_threshold:
                filtered_imgs.append(img)
                filtered_files.append(file)
                print(f"  ✓ {file.name} ({black_pixels*100:.1f}% black)")
            else:
                print(f"  ✗ {file.name} ({black_pixels*100:.1f}% black) - SKIPPED")
        
        print(f"Kept {len(filtered_imgs)}/{len(files)} images\n")
        
        if len(filtered_imgs) == 0:
            print("No images passed the black pixel filter, skipping this folder")
            continue
        
        # Run segmentation on filtered images
        print("Running cellpose-SAM...")
        masks, flows, styles = model.eval(
            filtered_imgs, 
            batch_size=32, 
            flow_threshold=flow_threshold, 
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize}
        )
        
        # Save segmentations
        print("Saving segmentations and generating visualizations...")
        
        for i, f in enumerate(filtered_files):
            f = Path(f)
            img = imgs[i]
            mask = masks[i]
            flow = flows[i]
            
            # Handle 2D images
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
            
            Ly, Lx = img.shape[-2:]
            
            # Prepare flows
            flows_new = []
            flows_new.append(flow[0].copy())
            flows_new.append(flow[1].copy())
            flows_new.append((np.clip(normalize99(flow[2].copy()), 0, 1) * 255).astype("uint8"))
            flows_new.append(flow[2].copy())
            
            # Resize flows if needed
            resized_flows = []
            for j, fl in enumerate(flows_new):
                if fl.shape[-2:] != (Ly, Lx):
                    if fl.ndim == 3:
                        resized = resize_image(fl, Ly=Ly, Lx=Lx, 
                                            no_channels=False, 
                                            interpolation=cv2.INTER_NEAREST)
                    elif fl.ndim == 2:
                        resized = resize_image(fl, Ly=Ly, Lx=Lx, 
                                            no_channels=True, 
                                            interpolation=cv2.INTER_NEAREST)
                    else:
                        resized = resize_image(fl, Ly=Ly, Lx=Lx, 
                                            no_channels=False, 
                                            interpolation=cv2.INTER_NEAREST)
                    resized_flows.append(resized)
                else:
                    resized_flows.append(fl)
            
            # Build save dictionary
            seg_data = {
                'masks': mask,
                'flows': resized_flows,
                'filename': str(f.resolve()),
                'diameter': None,
                'ismanual': np.zeros(len(np.unique(mask))-1, dtype=bool),
            }
            
            # Save segmentation
            save_path = save_dir / f"{f.stem}_seg.npy"
            np.save(save_path, seg_data, allow_pickle=True)
            
            # Generate visualization
            overlay_path, n_cells = visualize_segmentation(img, mask, f.stem, overlay_dir, alpha=0.4)
            
            print(f"Saved: {f.stem}")
            print(f"  Cells found: {n_cells}")
            
            total_images += 1
            total_cells += n_cells
        
        print(f"\nFolder complete: {len(files)} images processed")

    print("\n" + "="*60)
    print("BATCH SEGMENTATION COMPLETE!")
    print(f"Total images processed: {total_images}")
    print(f"Total cells found: {total_cells}")
    print(f"Segmentation files saved to: {save_base_dir}")
    print(f"Overlay visualizations saved to: {overlay_base_dir}")
    print("="*60)

if __name__ == "__main__":
    run_segmentation()

