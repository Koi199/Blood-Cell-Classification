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

dir = Path("D:/Kyle 2025/MMA/Slide 1-1/testfolder")
if not dir.exists():
    raise FileNotFoundError("directory does not exist")

image_ext = ".tif"
files = natsorted([f for f in dir.glob("*"+image_ext) 
                    if "_masks" not in f.name and "_flows" not in f.name])

if len(files) == 0:
    raise FileNotFoundError("no image files found")
else:
    print(f"{len(files)} images in folder.")

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

save_dir = Path("D:/Kyle 2025/Repos/Blood-Cell-Classification/Segemented_Images")
save_dir.mkdir(exist_ok=True)

overlay_dir = Path("D:/Kyle 2025/Repos/Blood-Cell-Classification/Overlays")
overlay_dir.mkdir(exist_ok=True)

print("loading images")
imgs = [io.imread(files[i]) for i in trange(len(files))]

print("running cellpose-SAM")
masks, flows, styles = model.eval(
    imgs, 
    batch_size=32, 
    flow_threshold=flow_threshold, 
    cellprob_threshold=cellprob_threshold,
    normalize={"tile_norm_blocksize": tile_norm_blocksize}
)

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
        img_2d = img[0]  # Take first z-slice
    elif img.ndim == 3:
        if img.shape[0] == 1:
            img_2d = img[0]  # Remove single z dimension
        elif img.shape[0] < 4:  # Likely (Z, Y, X)
            img_2d = img[0]
        else:  # Likely (Y, X, C)
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
    
    # Plot 1: Original image
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot 2: Masks only
    axes[1].imshow(masks_2d, cmap='nipy_spectral')
    n_cells = len(np.unique(masks_2d)) - 1
    axes[1].set_title(f'Segmentation Masks ({n_cells} cells)')
    axes[1].axis('off')
    
    # Plot 3: Overlay
    overlay = img_display.copy()
    cell_ids = np.unique(masks_2d)[1:]  # Skip background
    colors = random_colors(len(cell_ids))
    
    for idx, cell_id in enumerate(cell_ids):
        cell_mask = masks_2d == cell_id
        overlay[cell_mask] = (1 - alpha) * overlay[cell_mask] + alpha * colors[idx]
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f"{filename}_overlay.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path, n_cells

print("\nsaving segmentations and generating visualizations")

for i, f in enumerate(files):
    f = Path(f)
    img = imgs[i]
    mask = masks[i]
    flow = flows[i]
    
    # Handle 2D images (add z-axis if needed)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    
    Ly, Lx = img.shape[-2:]
    
    # Prepare flows in GUI format
    flows_new = []
    flows_new.append(flow[0].copy())  # RGB flow
    flows_new.append(flow[1].copy())  # dP (flow vectors)
    flows_new.append((np.clip(normalize99(flow[2].copy()), 0, 1) * 255).astype("uint8"))  # normalized cellprob
    flows_new.append(flow[2].copy())  # original cellprob
    
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
    
    # Build the save dictionary
    seg_data = {
        'masks': mask,
        'flows': resized_flows,
        'filename': str(f.resolve()),
        'diameter': None,
        'ismanual': np.zeros(len(np.unique(mask))-1, dtype=bool),
    }
    
    # Save as *_seg.npy
    save_path = save_dir / f"{f.stem}_seg.npy"
    np.save(save_path, seg_data, allow_pickle=True)
    
    # Generate and save visualization
    overlay_path, n_cells = visualize_segmentation(img, mask, f.stem, overlay_dir, alpha=0.4)
    
    print(f"Saved: {save_path}")
    print(f"  Cells found: {n_cells}")
    print(f"  Overlay: {overlay_path}")

print("\n" + "="*60)
print("SEGMENTATION COMPLETE!")
print(f"Segmentation files saved to: {save_dir}")
print(f"Overlay visualizations saved to: {overlay_dir}")
print("="*60)
