import numpy as np
from skimage import io, measure
import os
from PIL import Image
import cv2
from pathlib import Path

def extract_single_cells(seg_file, output_dir, pad=20, bg_size=256, denoise_strength=10, contrast=2.0):
    """
    Extract single cells from segmentation with maximum image quality
    
    Parameters:
    -----------
    seg_file : str or Path
        Path to _seg.npy file
    output_dir : str or Path
        Directory to save extracted cells
    pad : int
        Padding around cell in pixels
    bg_size : int
        Background image size (256, 512, etc.)
    denoise_strength : int
        Denoising strength (5-20, higher = more denoising)
    contrast : float
        Contrast multiplier (1.0 = no change, 2.0 = double contrast, 0.5 = half contrast)
    """
    
    # Load segmentation data
    data = np.load(seg_file, allow_pickle=True).item()
    masks = data['masks']
    img = io.imread(data['filename'])
    
    imgname = Path(data['filename']).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing image: {imgname}")
    print(f"Image dtype: {img.dtype}, shape: {img.shape}\n")
    
    # Handle 3D masks
    if masks.ndim == 3:
        masks = masks[0]
    
    # Handle 3D images
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[0] < 4:
            img = img[0]
    
    # Get cell properties
    props = measure.regionprops(masks)
    print(f"Found {len(props)} cells\n")
    
    # Extract each cell
    for i, prop in enumerate(props, start=1):
        bbox = prop.bbox
        
        # Handle both 2D and 3D bbox
        if len(bbox) == 6:
            min_z, minr, minc, max_z, maxr, maxc = bbox
        else:
            minr, minc, maxr, maxc = bbox
        
        # Expand bounding box with padding
        minr = max(minr - pad, 0)
        minc = max(minc - pad, 0)
        maxr = min(maxr + pad, masks.shape[0])
        maxc = min(maxc + pad, masks.shape[1])
        
        # Crop image and mask
        crop_img = img[minr:maxr, minc:maxc].copy()
        crop_mask = masks[minr:maxr, minc:maxc] == prop.label
        
        # Convert image to uint8
        if crop_img.dtype == np.uint16:
            crop_img = (crop_img.astype(np.float32) / crop_img.max() * 255).astype(np.uint8)
        elif crop_img.dtype != np.uint8:
            crop_img = ((crop_img.astype(np.float32) - crop_img.min()) / 
                       (crop_img.max() - crop_img.min() + 1e-10) * 255).astype(np.uint8)
        
        # Apply non-local means denoising
        crop_img = cv2.fastNlMeansDenoising(crop_img, h=denoise_strength, 
                                           templateWindowSize=7, searchWindowSize=21)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        crop_img = clahe.apply(crop_img)
        
        # Apply contrast adjustment
        if contrast != 1.0:
            # Convert to float, adjust contrast around midpoint (128)
            crop_img = crop_img.astype(np.float32)
            crop_img = 128 + (crop_img - 128) * contrast
            crop_img = np.clip(crop_img, 0, 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        if crop_img.ndim == 2:
            crop_rgb = np.stack([crop_img] * 3, axis=-1)
        else:
            crop_rgb = crop_img.copy()
        
        # Apply mask - set background to 0
        for c in range(crop_rgb.shape[-1]):
            crop_rgb[~crop_mask, c] = 0
        
        # Create canvas with the cropped cell centered
        canvas = np.zeros((bg_size, bg_size, 3), dtype=np.uint8)
        
        # Center the crop on canvas
        y_offset = (bg_size - crop_rgb.shape[0]) // 2
        x_offset = (bg_size - crop_rgb.shape[1]) // 2
        
        # Ensure we don't go out of bounds
        y_min = max(0, y_offset)
        y_max = min(bg_size, y_offset + crop_rgb.shape[0])
        x_min = max(0, x_offset)
        x_max = min(bg_size, x_offset + crop_rgb.shape[1])
        
        crop_y_min = max(0, -y_offset)
        crop_y_max = crop_y_min + (y_max - y_min)
        crop_x_min = max(0, -x_offset)
        crop_x_max = crop_x_min + (x_max - x_min)
        
        canvas[y_min:y_max, x_min:x_max] = crop_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # Convert to PIL and save
        pil_img = Image.fromarray(canvas, mode='RGB')
        out_path = output_dir / f"{imgname}_cell_{i:04d}.png"
        pil_img.save(out_path)
        
        if i % 10 == 0 or i == len(props):
            print(f"Saved {i}/{len(props)} cells")
    
    print(f"\nCompleted! {len(props)} cells extracted to {output_dir}")


# ============ USAGE ============

# # Base directories
# segmented_images_dir = Path("D:/Kyle 2025/Repos/Blood-Cell-Classification/Segmented_Images")
# output_base_dir = Path("D:/Kyle 2025/Repos/Blood-Cell-Classification/SingleCells_Batch")

# # Parameters
# pad = 20
# bg_size = 256
# denoise_strength = 15
# contrast_values = [1.0, 1.6]  # Process with different contrast settings

# # Find all _seg.npy files
# seg_files = list(segmented_images_dir.glob("**/*_seg.npy"))
# print(f"Found {len(seg_files)} segmentation files\n")

# if len(seg_files) == 0:
#     print("No segmentation files found. Check your directory path.")
# else:
#     # Process each contrast setting
#     for contrast in contrast_values:
#         print(f"\n{'='*60}")
#         print(f"Processing with contrast = {contrast}")
#         print(f"{'='*60}\n")
        
#         # Process each segmentation file
#         for idx, seg_file in enumerate(seg_files, start=1):
#             print(f"\n[{idx}/{len(seg_files)}] Processing: {seg_file.name}")
#             print(f"Path: {seg_file}\n")
            
#             try:
#                 # Extract slide folder name from the file's parent directory
#                 slide_folder = seg_file.parent.name
                
#                 # Create output directory with slide grouping
#                 output_dir = output_base_dir / f"contrast_{contrast}" / slide_folder
#                 output_dir.mkdir(parents=True, exist_ok=True)
                
#                 extract_single_cells(
#                     seg_file=seg_file,
#                     output_dir=output_dir,
#                     pad=pad,
#                     bg_size=bg_size,
#                     denoise_strength=denoise_strength,
#                     contrast=contrast
#                 )
#             except Exception as e:
#                 print(f"ERROR processing {seg_file.name}: {str(e)}\n")
    
#     print(f"\n{'='*60}")
#     print("ALL PROCESSING COMPLETE!")
#     print(f"{'='*60}")