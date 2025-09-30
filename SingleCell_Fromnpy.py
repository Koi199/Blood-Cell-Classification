import numpy as np
from skimage import io, measure
import os

# Load your Cellpose .npy results
data = np.load("C:\\Users\\kylea\\Repos\\Blood-Cell-Classification\\SampleImages\\Images\\tile_x003_y002_cropped_seg.npy", allow_pickle=True).item()
masks = data['masks']
img = io.imread(data['filename'])   # original TIFF image
imgname = os.path.basename(data['filename'])
imgname = imgname[:-4]
print(f"Processing image: {imgname}")

# Output folder
output_dir = imgname + "single_cells_padded"
os.makedirs(output_dir, exist_ok=True)

# Padding size in pixels
pad = 10   # change this to whatever margin you want

props = measure.regionprops(masks)

for i, prop in enumerate(props, start=1):
    minr, minc, maxr, maxc = prop.bbox
    
    # Expand bounding box by pad, but keep inside image bounds
    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, masks.shape[0])
    maxc = min(maxc + pad, masks.shape[1])
    
    # Crop image and mask
    crop = img[minr:maxr, minc:maxc].copy()
    cell_mask = (masks[minr:maxr, minc:maxc] == prop.label)
    
    # Apply mask so background is zeroed out
    if crop.ndim == 3:  # color image
        crop[~cell_mask] = 0
    else:               # grayscale
        crop[~cell_mask] = 0
    
    # Save padded cell image
    out_path = os.path.join(output_dir, f"_cell_{i}.png")
    io.imsave(out_path, crop.astype(np.uint8))

print(f"Saved {len(props)} padded single-cell images to '{output_dir}'")