import numpy as np
from skimage import io, measure
import os
from PIL import Image

# Load your Cellpose .npy results
data = np.load("C:\\Users\\kylea\\Repos\\Blood-Cell-Classification\\SampleImages\\Images\\tile_x003_y002_cropped_seg.npy", allow_pickle=True).item()
masks = data['masks']
img = io.imread(data['filename'])   # original TIFF image
imgname = os.path.basename(data['filename'])
imgname = imgname[:-4] # remove .tif extension
print(f"Processing image: {imgname}")

# Padding size in pixels
pad = 10   # change this to whatever margin you want
props = measure.regionprops(masks) # get properties of labeled regions
# User input for mode
# mode = input("Without Background? Enter 1 || With Background? Enter 2: ")

# Output folder
output_dir = imgname + "_single_cells_padded"
os.makedirs(output_dir, exist_ok=True)

for i, prop in enumerate(props, start=1):
    minr, minc, maxr, maxc = prop.bbox
    
    # Expand bounding box by pad, but keep inside image bounds
    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, masks.shape[0])
    maxc = min(maxc + pad, masks.shape[1])
    
    # Crop image and mask
    crop_img = img[minr:maxr, minc:maxc].copy()
    cell_mask = (masks[minr:maxr, minc:maxc] == prop.label)
    
    # Apply mask so background is zeroed out
    if crop_img.ndim == 3:  # color image
        crop_img[~cell_mask] = 0
    else:               # grayscale
        crop_img[~cell_mask] = 0

    # Convert NumPy crop to PIL RGBA
    pil_crop = Image.fromarray(crop_img.astype(np.uint8)).convert("RGBA")

    # Open new black background image
    bg = Image.open(
        r"C:\Users\kylea\Repos\Blood-Cell-Classification\SampleImages\Images\Black_background_128x128.png"
    ).convert("RGBA")
    
    # Center the cropped cell on the background
    x = (bg.width - pil_crop.width) // 2
    y = (bg.height - pil_crop.height) // 2
    bg.paste(pil_crop, (x, y), pil_crop)  # use pil_crop as its own mask

    # Save result
    out_path = os.path.join(output_dir, f"cell_{i}.png")
    bg.save(out_path)
    print(f"Saved {out_path}")
