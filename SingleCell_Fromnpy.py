import numpy as np
import os
from PIL import Image

def extract_single_cells(npy_path, output_dir, prefix="cell"):
    # Load the .npy file (assumed shape: [num_cells, height, width] or [num_cells, height, width, channels])
    cells = np.load(npy_path, allow_pickle = True)
    os.makedirs(output_dir, exist_ok=True)

    for idx, cell_img in enumerate(cells):
        # If grayscale, expand dims to (H, W, 1) for PIL compatibility
        if cell_img.ndim == 2:
            cell_img = np.expand_dims(cell_img, axis=-1)
        # If single channel, convert to 3 channels for RGB
        if cell_img.shape[-1] == 1:
            cell_img = np.repeat(cell_img, 3, axis=-1)
        # Convert to uint8 if necessary
        if cell_img.dtype != np.uint8:
            cell_img = (255 * (cell_img - cell_img.min()) / (cell_img.ptp() + 1e-8)).astype(np.uint8)
        img = Image.fromarray(cell_img)
        img.save(os.path.join(output_dir, f"{prefix}_{idx:04d}.png"))

if __name__ == "__main__":
    npy_file = "C:\\Users\\kylea\\Repos\\Blood-Cell-Classification\\SampleImages\\Images\\tile_x003_y002_cropped_seg.npy"  # Path to your .npy file
    output_folder = "single_cells"
    extract_single_cells(npy_file, output_folder)