import pandas as pd
import shutil
import os
from urllib.parse import unquote
import re
import os
from PIL import Image
import numpy as np

# Path to csv data
Label_Export = 'C:/repos/Blood-Cell-Classification/LabelledData/LabelExport_20260312.csv'
MCwRBC_folder = "D:/MMA_LabelledData/Unsliced/Monocyte_with_RBC"
MCwORBC_folder = "D:/MMA_LabelledData/Unsliced/Monocyte_without_RBC"
Lymphocyte_folder = "D:/MMA_LabelledData/Unsliced/Lymphocyte"
RBConly_folder = "D:/MMA_LabelledData/Unsliced/RBC alone"
Unusable_folder = "D:/MMA_LabelledData/Unsliced/Unusable"
Clusteredcell_folder = "D:/MMA_LabelledData/Unsliced/Clustered_cell"

# Base parent folders
Unsliced_parent = "D:/MMA_LabelledData/Unsliced"
Sliced_parent = "D:/MMA_LabelledData/Sliced"

blackcontext_folder = "D:/MMA_LabelledData/contextside_isBlack"

# Helper functions
def fix_image_path(label_studio_path):
    # Remove the Label Studio prefix
    prefix = "/data/local-files/?d="
    stripped = label_studio_path.strip().replace(prefix, "")
    
    # URL-decode (e.g. %5C -> \, %20 -> space)
    decoded = unquote(stripped)
    
    # Convert forward slashes to backslashes and prepend drive letter
    local_path = "C:\\" + decoded.replace("/", "\\")
    
    return local_path

def build_dst_filename(local_path):
    # Extract slide identifier e.g. "Slide 1-1" -> "1_1"
    match = re.search(r'Slide (\d+)-(\d+)', local_path)
    if match:
        slide_id = f"{match.group(1)}_{match.group(2)}"
    else:
        slide_id = "unknown"
    
    # Split filename and extension, append slide ID
    basename = os.path.basename(local_path)
    name, ext = os.path.splitext(basename)
    return f"{name}_slide{slide_id}{ext}"  # e.g. tile_x001_y001_cell_0001_slide1_1.png

def slice_image(src_path, dst_path):
    """Crop image to top-left 256x256 pixels and save."""
    with Image.open(src_path) as img:
        """
        Checks if the right 256x256 portion of a 256x512 image is a black square.
        black_threshold: pixel value below which we consider a pixel black (0-255)
        """        
        # Crop right 256x256 (left, upper, right, lower)
        right_half = img.crop((256, 0, 512, 256))
        
        # Convert to numpy and check if all pixels are below threshold - 10 where 0 is pitch black
        pixels = np.array(right_half)
        is_black = np.all(pixels <= 10)

        if is_black:
            os.makedirs(blackcontext_folder, exist_ok=True)
            shutil.move(src_path, blackcontext_folder)
            print(f"  → Moved to black folder: {os.path.basename(src_path)}")

        else:
            cropped = img.crop((0, 0, 256, 256))
            cropped.save(dst_path)

# Import csv as dataframe
df_Labels = pd.read_csv(Label_Export)

# Map choice labels to destination folders
folder_map = {
    "Monocyte with RBC": MCwRBC_folder,
    "Monocyte without RBC": MCwORBC_folder,
    "Lymphocyte?": Lymphocyte_folder,
    "RBC alone": RBConly_folder,
    "Unusable": Unusable_folder,
    "Clustered cell": Clusteredcell_folder
}

# Create destination folders if they don't exist
for folder in folder_map.values():
    os.makedirs(folder, exist_ok=True)

# Copy each image to the appropriate folder
skipped = []
for _, row in df_Labels.iterrows():
    src = fix_image_path(row['image'])
    choice = row['choice']

    if choice not in folder_map:
        skipped.append((src, choice))
        continue

    dst_folder = folder_map[choice]
    dst_filename = build_dst_filename(src)        # <-- unique filename
    dst = os.path.join(dst_folder, dst_filename)

    try:
        shutil.copy2(src, dst)
    except FileNotFoundError:
        print(f"File not found, skipping: {src}")

print(f"Done. Files copied to subfolders.")
if skipped:
    print(f"Skipped {len(skipped)} rows with unrecognised labels: {set(s[1] for s in skipped)}")


# Walk through all subfolders in Unsliced
for subfolder in os.listdir(Unsliced_parent):
    unsliced_folder = os.path.join(Unsliced_parent, subfolder)
    
    if not os.path.isdir(unsliced_folder):
        continue

    # Mirror the subfolder under Sliced parent
    sliced_folder = os.path.join(Sliced_parent, subfolder)
    os.makedirs(sliced_folder, exist_ok=True)

    # Process each image in the subfolder
    processed, skipped = 0, 0
    for filename in os.listdir(unsliced_folder):
        src_path = os.path.join(unsliced_folder, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            continue

        dst_path = os.path.join(sliced_folder, filename)

        try:
            slice_image(src_path, dst_path)
            processed += 1
        except Exception as e:
            print(f"Failed to slice {filename}: {e}")
            skipped += 1

    print(f"[{subfolder}] Processed: {processed} | Skipped: {skipped}")

print("Done. All images sliced.")