import os
from PIL import Image

# Path to the folder containing .tif images
input_folder = r"Cell_Data/images"
output_folder = r"Cell_Data_png"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
        # Open the image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        
        # Convert and save as PNG
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)
        img.save(output_path, "PNG")
        print(f"Converted {filename} -> {output_filename}")

print("All conversions done!")
