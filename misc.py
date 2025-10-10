import numpy as np
import pprint

# Path to your .npy file
npy_path = "D:/Kyle 2025/Repos/Blood-Cell-Classification/SampleImages/Images/tile_x003_y002_cropped_seg.npy"

# Load the file
data = np.load(npy_path, allow_pickle=True)

# If it's a dictionary (common with Cellpose), convert to readable format
if isinstance(data.item(), dict):
    data_dict = data.item()
    print("Keys in .npy file:")
    for key in data_dict:
        print(f" - {key}: type = {type(data_dict[key])}, shape = {getattr(data_dict[key], 'shape', 'N/A')}")
    
    print("\nFull contents:")
    pprint.pprint(data_dict)
else:
    print("Raw array contents:")
    print(data)
