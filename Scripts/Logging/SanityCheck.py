import os
import re
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir":    "D:/MMA_LabelledData/Sliced",
    "img_size":    256,
    "batch_size":  32,
    "num_workers": 4,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",

    # Uncertainty band
    "uncertainty_low":  0.40,
    "uncertainty_high": 0.60,

    # Model checkpoints
    "stage1_checkpoint": "checkpoints_stage1/resnet34_binary.pth",
    "stage2_checkpoint": "checkpoints_stage2/resnet34_stage2.pth",
    "3class_checkpoint": "checkpoints_3class/resnet34_3class.pth",

    # Output CSV paths
    "stage1_output": "uncertainty_stage1.csv",
    "stage2_output": "uncertainty_stage2.csv",
    "3class_output": "uncertainty_3class.csv",

    # Label export CSV for ID lookup
    "label_export_csv": "C:/repos/Blood-Cell-Classification/LabelledData/LabelExport_20260311.csv",
}

# ─────────────────────────────────────────────
# ALL FOLDERS TO SCAN
# ─────────────────────────────────────────────
ALL_FOLDERS = [
    "Monocyte_with_RBC",
    "Monocyte_without_RBC",
    "Clustered_cell",
    "Unusable",
    "Lymphocyte",
    "RBC alone",
]

# ─────────────────────────────────────────────
# LABEL STUDIO ID LOOKUP
# Builds a dict keyed by Slide%20X-Y%5Ccell_stem
# so each file path can be matched unambiguously
# ─────────────────────────────────────────────
def build_label_lookup(label_csv_path):
    print(f"\n── Building label ID lookup ──")
    df = pd.read_csv(label_csv_path)

    lookup    = {}
    unmatched = 0
    for _, row in df.iterrows():
        image = str(row["image"])
        match = re.search(r"(Slide%20\d+-\d+%5C[^.]+)", image)
        if match:
            lookup[match.group(1)] = row["id"]
        else:
            unmatched += 1

    print(f"  ✓ {len(lookup)} entries built ({unmatched} unmatched rows in export)")
    return lookup


def get_label_id(file_path, lookup):
    """Extract slide+cell key from file path and look up Label Studio ID."""
    filename    = file_path.replace("\\", "/").split("/")[-1]
    slide_match = re.search(r"_slide(\d+)_(\d+)\.png$", filename)
    if not slide_match:
        return None

    slide_folder = f"Slide%20{slide_match.group(1)}-{slide_match.group(2)}"
    cell_stem    = re.sub(r"_slide\d+_\d+\.png$", "", filename)
    search_key   = f"{slide_folder}%5C{cell_stem}"

    return lookup.get(search_key, None)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img  = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# LOAD ALL IMAGE PATHS
# ─────────────────────────────────────────────
def load_all_paths(data_dir, folders):
    all_paths = []
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: folder not found — {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append({
                    "path":   os.path.join(folder_path, fname),
                    "folder": folder,
                })
    print(f"Total images found: {len(all_paths)}")
    return all_paths

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_classes, checkpoint_path, device):
    model = models.resnet34(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  ✓ Loaded {checkpoint_path}")
    return model

# ─────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, all_path_dicts, config):
    device      = config["device"]
    image_paths = [d["path"] for d in all_path_dicts]
    dataset     = ImagePathDataset(image_paths, transform=infer_transform)
    loader      = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    all_probs, all_paths_out = [], []
    for imgs, paths in loader:
        imgs    = imgs.to(device)
        outputs = model(imgs)
        probs   = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_paths_out.extend(paths)

    return all_probs, all_paths_out

# ─────────────────────────────────────────────
# SAVE CSV HELPER
# ─────────────────────────────────────────────
def save_uncertain_csv(rows, prob_col, output_path, lookup):
    df           = pd.DataFrame(rows)
    df_uncertain = df[df["uncertain"]].copy()
    df_uncertain = df_uncertain.sort_values(prob_col)

    # Append label Studio ID in one pass
    df_uncertain["label_studio_id"] = df_uncertain["file_path"].apply(
        lambda p: get_label_id(p, lookup)
    )

    not_found = df_uncertain["label_studio_id"].isna().sum()
    df_uncertain.to_csv(output_path, index=False)

    print(f"  Total images:          {len(df)}")
    print(f"  Uncertain (0.40–0.60): {len(df_uncertain)}")
    print(f"  IDs matched:           {len(df_uncertain) - not_found}")
    print(f"  IDs not found:         {not_found}")
    print(f"  ✓ Saved {output_path}")

    return df_uncertain

# ─────────────────────────────────────────────
# STAGE 1 — NonMonocyte (0) vs Monocyte (1)
# ─────────────────────────────────────────────
def run_stage1(all_path_dicts, config, lookup):
    print("\n── Stage 1: NonMono vs Monocyte ──")
    model = build_model(
        num_classes=3,
        checkpoint_path=config["stage1_checkpoint"],
        device=config["device"]
    )

    all_probs, all_paths = run_inference(model, all_path_dicts, config)
    folder_map = {d["path"]: d["folder"] for d in all_path_dicts}

    low, high = config["uncertainty_low"], config["uncertainty_high"]
    rows = []
    for path, probs in zip(all_paths, all_probs):
        mono_prob = float(probs[2])  # class 2 = Monocyte
        rows.append({
            "file_path":        path,
            "folder":           folder_map.get(path, ""),
            "mono_probability": round(mono_prob, 4),
            "prediction":       "Monocyte" if mono_prob >= 0.5 else "NonMonocyte",
            "uncertain":        low <= mono_prob <= high,
        })

    df_uncertain = save_uncertain_csv(rows, "mono_probability", config["stage1_output"], lookup)

    print("\n  Uncertain images by folder:")
    for folder, group in df_uncertain.groupby("folder"):
        print(f"    {folder}: {len(group)}")

# ─────────────────────────────────────────────
# STAGE 2 — Unclustered (0) vs Clustered (1)
# ─────────────────────────────────────────────
def run_stage2(all_path_dicts, config, lookup):
    print("\n── Stage 2: Unclustered vs Clustered ──")
    model = build_model(
        num_classes=2,
        checkpoint_path=config["stage2_checkpoint"],
        device=config["device"]
    )

    mono_folders    = {"Monocyte_with_RBC", "Monocyte_without_RBC", "Clustered_cell"}
    mono_path_dicts = [d for d in all_path_dicts if d["folder"] in mono_folders]
    print(f"  Running on {len(mono_path_dicts)} Monocyte images only")

    all_probs, all_paths = run_inference(model, mono_path_dicts, config)
    folder_map = {d["path"]: d["folder"] for d in mono_path_dicts}

    low, high = config["uncertainty_low"], config["uncertainty_high"]
    rows = []
    for path, probs in zip(all_paths, all_probs):
        clust_prob = float(probs[1])
        rows.append({
            "file_path":             path,
            "folder":                folder_map.get(path, ""),
            "clustered_probability": round(clust_prob, 4),
            "prediction":            "Clustered" if clust_prob >= 0.5 else "Unclustered",
            "uncertain":             low <= clust_prob <= high,
        })

    df_uncertain = save_uncertain_csv(rows, "clustered_probability", config["stage2_output"], lookup)

    print("\n  Uncertain images by folder:")
    for folder, group in df_uncertain.groupby("folder"):
        print(f"    {folder}: {len(group)}")

# ─────────────────────────────────────────────
# 3-CLASS — Unclustered (0) / Clustered (1) / NonMono (2)
# ─────────────────────────────────────────────
def run_3class(all_path_dicts, config, lookup):
    print("\n── 3-Class: Unclustered / Clustered / NonMonocyte ──")
    model = build_model(
        num_classes=3,
        checkpoint_path=config["3class_checkpoint"],
        device=config["device"]
    )

    all_probs, all_paths = run_inference(model, all_path_dicts, config)
    folder_map  = {d["path"]: d["folder"] for d in all_path_dicts}
    class_names = ["Unclustered", "Clustered", "NonMonocyte"]

    low, high = config["uncertainty_low"], config["uncertainty_high"]
    rows = []
    for path, probs in zip(all_paths, all_probs):
        max_prob = float(np.max(probs))
        rows.append({
            "file_path":         path,
            "folder":            folder_map.get(path, ""),
            "prob_unclustered":  round(float(probs[0]), 4),
            "prob_clustered":    round(float(probs[1]), 4),
            "prob_nonmono":      round(float(probs[2]), 4),
            "max_probability":   round(max_prob, 4),
            "prediction":        class_names[int(np.argmax(probs))],
            "uncertain":         low <= max_prob <= high,
        })

    df_uncertain = save_uncertain_csv(rows, "max_probability", config["3class_output"], lookup)

    print("\n  Uncertain images by folder:")
    for folder, group in df_uncertain.groupby("folder"):
        print(f"    {folder}: {len(group)}")

    print("\n  Uncertain images by predicted class:")
    for cls, group in df_uncertain.groupby("prediction"):
        print(f"    {cls}: {len(group)}")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running on: {CONFIG['device']}")

    # Build lookup once — reused across all three models
    lookup = build_label_lookup(CONFIG["label_export_csv"])

    # Load all image paths once — reused across all three models
    all_path_dicts = load_all_paths(CONFIG["data_dir"], ALL_FOLDERS)

    run_stage1(all_path_dicts, CONFIG, lookup)
    run_stage2(all_path_dicts, CONFIG, lookup)
    run_3class(all_path_dicts, CONFIG, lookup)

    print("\n── Done ──")
    print(f"  Stage 1: {CONFIG['stage1_output']}")
    print(f"  Stage 2: {CONFIG['stage2_output']}")
    print(f"  3-class: {CONFIG['3class_output']}")