"""
Active Learning Driver (Final Fixed Version)

This script:
- Scans stitched folder for all unique images
- Scans labelled dataset folders for labeled images
- Maintains AL state (labeled, unlabeled, queried)
- Builds correct 3‑tuple training samples:
      (full_path, training_label, fine_label)
- Trains ConvNeXt on labeled samples
- Scores unlabeled samples by uncertainty
- Selects top‑K uncertain samples
- Copies them into active_learning/to_label/
"""

import os
import json
import shutil
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from Convnext_MvNM import val_transforms

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STITCHED_DIR = r"D:/MMA_batch1/contrast_1.0_Sliced/Stitched Image names"
LABELLED_ROOT = r"D:/MMA_LabelledData/Sliced"
STATE_JSON = r"D:/MMA_batch1/active_learning_state.json"

QUERY_PER_ROUND = 128
N_ROUNDS = 5

TO_LABEL_DIR = "active_learning/to_label"
os.makedirs(TO_LABEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────

FINE_MAP = {
    "Monocyte_with_RBC":    "MCwRBC",
    "Monocyte_without_RBC": "MCwoRBC",
    "Clustered_cell":       "Clustered",
    "Unusable":             "Unusable",
    "Lymphocyte":           "Lymphocyte",
    "RBC alone":            "RBCalone",
}

LABEL_MAP = {
    "Unusable":   0,
    "Lymphocyte": 0,
    "RBCalone":   1,
    "MCwRBC":     2,
    "MCwoRBC":    2,
    "Clustered":  2,
}

class UnlabeledDataset(Dataset):
    def __init__(self, fnames):
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(STITCHED_DIR, fname)).convert("RGB")
        return val_transforms(img), fname

# ─────────────────────────────────────────────
# MANIFEST
# ─────────────────────────────────────────────

def build_manifest(stitched_dir: str) -> List[str]:
    return sorted([
        f for f in os.listdir(stitched_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

# ─────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────

def init_or_load_state(manifest: List[str], state_path: str):
    if os.path.isfile(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        state.setdefault("labeled", [])
        state.setdefault("unlabeled", [])
        state.setdefault("queried", [])
        return state

    return {
        "labeled": [],
        "unlabeled": manifest.copy(),
        "queried": [],
    }

def save_state(state, path):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

# ─────────────────────────────────────────────
# AUTO‑SYNC LABELED/UNLABELED
# ─────────────────────────────────────────────

def auto_update_state_from_labelled_folders(state, stitched_dir, labelled_root):
    all_files = {
        f for f in os.listdir(stitched_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    labeled_files = set()
    for class_folder in os.listdir(labelled_root):
        class_path = os.path.join(labelled_root, class_folder)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                labeled_files.add(fname)

    # Exclude queried from unlabeled
    queried = set(state.get("queried", []))

    unlabeled_files = all_files - labeled_files - queried

    state["labeled"] = sorted(list(labeled_files))
    state["unlabeled"] = sorted(list(unlabeled_files))
    state["queried"] = sorted(list(queried))

    return state

# ─────────────────────────────────────────────
# LABEL RESOLUTION (3‑TUPLE)
# ─────────────────────────────────────────────

def resolve_label_for_filename(fname: str) -> Tuple[str, int, str]:
    """
    Return (full_path, training_label, fine_label)
    """
    for class_folder in os.listdir(LABELLED_ROOT):
        class_path = os.path.join(LABELLED_ROOT, class_folder)
        if not os.path.isdir(class_path):
            continue

        candidate = os.path.join(class_path, fname)
        if os.path.isfile(candidate):

            if class_folder not in FINE_MAP:
                raise KeyError(f"Class folder '{class_folder}' missing in FINE_MAP")

            fine_label = FINE_MAP[class_folder]
            training_label = LABEL_MAP[fine_label]

            return candidate, training_label, fine_label

    raise FileNotFoundError(f"{fname} not found in labeled folders")

def build_train_samples_from_state(state):
    samples = []
    for fname in state["labeled"]:
        full_path, training_label, fine_label = resolve_label_for_filename(fname)
        samples.append((full_path, training_label, fine_label))
    return samples

# ─────────────────────────────────────────────
# TRAINING HOOK
# ─────────────────────────────────────────────

def train_model_on_labeled(train_samples):
    import copy
    import torch
    from Convnext_MvNM import train, DEFAULT_CONFIG, build_model, build_samples_from_filenames

    SPLIT_JSON = r"D:/MMA_LabelledData/splits/dataset_splits.json"
    DATA_DIR = r"D:/MMA_LabelledData/Sliced"

    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)

    val_samples = build_samples_from_filenames(split["val"], DATA_DIR)
    test_samples = build_samples_from_filenames(split["test"], DATA_DIR)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["custom_train_samples"] = train_samples
    cfg["custom_val_samples"] = val_samples
    cfg["custom_test_samples"] = test_samples
    cfg["checkpoint_path"] = "checkpoints_stage1/al_round_best.pth"

    train(cfg, notes="ActiveLearningRound")

    model = build_model(
        architecture=cfg["architecture"],
        num_classes=3,
        freeze_backbone=False
    )
    model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location="cpu"))
    model.eval()
    return model

# ─────────────────────────────────────────────
# UNCERTAINTY SCORING
# ─────────────────────────────────────────────

def score_unlabeled_samples(model, unlabeled_fnames):
    
    loader = DataLoader(UnlabeledDataset(unlabeled_fnames),
                        batch_size=32, shuffle=False, num_workers=4)

    model.eval()
    device = next(model.parameters()).device
    scores = {}

    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            uncertainty = 1 - probs.max(dim=1).values
            for fname, u in zip(fnames, uncertainty.cpu().numpy()):
                scores[fname] = float(u)

    return scores

def select_most_uncertain(unlabeled, scores, k):
    ranked = sorted(unlabeled, key=lambda f: scores[f], reverse=True)
    return ranked[:k]

# ─────────────────────────────────────────────
# ACTIVE LEARNING LOOP
# ─────────────────────────────────────────────

def run_active_learning():
    manifest = build_manifest(STITCHED_DIR)
    state = init_or_load_state(manifest, STATE_JSON)

    state = auto_update_state_from_labelled_folders(state, STITCHED_DIR, LABELLED_ROOT)
    save_state(state, STATE_JSON)

    print(f"Labeled: {len(state['labeled'])}, Unlabeled: {len(state['unlabeled'])}")

    for round_idx in range(N_ROUNDS):
        print(f"\n── Active Learning Round {round_idx+1}/{N_ROUNDS} ──")

        if not state["labeled"]:
            print("No labeled data found. Add labeled images first.")
            return

        train_samples = build_train_samples_from_state(state)
        print(f"Training on {len(train_samples)} samples")

        model = train_model_on_labeled(train_samples)

        unlabeled = state["unlabeled"]
        scores = score_unlabeled_samples(model, unlabeled)

        k = min(QUERY_PER_ROUND, len(unlabeled))
        selected = select_most_uncertain(unlabeled, scores, k)

        for fname in selected:
            shutil.copy(os.path.join(STITCHED_DIR, fname),
                        os.path.join(TO_LABEL_DIR, fname))

        print(f"Selected {len(selected)} images → {TO_LABEL_DIR}")

        state["queried"].extend(selected)
        save_state(state, STATE_JSON)

        print("Label these images, move them into correct folders, then rerun.")
        break

    print("\nActive Learning finished.")

# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_active_learning()
