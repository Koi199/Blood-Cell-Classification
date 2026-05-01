"""
kfold_trainer.py
────────────────────────────────────────────────────────────────────────────
Generic donor-grouped k-fold trainer for cascaded classifiers.

Designed so that each classifier in your cascade only requires a new CONFIG
block — the training, evaluation, and reporting logic is shared.

FOLDER STRUCTURE EXPECTED:
    data_dir/
    ├── ClassA/
    │   ├── slide1-1/
    │   │   ├── img001.png
    │   │   └── ...
    │   └── slide2-1/
    ├── ClassB/
    │   ├── slide1-1/
    │   └── ...
    └── ...

If a class folder maps multiple raw folder names to one label (e.g. collapsing
RBC_1 through RBC_5), define that in FOLDER_MAP. Otherwise, each folder name
is its own class.

USAGE — configure and run a classifier:
    python kfold_trainer.py --config clustered_binary

    Or import and call directly:
        from kfold_trainer import train_kfold, CONFIGS
        results, summary = train_kfold(CONFIGS["clustered_binary"])
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from datetime import date

import sys
sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Logging")
from Logger import (
    setup_mlflow, start_run, log_params, log_epoch,
    log_results, log_artifacts, log_confusion_matrix, end_run
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS
# Add one entry per classifier in your cascade.
#
# Required keys:
#   name            str   — used for MLflow run names, checkpoint filenames
#   data_dir        str   — root folder containing class subfolders
#   folder_map      dict  — maps each subfolder name → integer label
#                           collapse multiple folders to the same int to merge
#   class_names     list  — display name for each label (index = label int)
#   checkpoint_dir  str   — where to save per-fold .pth files
#
# Optional keys (fall back to TRAINING_DEFAULTS if omitted):
#   img_size, batch_size, num_epochs, early_stopping_patience,
#   lr, weight_decay, num_workers, n_splits, architecture
#   subclass_targets  dict  — {display_name: target_n} for WeightedRandomSampler
#                             omit to use natural class frequencies
# ─────────────────────────────────────────────────────────────────────────────

# ── Sampler tuning knobs ──────────────────────────────────────────────────────
# Adjust these to experiment with class balance without touching the configs.

# Stage 1 — usability classifier
_stage1_mono_total = 1800   # total budget shared across MCwRBC, MCwoRBC, Clustered

# Stage 2 — clustered vs unclustered
_stage2_target     = 1200   # per-class target for Clustered and Unclustered

# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {

    # ── Stage 1: Usability filter ─────────────────────────────────────────────
    # Input : all raw sliced cells
    # Output: Unusable (0)  vs  Usable (1)
    # RBCalone is collapsed into Unusable; MCwRBC/MCwoRBC/Clustered → Usable
    "stage1_usability": {
        "name":        "stage1_usability",
        "data_dir":    "D:/MMA_LabelledData/training_perslide",
        "folder_map": {
            "Unusable":             0,   # Unusable
            "RBC alone":            0,   # collapsed → Unusable
            "Monocyte_with_RBC":    1,   # Usable
            "Monocyte_without_RBC": 1,   # Usable
            "Clustered_cell":       1,   # Usable
        },
        "class_names":    ["Unusable", "Usable"],
        "checkpoint_dir": "C:/repos/Blood-Cell-Classification/checkpoints_stage1",
        "subclass_targets": {
            "Unusable": 1800,
            "Usable":   _stage1_mono_total,
        },
    },

    # ── Stage 2: Clustered vs Unclustered ─────────────────────────────────────
    # Input : Usable cells only (output of Stage 1)
    # Output: Unclustered (0)  vs  Clustered (1)
    # MCwRBC and MCwoRBC are collapsed into Unclustered
    "stage2_clustered": {
        "name":        "stage2_clustered",
        "data_dir":    "D:/MMA_LabelledData/training_perslide",
        "folder_map": {
            "Monocyte_with_RBC":    0,   # Unclustered
            "Monocyte_without_RBC": 0,   # Unclustered (collapsed)
            "Clustered_cell":       1,   # Clustered
        },
        "class_names":    ["Unclustered", "Clustered"],
        "checkpoint_dir": "C:/repos/Blood-Cell-Classification/checkpoints_stage2",
        "subclass_targets": {
            "Unclustered": _stage2_target,
            "Clustered":   _stage2_target,
        },
    },

    # ── Unclustered RBC binary ────────────────────────────────────────────────
    # Input : Unclustered monocytes only (output of Stage 2 = 0)
    # Output: No_RBC (0)  vs  Has_RBC (1)
    "unclustered_binary": {
        "name":        "unclustered_binary",
        "data_dir":    "D:/MMA_LabelledData/training_perslide/Unclustered_RBCCount",
        "folder_map": {
            "RBC_0": 0,   # No RBC
            "RBC_1": 1,   # Has RBC (collapsed)
            "RBC_2": 1,
            "RBC_3": 1,
            "RBC_4": 1,
            "RBC_5": 1,
        },
        "class_names":    ["No_RBC", "Has_RBC"],
        "checkpoint_dir": "C:/repos/Blood-Cell-Classification/checkpoints_rbc_binary",
        "subclass_targets": {
            "No_RBC":  400,
            "Has_RBC": 400,
        },
    },

    # ── Clustered RBC binary ──────────────────────────────────────────────────
    # Input : Clustered monocytes only (output of Stage 2 = 1)
    # Output: No_RBC (0)  vs  Has_RBC (1)  vs  RBC_alone (2)
    "clustered_binary": {
        "name":        "clustered_binary",
        "data_dir":    "D:/MMA_LabelledData/training_perslide/clustered_RBCCount",
        "folder_map": {
            "RBC_0":     0,   # No RBC
            "RBC_1":     1,   # Has RBC (collapsed)
            "RBC_2":     1,
            "RBC_3":     1,
            "RBC_4":     1,
            "RBC_5":     1,
            "RBC_alone": 2,   # RBC without monocyte
        },
        "class_names":    ["No_RBC", "Has_RBC", "RBC_alone"],
        "checkpoint_dir": "C:/repos/Blood-Cell-Classification/checkpoints_rbc_clustered_binary",
        "subclass_targets": {
            "No_RBC":    800,
            "Has_RBC":   800,
            "RBC_alone": 400,
        },
    },

    # ── Add more classifiers here following the same pattern ──────────────────
    # "my_new_classifier": {
    #     "name":        "my_new_classifier",
    #     "data_dir":    "D:/MMA_LabelledData/MyData",
    #     "folder_map": {
    #         "FolderA": 0,
    #         "FolderB": 1,
    #         "FolderC": 1,   # collapsed with FolderB
    #         "FolderD": 2,
    #     },
    #     "class_names":    ["ClassA", "ClassB_C", "ClassD"],
    #     "checkpoint_dir": "checkpoints_my_new_classifier",
    # },
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DEFAULTS
# Any config key not specified above falls back to these values.
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DEFAULTS = {
    "img_size":                256,
    "batch_size":              32,
    "num_epochs":              20,
    "early_stopping_patience": 5,
    "lr":                      5e-5,
    "weight_decay":            1e-4,
    "num_workers":             4,
    "architecture":            "convnext_tiny",
    "n_splits":                5,
}

VIT_ARCHITECTURES = {"vit_b16", "vit_b32", "vit_l16"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_config(user_config: dict) -> dict:
    """Merge user config with TRAINING_DEFAULTS. User values take priority."""
    cfg = copy.deepcopy(TRAINING_DEFAULTS)
    cfg.update(user_config)
    return cfg


def get_img_size(architecture: str, config_img_size: int) -> int:
    if architecture.lower() in VIT_ARCHITECTURES:
        print(f"  ViT detected — overriding img_size to 224")
        return 224
    return config_img_size


def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(architecture: str, num_classes: int, freeze_backbone: bool = False):
    arch = architecture.lower()
    weight_map = {
        "resnet18":       (models.resnet18,       models.ResNet18_Weights.DEFAULT),
        "resnet34":       (models.resnet34,       models.ResNet34_Weights.DEFAULT),
        "resnet50":       (models.resnet50,       models.ResNet50_Weights.DEFAULT),
        "resnet101":      (models.resnet101,      models.ResNet101_Weights.DEFAULT),
        "resnet152":      (models.resnet152,      models.ResNet152_Weights.DEFAULT),
        "efficientnet_b0":(models.efficientnet_b0,models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1":(models.efficientnet_b1,models.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b4":(models.efficientnet_b4,models.EfficientNet_B4_Weights.DEFAULT),
        "convnext_tiny":  (models.convnext_tiny,  models.ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_base":  (models.convnext_base,  models.ConvNeXt_Base_Weights.DEFAULT),
        "vit_b16":        (models.vit_b_16,       models.ViT_B_16_Weights.DEFAULT),
    }
    if arch not in weight_map:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Supported: {list(weight_map.keys())}")

    model_fn, weights = weight_map[arch]
    model = model_fn(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
    elif arch.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
    elif arch.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif arch == "vit_b16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture:     {architecture}")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class BloodCellDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples: list of (path, label)
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(data_dir: str, folder_map: dict, class_names: list) -> list:
    """
    Walk data_dir/FolderName/donor_id/image.png
    Returns list of (path, class_display_name, label_int, donor_id)

    folder_map  — e.g. {"RBC_0": 0, "RBC_1": 1, "RBC_2": 1}
    class_names — e.g. ["No_RBC", "Has_RBC"] — index = label int
    """
    # Build reverse map: label_int → display name (first class_name with that label)
    label_to_display = {i: class_names[i] for i in range(len(class_names))}

    samples = []
    for folder_name, label in folder_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  Warning: folder not found — {folder_path}")
            continue

        count = 0
        for donor_id in os.listdir(folder_path):
            donor_path = os.path.join(folder_path, donor_id)
            if not os.path.isdir(donor_path):
                continue
            for fname in os.listdir(donor_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((
                        os.path.join(donor_path, fname),
                        label_to_display[label],   # display name for sampler
                        label,                     # integer label for model
                        donor_id,                  # donor group key
                    ))
                    count += 1

        display = label_to_display[label]
        print(f"  {folder_name} → {display} (label {label}): {count} images")

    return samples


def make_weighted_sampler(train_samples: list, subclass_targets: dict):
    """
    Build a WeightedRandomSampler from train_samples.
    subclass_targets — {display_name: target_n} or None to use equal weighting.
    """
    # Count by display name
    class_counts = {}
    for _, display_name, _, _ in train_samples:
        class_counts[display_name] = class_counts.get(display_name, 0) + 1

    # If no targets provided, balance all classes equally
    if not subclass_targets:
        max_count       = max(class_counts.values())
        subclass_targets = {name: max_count for name in class_counts}

    print("\n  Weighted sampler:")
    for name, count in sorted(class_counts.items()):
        target = subclass_targets.get(name, count)
        print(f"    {name}: {count} → {target} ({target/count:.2f}x)")

    weights = np.zeros(len(train_samples), dtype=np.float32)
    for idx, (_, display_name, _, _) in enumerate(train_samples):
        natural       = class_counts[display_name]
        target        = subclass_targets.get(display_name, natural)
        weights[idx]  = target / natural

    total_samples = sum(subclass_targets.values())
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights),
        num_samples=total_samples,
        replacement=True,
    )
    print(f"    Effective samples/epoch: {total_samples}")
    return sampler


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs     = model(imgs)
        loss        = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs      = model(imgs)
        total_loss  += criterion(outputs, labels).item() * imgs.size(0)
        preds        = outputs.argmax(1)
        correct     += (preds == labels).sum().item()
        total       += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def save_fold_plots(train_losses, val_losses, train_accs, val_accs,
                    true_labels, preds, class_names, output_dir, fold):
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"fold{fold}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses,   label="Test Loss",  marker="o", markersize=3)
    plt.title(f"Fold {fold} — Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{prefix}_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Acc", marker="o", markersize=3)
    plt.plot(val_accs,   label="Test Acc",  marker="o", markersize=3)
    plt.title(f"Fold {fold} — Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{prefix}_accuracy.png", dpi=150); plt.close()

    n = len(class_names)
    cm = confusion_matrix(true_labels, preds, labels=list(range(n)))
    plt.figure(figsize=(max(6, n + 2), max(5, n + 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Fold {fold} — Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrix.png", dpi=150); plt.close()

    print(f"  ✓ Saved fold {fold} plots to {output_dir}")
    return [f"{prefix}_loss.png", f"{prefix}_accuracy.png",
            f"{prefix}_confusion_matrix.png"]


def save_summary_plot(fold_results, metric_keys, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    folds  = [r["fold"] for r in fold_results]
    n_cols = len(metric_keys)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_keys):
        values = [r[metric] for r in fold_results]
        ax.bar(folds, values, color="steelblue", alpha=0.8)
        ax.axhline(np.mean(values), color="red", linestyle="--",
                   label=f"Mean: {np.mean(values):.3f}")
        ax.set_title(metric)
        ax.set_xlabel("Fold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle(f"{name} — K-Fold Summary", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "kfold_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved k-fold summary plot to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_kfold_excel(cfg, fold_results, summary,
                    log_path="C:/repos/Blood-Cell-Classification/experiment_log.xlsx",
                    notes=""):
    """Appends one summary row + one row per fold to the experiment log."""
    try:
        wb = load_workbook(log_path)
        ws = wb.active
    except FileNotFoundError:
        print(f"  Warning: Excel log not found at {log_path}, skipping.")
        return

    class_names = cfg["class_names"]

    def make_row(label, acc, per_class_f1, macro_f1, weighted_f1=None):
        row = [
            label, str(date.today()), notes,
            cfg.get("architecture", "convnext_tiny"),
            cfg["name"], cfg["n_splits"],
            acc,
        ]
        for f1 in per_class_f1:
            row.append(f1)
        row.append(macro_f1)
        if weighted_f1 is not None:
            row.append(weighted_f1)
        return row

    # Summary row
    mean_acc     = summary["test_acc"][0]
    mean_macro   = summary["macro_f1"][0]
    per_class_f1 = [summary.get(f"{n}_f1", (0, 0))[0] for n in class_names]
    ws.append(make_row("MEAN", mean_acc, per_class_f1, mean_macro))

    # Per-fold rows
    for r in fold_results:
        per_f1 = [r.get(f"{n}_f1", 0) for n in class_names]
        ws.append(make_row(
            f"Fold {r['fold']} ({r['test_donors']})",
            r["test_acc"], per_f1, r["macro_f1"]
        ))

    wb.save(log_path)
    print(f"  ✓ K-fold results logged to {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN K-FOLD TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_kfold(user_config: dict, notes: str = ""):
    """
    Run donor-grouped stratified k-fold cross validation.

    Args:
        user_config — one of the dicts from CONFIGS, or a custom dict.
                      Only keys that differ from TRAINING_DEFAULTS need
                      to be specified (plus the required keys).
        notes       — logged to MLflow and Excel.

    Returns:
        fold_results — list of per-fold metric dicts
        summary      — {metric: (mean, std)} across folds
    """
    cfg         = resolve_config(user_config)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch        = cfg["architecture"]
    img_size    = get_img_size(arch, cfg["img_size"])
    n_splits    = cfg["n_splits"]
    class_names = cfg["class_names"]
    num_classes = len(class_names)
    name        = cfg["name"]

    train_tf, val_tf = make_transforms(img_size)

    print(f"\n{'='*60}")
    print(f"  Classifier : {name}")
    print(f"  Classes    : {class_names}")
    print(f"  Device     : {device}")
    print(f"  Arch       : {arch}  |  img_size: {img_size}")
    print(f"  Folds      : {n_splits}")
    print(f"{'='*60}")

    # ── Load all samples ──
    print("\nLoading samples:")
    all_samples = load_samples(cfg["data_dir"], cfg["folder_map"], class_names)
    print(f"  Total images loaded: {len(all_samples)}")

    X      = np.array([s[0] for s in all_samples])   # paths
    y      = np.array([s[2] for s in all_samples])   # int labels
    groups = np.array([s[3] for s in all_samples])   # donor IDs

    # ── Donor summary ──
    unique_donors = np.unique(groups)
    print(f"\nDonors found ({len(unique_donors)}): {unique_donors.tolist()}")
    for donor in unique_donors:
        mask   = groups == donor
        counts = {class_names[i]: int((y[mask] == i).sum()) for i in range(num_classes)}
        print(f"  {donor}: {counts}")

    # ── K-Fold split ──
    sgkf         = StratifiedGroupKFold(n_splits=n_splits)
    fold_results = []

    setup_mlflow(f"{name}_KFold")

    for fold, (train_idx, test_idx) in enumerate(
            sgkf.split(X, y, groups), start=1):

        print(f"\n{'─'*60}")
        print(f"  FOLD {fold}/{n_splits}")
        print(f"  Train donors : {np.unique(groups[train_idx]).tolist()}")
        print(f"  Test donors  : {np.unique(groups[test_idx]).tolist()}")

        train_samples = [all_samples[i] for i in train_idx]
        test_samples  = [all_samples[i] for i in test_idx]

        # Class distribution in test fold
        test_y = y[test_idx]
        for i, cname in enumerate(class_names):
            print(f"    Test — {cname}: {(test_y == i).sum()} images")

        # ── DataLoaders ──
        subclass_targets = cfg.get("subclass_targets", None)
        sampler = make_weighted_sampler(train_samples, subclass_targets)

        train_loader = DataLoader(
            BloodCellDataset([(s[0], s[2]) for s in train_samples], transform=train_tf),
            batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=cfg["num_workers"], pin_memory=True,
        )
        test_loader = DataLoader(
            BloodCellDataset([(s[0], s[2]) for s in test_samples], transform=val_tf),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )

        # ── Model ──
        model     = build_model(arch, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["num_epochs"]
        )

        checkpoint_path = os.path.join(cfg["checkpoint_dir"], f"{name}_fold{fold}.pth")
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

        start_run(run_name=f"{name}_fold{fold}_{notes}" if notes else f"{name}_fold{fold}",
                  notes=notes)
        log_params(cfg)

        # ── Training loop ──
        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        best_acc                 = 0.0
        epochs_no_improve        = 0

        for epoch in range(1, cfg["num_epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc, _, _ = evaluate(
                model, test_loader, criterion, device
            )
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(test_loss)
            train_accs.append(train_acc)
            val_accs.append(test_acc)

            print(f"    Epoch {epoch:02d}/{cfg['num_epochs']} | "
                  f"Train {train_loss:.4f}/{train_acc:.4f} | "
                  f"Test  {test_loss:.4f}/{test_acc:.4f}")

            log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)

            if test_acc > best_acc:
                best_acc          = test_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"      ✓ Best model saved (acc={test_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg["early_stopping_patience"]:
                    print(f"      Early stopping at epoch {epoch}")
                    break

        # ── Evaluate best model ──
        model.load_state_dict(torch.load(checkpoint_path))
        _, test_acc, preds, true_labels = evaluate(model, test_loader, criterion, device)

        print(f"\n── Fold {fold} Results ──")
        print(classification_report(true_labels, preds,
                                    target_names=class_names, zero_division=0))

        # Per-class metrics
        n_labels = list(range(num_classes))
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, preds, labels=n_labels, zero_division=0
        )
        _, _, f1m, _ = precision_recall_fscore_support(
            true_labels, preds, average="macro", zero_division=0
        )
        _, _, f1w, _ = precision_recall_fscore_support(
            true_labels, preds, average="weighted", zero_division=0
        )

        fold_result = {
            "fold":        fold,
            "test_donors": np.unique(groups[test_idx]).tolist(),
            "test_acc":    round(test_acc, 4),
            "macro_f1":    round(float(f1m), 4),
            "weighted_f1": round(float(f1w), 4),
        }
        for i, cname in enumerate(class_names):
            fold_result[f"{cname}_prec"]   = round(float(prec[i]), 4)
            fold_result[f"{cname}_recall"] = round(float(rec[i]),  4)
            fold_result[f"{cname}_f1"]     = round(float(f1[i]),   4)

        fold_results.append(fold_result)

        # Save plots and artifacts
        artifact_paths = save_fold_plots(
            train_losses, val_losses, train_accs, val_accs,
            true_labels, preds, class_names, cfg["checkpoint_dir"], fold
        )
        log_results(fold_result)
        log_confusion_matrix(true_labels, preds, class_names, artifact_paths[2])
        log_artifacts(artifact_paths + [checkpoint_path])
        end_run()

    # ── Aggregate summary ──
    print(f"\n{'='*60}")
    print(f"  K-FOLD SUMMARY — {name}")
    print(f"{'='*60}")

    metric_keys = ["test_acc", "macro_f1"] + [f"{c}_f1" for c in class_names]
    summary     = {}

    for metric in metric_keys:
        values         = [r[metric] for r in fold_results]
        mean, std      = float(np.mean(values)), float(np.std(values))
        summary[metric] = (mean, std)
        print(f"  {metric:30s}: {mean:.4f} ± {std:.4f}")

    print("\n  Per-fold breakdown:")
    for r in fold_results:
        class_f1s = "  ".join(
            f"{c}={r[f'{c}_f1']:.3f}" for c in class_names
        )
        print(f"    Fold {r['fold']} {r['test_donors']}: "
              f"acc={r['test_acc']:.4f}  macro_f1={r['macro_f1']:.4f}  |  {class_f1s}")

    # Summary plot
    summary_plot = save_summary_plot(
        fold_results, ["test_acc", "macro_f1"] + [f"{c}_f1" for c in class_names],
        cfg["checkpoint_dir"], name
    )

    log_kfold_excel(cfg, fold_results, summary, notes=notes)

    return fold_results, summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donor-grouped k-fold trainer")
    parser.add_argument(
        "--config", type=str, required=True,
        choices=list(CONFIGS.keys()),
        help="Which classifier config to run"
    )
    parser.add_argument("--notes", type=str, default="", help="Run notes")
    parser.add_argument("--n_splits", type=int, default=None,
                        help="Override number of folds")
    args = parser.parse_args()

    cfg = copy.deepcopy(CONFIGS[args.config])
    if args.n_splits is not None:
        cfg["n_splits"] = args.n_splits

    results, summary = train_kfold(cfg, notes=args.notes)