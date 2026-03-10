import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from openpyxl import load_workbook
from datetime import date
from sklearn.metrics import precision_recall_fscore_support

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir": "D:/MMA_LabelledData/Sliced",
    "img_size": 256,
    "batch_size": 32,
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "checkpoint_path": "checkpoints/resnet34_binary.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Top-level binary targets (Monocyte subsampled to match Non-Monocyte)
    "binary_targets": {
        0: 3000,   # Non-Monocyte 
        1: 3000,   # Monocyte
    },

    # Within-Monocyte subtype targets (must sum to binary_targets[1] = 2500)
    "monocyte_subtype_targets": {
        "MCwRBC": 1000,   # boosted from ~802
        "MCwoRBC": 1000,   # subsampled from ~3948
        "Clustered": 1000,   # subsampled from ~5357
    }
}

# ─────────────────────────────────────────────
# LABEL MAPS
# Fine map tracks subtypes for two-level sampling.
# Binary map collapses to Monocyte (1) vs Non-Monocyte (0).
# ─────────────────────────────────────────────
FINE_MAP = {
    "Monocyte_with_RBC":    "MCwRBC",
    "Monocyte_without_RBC": "MCwoRBC",
    "Clustered_cell":       "Clustered",
    "Unusable":             "NonMonocyte",
    "Lymphocyte":           "NonMonocyte",
    "RBC alone":            "NonMonocyte",
}

BINARY_MAP = {
    "MCwRBC":       1,
    "MCwoRBC":      1,
    "Clustered":    1,
    "NonMonocyte":  0,
}

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class BloodCellDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, binary_label) tuples
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_samples(data_dir, fine_map, binary_map):
    """
    Walk subfolders and collect (path, fine_label, binary_label) tuples.
    fine_label is used for two-level sampling.
    binary_label is used for model training.
    """
    samples = []
    for subfolder, fine_label in fine_map.items():
        folder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(folder_path):
            print(f"Warning: folder not found — {folder_path}")
            continue
        binary_label = binary_map[fine_label]
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(folder_path, fname), fine_label, binary_label))
    return samples


def make_weighted_sampler(train_samples, binary_targets, subtype_targets):
    """
    Two-level sampling strategy:
      Level 1 — Balance Monocyte vs Non-Monocyte at binary level
      Level 2 — Within Monocyte, balance subtypes evenly
    """
    # Count natural occurrences per fine label
    fine_counts = {}
    for _, fine_label, _ in train_samples:
        fine_counts[fine_label] = fine_counts.get(fine_label, 0) + 1

    print("\nNatural class counts in training set:")
    for label, count in sorted(fine_counts.items()):
        binary = "Monocyte" if BINARY_MAP.get(label, 0) == 1 else "Non-Monocyte"
        print(f"  {label} ({binary}): {count}")

    # Assign per-sample weights
    sample_weights = np.zeros(len(train_samples), dtype=np.float32)
    for idx, (_, fine_label, _) in enumerate(train_samples):
        if fine_label == "NonMonocyte":
            # Non-Monocyte: weight relative to binary target
            w = binary_targets[0] / fine_counts["NonMonocyte"]
        else:
            # Monocyte subtype: weight relative to subtype target
            w = subtype_targets[fine_label] / fine_counts[fine_label]
        sample_weights[idx] = w

    total_samples = binary_targets[0] + binary_targets[1]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights),
        num_samples=total_samples,
        replacement=True
    )

    print("\nEffective samples per epoch (after sampling):")
    print(f"  Non-Monocyte: {fine_counts.get('NonMonocyte', 0)} → {binary_targets[0]}")
    for subtype, target in subtype_targets.items():
        print(f"  {subtype}: {fine_counts.get(subtype, 0)} → {target}")
    print(f"  Total per epoch: {total_samples}")

    return sampler, fine_counts


def make_dataloaders(config):
    all_samples = load_samples(config["data_dir"], FINE_MAP, BINARY_MAP)

    # Use binary label for stratified splitting to preserve class ratios
    binary_labels = [s[2] for s in all_samples]

    # Step 1: 70% train, 30% temp
    train, temp = train_test_split(
        all_samples, test_size=0.30,
        stratify=binary_labels, random_state=42
    )

    # Step 2: Split temp into 20% test, 10% val (0.333 * 0.30 = 0.10 of total)
    temp_labels = [s[2] for s in temp]
    test, val = train_test_split(
        temp, test_size=0.333,
        stratify=temp_labels, random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train)} ({len(train)/len(all_samples)*100:.1f}%)")
    print(f"  Test:  {len(test)}  ({len(test)/len(all_samples)*100:.1f}%)")
    print(f"  Val:   {len(val)}   ({len(val)/len(all_samples)*100:.1f}%)")

    # Build sampler on training set only
    sampler, fine_counts = make_weighted_sampler(
        train,
        config["binary_targets"],
        config["monocyte_subtype_targets"]
    )

    # Collapse to binary labels for model training
    train_binary = [(path, binary) for path, _, binary in train]
    val_binary   = [(path, binary) for path, _, binary in val]
    test_binary  = [(path, binary) for path, _, binary in test]

    train_ds = BloodCellDataset(train_binary, transform=train_transforms)
    val_ds   = BloodCellDataset(val_binary,   transform=val_transforms)
    test_ds  = BloodCellDataset(test_binary,  transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        sampler=sampler, num_workers=config["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    return train_loader, val_loader, test_loader, fine_counts

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes=2, freeze_backbone=False):
    """
    ResNet34 pretrained on ImageNet.
    freeze_backbone=True: train only the FC head (faster first pass).
    freeze_backbone=False: fine-tune all layers (better final accuracy).
    """
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def save_plots(train_losses, val_losses, train_accs, val_accs,
               true_labels, preds, output_dir="checkpoints"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Loss curve ──
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses,   label="Val Loss",   marker="o", markersize=3)
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print("  ✓ Saved loss_curve.png")

    # ── Accuracy curve ──
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy", marker="o", markersize=3)
    plt.plot(val_accs,   label="Val Accuracy",   marker="o", markersize=3)
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=150)
    plt.close()
    print("  ✓ Saved accuracy_curve.png")

    # ── Confusion matrix heatmap ──
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Monocyte", "Monocyte"],
                yticklabels=["Non-Monocyte", "Monocyte"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  ✓ Saved confusion_matrix.png")

# ─────────────────────────────────────────────
# EXPERIMENT LOGGING
# ─────────────────────────────────────────────
def log_experiment(config, results, log_path="experiment_log.xlsx", notes=""):
    """
    Appends a new run row to the experiment log Excel file.
    results: dict with keys — test_acc, non_mono_prec, non_mono_recall,
             non_mono_f1, mono_prec, mono_recall, mono_f1,
             macro_f1, weighted_f1, tp, tn, fp, fn
    """
    wb = load_workbook(log_path)
    ws = wb.active

    # Find next empty row
    next_row = ws.max_row + 1
    run_num  = next_row - 2  # subtract 2 header rows

    mono_target = sum(config["monocyte_subtype_targets"].values())
    aug_str = "HFlip, VFlip, Rotation(15°), ColorJitter(b=0.2,c=0.2), ImageNet norm"

    row = [
        run_num,
        str(date.today()),
        notes,
        "ResNet34", "ImageNet",
        "Yes" if config.get("freeze_backbone", False) else "No",
        0.4,
        12494,
        70, 10, 20,
        config["binary_targets"][0],
        mono_target,
        config["monocyte_subtype_targets"]["MCwRBC"],
        config["monocyte_subtype_targets"]["MCwoRBC"],
        config["monocyte_subtype_targets"]["Clustered"],
        "Inverse natural frequency",
        "AdamW",
        config["lr"],
        config["weight_decay"],
        config["batch_size"],
        config["num_epochs"],
        f"CosineAnnealingLR (T_max={config['num_epochs']})",
        f"{config['img_size']}x{config['img_size']}",
        aug_str,
        results["test_acc"],
        results["non_mono_prec"],
        results["non_mono_recall"],
        results["non_mono_f1"],
        results["mono_prec"],
        results["mono_recall"],
        results["mono_f1"],
        results["macro_f1"],
        results["weighted_f1"],
        results["tp"],
        results["tn"],
        results["fp"],
        results["fn"],
        results["val_acc"],
    ]

    for col_idx, value in enumerate(row, start=1):
        ws.cell(row=next_row, column=col_idx, value=value)

    wb.save(log_path)
    print(f"  ✓ Run {run_num} logged to {log_path}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train(config):
    device = config["device"]
    print(f"Training on: {device}")
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    train_loader, val_loader, test_loader, fine_counts = make_dataloaders(config)
    model = build_model(num_classes=2, freeze_backbone=False).to(device)

    non_mono_count = fine_counts.get("NonMonocyte", 1)
    mono_count = sum(v for k, v in fine_counts.items() if k != "NonMonocyte")
    class_weights = torch.tensor(
        [1.0 / non_mono_count, 1.0 / mono_count], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # ── Track metrics per epoch ──
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    print("\n── Training ──")
    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config["checkpoint_path"])
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

    # ── Final test evaluation ──
    print("\n── Test Set Evaluation ──")
    model.load_state_dict(torch.load(config["checkpoint_path"]))
    _, test_acc, preds, true_labels = evaluate(
        model, test_loader, criterion, device
    )
    # ── Validation set evaluation ──
    print("\n── Validation Set Evaluation ──")
    _, val_acc_final, val_preds, val_true = evaluate(
        model, val_loader, criterion, device
    )
    print(f"Val Accuracy: {val_acc_final:.4f}\n")
    print(classification_report(
        val_true, val_preds,
        target_names=["Non-Monocyte", "Monocyte"]
    ))
    # -- Test Accuracy --
    print(f"Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(
        true_labels, preds,
        target_names=["Non-Monocyte", "Monocyte"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    # ── Save plots ──
    print("\n── Saving Plots ──")
    plot_dir = os.path.dirname(config["checkpoint_path"])
    save_plots(train_losses, val_losses, train_accs, val_accs,
               true_labels, preds, output_dir=plot_dir)
    
    # ── Log experiment ──
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, preds, labels=[0, 1]
    )
    macro_f1    = (f1[0] + f1[1]) / 2
    weighted_f1 = (f1[0] * 477 + f1[1] * 2022) / 2499  # adjust support counts

    cm = confusion_matrix(true_labels, preds)
    log_experiment(config, {
        "test_acc":        round(test_acc, 4),
        "non_mono_prec":   round(prec[0], 4),
        "non_mono_recall": round(rec[0],  4),
        "non_mono_f1":     round(f1[0],   4),
        "mono_prec":       round(prec[1], 4),
        "mono_recall":     round(rec[1],  4),
        "mono_f1":         round(f1[1],   4),
        "macro_f1":        round(macro_f1,    4),
        "weighted_f1":     round(weighted_f1, 4),
        "tp": int(cm[1,1]), "tn": int(cm[0,0]),
        "fp": int(cm[0,1]), "fn": int(cm[1,0]),
        "val_acc": round(val_acc, 4),
    }, log_path="C:/repos/Blood-Cell-Classification/experiment_log.xlsx",
       notes="")  # <-- fill in notes per run

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train(CONFIG)