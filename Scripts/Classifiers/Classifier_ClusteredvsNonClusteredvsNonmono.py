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
from openpyxl import load_workbook
from datetime import date
from sklearn.metrics import precision_recall_fscore_support

import sys
sys.path.append("C:/repos/Blood-Cell-Classification/Scripts/Logging")

from Logger import setup_mlflow, start_run, log_params, log_epoch, log_results, log_artifacts, log_confusion_matrix, end_run

# ─────────────────────────────────────────────
# CONFIG
#
# Single 3-class classifier — no cascade:
#   0 = Unclustered  (MCwRBC + MCwoRBC)
#   1 = Clustered
#   2 = NonMonocyte  (Unusable + Lymphocyte + RBC alone)
#
# Sampling targets — equal balance across all 3 classes:
#   Unusable   (~1241) → 600  subsampled
#   Lymphocyte  (~153) → 300  ~2x oversample
#   RBCalone    (~273) → 300  ~1.1x natural
#   MCwRBC      (~515) → 900  ~1.7x oversample
#   MCwoRBC    (~2764) → 900  subsampled
#   Clustered  (~3748) → 1800 subsampled
#   ─────────────────────────────────────────
#   NonMono total:    1200 per epoch
#   Unclustered total: 1800 per epoch
#   Clustered total:   1800 per epoch
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir": "D:/MMA_LabelledData/Sliced",
    "img_size": 256,
    "batch_size": 32,
    "num_epochs": 50,
    "early_stopping_patience": 7,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "checkpoint_path": "checkpoints_3class/resnet34_3class.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "classification_threshold": 0.5,  # used for binary cascade comparison only

    "subclass_targets": {
        # NonMonocyte subclasses — doesnt sum to 1800
        "Unusable":   1000,
        "Lymphocyte": 150,
        "RBCalone":   400,
        # Unclustered subclasses — sum to 1800
        "MCwRBC":     900,
        "MCwoRBC":    900,
        # Clustered
        "Clustered":  1800,
    }
}

# ─────────────────────────────────────────────
# LABEL MAPS
#
# FINE_MAP:  folder name → fine label
# LABEL_MAP: fine label  → training class
#   0 = Unclustered
#   1 = Clustered
#   2 = NonMonocyte
# ─────────────────────────────────────────────
FINE_MAP = {
    "Monocyte_with_RBC":    "MCwRBC",
    "Monocyte_without_RBC": "MCwoRBC",
    "Clustered_cell":       "Clustered",
    "Unusable":             "Unusable",
    "Lymphocyte":           "Unusable",
    "RBC alone":            "RBCalone",
}

LABEL_MAP = {
    "MCwRBC":     0,  # Unclustered
    "MCwoRBC":    0,  # Unclustered
    "Clustered":  1,  # Clustered
    "Unusable":   2,  # NonMonocyte
    "RBCalone":   2,  # NonMonocyte
}

CLASS_NAMES = ["Unclustered", "Clustered", "NonMonocyte"]

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class BloodCellDataset(Dataset):
    def __init__(self, samples, transform=None):
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
def load_samples(data_dir, fine_map, label_map):
    samples = []
    for subfolder, fine_label in fine_map.items():
        folder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(folder_path):
            print(f"Warning: folder not found — {folder_path}")
            continue
        training_label = label_map[fine_label]
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((
                    os.path.join(folder_path, fname),
                    fine_label,
                    training_label
                ))
    return samples


def make_weighted_sampler(train_samples, subclass_targets):
    fine_counts = {}
    for _, fine_label, _ in train_samples:
        fine_counts[fine_label] = fine_counts.get(fine_label, 0) + 1

    print("\nNatural class counts in training set:")
    for label, count in sorted(fine_counts.items()):
        target = subclass_targets.get(label, count)
        dup = target / count
        class_name = CLASS_NAMES[LABEL_MAP[label]]
        print(f"  {label} ({class_name}): {count} → {target} ({dup:.2f}x)")

    sample_weights = np.zeros(len(train_samples), dtype=np.float32)
    for idx, (_, fine_label, _) in enumerate(train_samples):
        natural = fine_counts[fine_label]
        target  = subclass_targets.get(fine_label, natural)
        sample_weights[idx] = target / natural

    total_samples = sum(subclass_targets.values())
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights),
        num_samples=total_samples,
        replacement=True
    )

    nonmono     = subclass_targets.get("Unusable", 0) + subclass_targets.get("Lymphocyte", 0) + subclass_targets.get("RBCalone", 0)
    unclustered = subclass_targets.get("MCwRBC", 0) + subclass_targets.get("MCwoRBC", 0)
    clustered   = subclass_targets.get("Clustered", 0)

    print(f"\nEffective samples per epoch:")
    print(f"  NonMonocyte  (class 2): {nonmono}")
    print(f"  Unclustered  (class 0): {unclustered}")
    print(f"  Clustered    (class 1): {clustered}")
    print(f"  Total:                  {total_samples}")

    return sampler, fine_counts


def make_dataloaders(config):
    all_samples = load_samples(config["data_dir"], FINE_MAP, LABEL_MAP)

    strat_labels = [s[2] for s in all_samples]

    train, temp = train_test_split(
        all_samples, test_size=0.30,
        stratify=strat_labels, random_state=42
    )
    temp_labels = [s[2] for s in temp]
    test, val = train_test_split(
        temp, test_size=0.333,
        stratify=temp_labels, random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train)} ({len(train)/len(all_samples)*100:.1f}%)")
    print(f"  Test:  {len(test)}  ({len(test)/len(all_samples)*100:.1f}%)")
    print(f"  Val:   {len(val)}   ({len(val)/len(all_samples)*100:.1f}%)")

    sampler, fine_counts = make_weighted_sampler(train, config["subclass_targets"])

    train_ds = BloodCellDataset(
        [(path, label) for path, _, label in train],
        transform=train_transforms
    )
    val_ds = BloodCellDataset(
        [(path, label) for path, _, label in val],
        transform=val_transforms
    )
    test_ds = BloodCellDataset(
        [(path, label) for path, _, label in test],
        transform=val_transforms
    )

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
# MODEL — 3 output classes
# ─────────────────────────────────────────────
def build_model(num_classes=3, freeze_backbone=False):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
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
# EVALUATE — 3-class accuracy
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def save_plots(train_losses, val_losses, train_accs, val_accs,
               true_labels, preds, output_dir="checkpoints_3class"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses,   label="Val Loss",   marker="o", markersize=3)
    plt.title("3-Class — Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve_3class.png"), dpi=150)
    plt.close()
    print("  ✓ Saved loss_curve_3class.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy", marker="o", markersize=3)
    plt.plot(val_accs,   label="Val Accuracy",   marker="o", markersize=3)
    plt.title("3-Class — Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve_3class.png"), dpi=150)
    plt.close()
    print("  ✓ Saved accuracy_curve_3class.png")

    cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title("3-Class — Confusion Matrix (%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    for text in plt.gca().texts:
        text.set_text(text.get_text() + "%")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_3class.png"), dpi=150)
    plt.close()
    print("  ✓ Saved confusion_matrix_3class.png")

# ─────────────────────────────────────────────
# EXPERIMENT LOGGING
# ─────────────────────────────────────────────
def log_experiment(config, results, log_path="experiment_log.xlsx", notes=""):
    wb = load_workbook(log_path)
    ws = wb.active

    next_row = ws.max_row + 1
    run_num  = next_row - 2

    aug_str = "HFlip, VFlip, Rotation(15°), ColorJitter(b=0.2,c=0.2), ImageNet norm"
    targets = config["subclass_targets"]

    nonmono     = targets.get("Unusable", 0) + targets.get("Lymphocyte", 0) + targets.get("RBCalone", 0)
    unclustered = targets.get("MCwRBC", 0) + targets.get("MCwoRBC", 0)
    clustered   = targets.get("Clustered", 0)

    row = [
        run_num,
        str(date.today()),
        notes,
        "ResNet34", "ImageNet",
        "No",
        0.4,
        "3-class single model",
        70, 10, 20,
        nonmono,
        unclustered + clustered,
        targets.get("MCwRBC", 0),
        targets.get("MCwoRBC", 0),
        targets.get("Clustered", 0),
        "Equal balance per class",
        "AdamW",
        config["lr"],
        config["weight_decay"],
        config["batch_size"],
        config["num_epochs"],
        f"CosineAnnealingLR (T_max={config['num_epochs']})",
        f"{config['img_size']}x{config['img_size']}",
        aug_str,
        results["test_acc"],
        results["nonmono_prec"],
        results["nonmono_recall"],
        results["nonmono_f1"],
        results["unclust_prec"],
        results["unclust_recall"],
        results["unclust_f1"],
        results["clust_prec"],
        results["clust_recall"],
        results["clust_f1"],
        results["macro_f1"],
        results["weighted_f1"],
        results["val_acc"],
    ]

    for col_idx, value in enumerate(row, start=1):
        ws.cell(row=next_row, column=col_idx, value=value)

    wb.save(log_path)
    print(f"  ✓ Run {run_num} logged to {log_path}")

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(config):
    device = config["device"]
    print(f"Training on: {device}")
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    # MLflow Logging setup 
    setup_mlflow("3Class_NonmonovsClusteredvsUnclustered")
    start_run(run_name="3subclass_split", notes="First layer??")
    log_params(config)

    train_loader, val_loader, test_loader, fine_counts = make_dataloaders(config)

    model = build_model(num_classes=3, freeze_backbone=False).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc  = 0.0
    epochs_without_improvement = 0

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
            epochs_without_improvement = 0
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{config['early_stopping_patience']})")
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

        # log epoch information
        log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

    # ── Load best checkpoint ──
    print("\n── Loading Best Checkpoint ──")
    model.load_state_dict(torch.load(config["checkpoint_path"]))

    # ── Validation set evaluation ──
    print("\n── Validation Set Evaluation ──")
    _, val_acc_final, val_preds, val_true = evaluate(
        model, val_loader, criterion, device
    )
    print(f"Val Accuracy: {val_acc_final:.4f}\n")
    print(classification_report(
        val_true, val_preds, target_names=CLASS_NAMES
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(val_true, val_preds, labels=[0, 1, 2]))

    # ── Test set evaluation ──
    print("\n── Test Set Evaluation ──")
    _, test_acc, preds, true_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(
        true_labels, preds, target_names=CLASS_NAMES
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, preds, labels=[0, 1, 2]))

    # ── Save plots ──
    print("\n── Saving Plots ──")
    plot_dir = os.path.dirname(config["checkpoint_path"])
    save_plots(train_losses, val_losses, train_accs, val_accs,
               true_labels, preds, output_dir=plot_dir)

    # ── Log experiment ──
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, preds, labels=[0, 1, 2]
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted"
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro"
    )

    results = {
        "test_acc":        round(test_acc, 4),
        "nonmono_prec":    round(prec[2], 4),
        "nonmono_recall":  round(rec[2],  4),
        "nonmono_f1":      round(f1[2],   4),
        "unclust_prec":    round(prec[0], 4),
        "unclust_recall":  round(rec[0],  4),
        "unclust_f1":      round(f1[0],   4),
        "clust_prec":      round(prec[1], 4),
        "clust_recall":    round(rec[1],  4),
        "clust_f1":        round(f1[1],   4),
        "macro_f1":        round(f1_macro,    4),
        "weighted_f1":     round(f1_weighted, 4),
        "val_acc":         round(val_acc_final, 4),
    }

    log_experiment(config, results, log_path="C:/repos/Blood-Cell-Classification/experiment_log.xlsx",
       notes="3-class single model — Unclustered / Clustered / NonMonocyte")
    
    # After training:
    log_results(results)
    log_confusion_matrix(true_labels, preds, ["Non-Monocyte", "Monocyte"], "checkpoints_3class/confusion_matrix_3class.png")
    log_artifacts(["checkpoints_3class/loss_curve_3class.png", "checkpoints_3class/accuracy_curve_3class.png", config["checkpoint_path"]])
    end_run()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train(CONFIG)