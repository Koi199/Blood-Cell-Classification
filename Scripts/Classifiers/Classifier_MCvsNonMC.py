import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from datetime import date
from sklearn.metrics import precision_recall_fscore_support

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
_mono_total = 2211  # 737 × 3

CONFIG = {
    "data_dir": "D:/MMA_LabelledData/Sliced",
    "img_size": 256,
    "batch_size": 32,
    "num_epochs": 50,                  # high ceiling — early stopping decides
    "early_stopping_patience": 7,      # stop if no improvement for 7 epochs
    "lr": 5e-5,
    "weight_decay": 5e-4,              # increased from 1e-4 for more regularisation
    "num_workers": 4,
    "checkpoint_path": "checkpoints/resnet34_binary.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "class_weights": [1.0, 3.0],       # penalise monocyte misses 3x more
    "optimise_metric": "F2",           # optimise for recall-weighted F2 score
    "classification_threshold": 0.35,  # refined by threshold search after training

    # Top-level binary targets
    "binary_targets": {
        0: 2211,   # Non-Monocyte — natural training count
        1: _mono_total,   # Monocyte — 737 × 3
    },

    # Within-Monocyte subtype targets (must sum to binary_targets[1])
    "monocyte_subtype_targets": {
        "MCwRBC":    _mono_total // 3,
        "MCwoRBC":   _mono_total // 3,
        "Clustered": _mono_total - 2 * (_mono_total // 3),  # absorbs rounding remainder
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
# Stronger augmentation to simulate cross-slide variation
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
            w = binary_targets[0] / fine_counts["NonMonocyte"]
        else:
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
        nn.Dropout(p=0.5),                              # increased from 0.4
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
# EVALUATE
# Supports custom threshold for cascaded architecture
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        # Use threshold instead of argmax for flexible decision boundary
        probs = torch.softmax(outputs, dim=1)
        preds = (probs[:, 1] >= threshold).long()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ─────────────────────────────────────────────
# THRESHOLD SEARCH
# Find optimal threshold for cascaded architecture —
# prioritises monocyte recall > 0.95
# ─────────────────────────────────────────────
def threshold_search(model, test_loader, criterion, device):
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\n── Threshold Search ──")
    print(f"{'Threshold':>10} {'Mono Recall':>12} {'Mono Prec':>10} {'NonMono Prec':>13} {'Accuracy':>10} {'F2':>8}")
    print("-" * 70)

    best_threshold = 0.5
    best_score = 0

    for threshold in np.arange(0.20, 0.71, 0.05):
        preds = (np.array(all_probs) >= threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, preds, labels=[0, 1], zero_division=0
        )
        f2 = fbeta_score(all_labels, preds, beta=2, average="weighted")
        acc = np.mean(np.array(preds) == np.array(all_labels))

        print(f"{threshold:>10.2f} {rec[1]:>12.4f} {prec[1]:>10.4f} "
              f"{prec[0]:>13.4f} {acc:>10.4f} {f2:>8.4f}")

        # Target: monocyte recall >= 0.95, maximise non-mono precision
        if rec[1] >= 0.95 and prec[0] > best_score:
            best_score = prec[0]
            best_threshold = threshold

    print(f"\n  Recommended threshold: {best_threshold:.2f}")
    print(f"  (Monocyte recall ≥ 0.95, best Non-Monocyte precision)")
    return best_threshold

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

    # ── Confusion matrix heatmap (%) ──
    cm = confusion_matrix(true_labels, preds)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=["Non-Monocyte", "Monocyte"],
                yticklabels=["Non-Monocyte", "Monocyte"])
    plt.title("Confusion Matrix (%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    for text in plt.gca().texts:
        text.set_text(text.get_text() + "%")
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
    """
    wb = load_workbook(log_path)
    ws = wb.active

    next_row = ws.max_row + 1
    run_num  = next_row - 2  # subtract 2 header rows

    mono_target = sum(config["monocyte_subtype_targets"].values())
    aug_str = "HFlip, VFlip, Rot(15°), ColorJitter(b=0.3,c=0.3,s=0.2,h=0.05), Grayscale(0.05), GaussianBlur, Affine(t=0.05)"

    row = [
        run_num,
        str(date.today()),
        notes,
        "ResNet34", "ImageNet",
        "Yes" if config.get("freeze_backbone", False) else "No",
        0.5,                                              # updated dropout
        12494,
        70, 10, 20,
        config["binary_targets"][0],
        mono_target,
        config["monocyte_subtype_targets"]["MCwRBC"],
        config["monocyte_subtype_targets"]["MCwoRBC"],
        config["monocyte_subtype_targets"]["Clustered"],
        f"Fixed [{config['class_weights'][0]}, {config['class_weights'][1]}]",
        "AdamW",
        config["lr"],
        config["weight_decay"],
        config["batch_size"],
        config["num_epochs"],
        f"ReduceLROnPlateau(factor=0.5, patience=3)",
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

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(config):
    device = config["device"]
    print(f"Training on: {device}")
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    train_loader, val_loader, test_loader, fine_counts = make_dataloaders(config)

    # Two-stage training: freeze backbone for first 5 epochs, then unfreeze
    model = build_model(num_classes=2, freeze_backbone=True).to(device)

    # Fixed asymmetric class weights — penalise monocyte misses 3x more
    class_weights = torch.tensor(
        config["class_weights"], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1    # prevents overconfidence
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ── Track metrics per epoch ──
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_f2       = 0.0
    best_val_acc  = 0.0
    epochs_without_improvement = 0

    print("\n── Training ──")
    for epoch in range(1, config["num_epochs"] + 1):

        # ── Unfreeze backbone after epoch 5 ──
        if epoch == 6:
            for param in model.parameters():
                param.requires_grad = True
            print("  ✓ Backbone unfrozen — fine-tuning all layers")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_preds_epoch, val_labels_epoch = evaluate(
            model, val_loader, criterion, device,
            threshold=config["classification_threshold"]
        )

        # Compute F2 on val set — weights recall higher than precision
        f2 = fbeta_score(val_labels_epoch, val_preds_epoch, beta=2, average="weighted")

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F2: {f2:.4f}")

        # Save best model based on F2 score (recall-weighted)
        if f2 > best_f2:
            best_f2 = f2
            best_val_acc = val_acc
            torch.save(model.state_dict(), config["checkpoint_path"])
            epochs_without_improvement = 0
            print(f"  ✓ New best model saved (F2={f2:.4f}, val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{config['early_stopping_patience']})")
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

    # ── Load best checkpoint ──
    print("\n── Loading Best Checkpoint ──")
    model.load_state_dict(torch.load(config["checkpoint_path"]))

    # ── Threshold search — find optimal for cascaded architecture ──
    print("\n── Threshold Search for Cascaded Architecture ──")
    optimal_threshold = threshold_search(model, test_loader, criterion, device)

    # ── Validation set evaluation (at optimal threshold) ──
    print("\n── Validation Set Evaluation ──")
    _, val_acc_final, val_preds, val_true = evaluate(
        model, val_loader, criterion, device, threshold=optimal_threshold
    )
    print(f"Val Accuracy: {val_acc_final:.4f}\n")
    print(classification_report(
        val_true, val_preds,
        target_names=["Non-Monocyte", "Monocyte"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(val_true, val_preds))

    # ── Test set evaluation (at optimal threshold) ──
    print("\n── Test Set Evaluation ──")
    _, test_acc, preds, true_labels = evaluate(
        model, test_loader, criterion, device, threshold=optimal_threshold
    )
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
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted"
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro"
    )
    cm = confusion_matrix(true_labels, preds)

    log_experiment(config, {
        "test_acc":        round(test_acc, 4),
        "non_mono_prec":   round(prec[0], 4),
        "non_mono_recall": round(rec[0],  4),
        "non_mono_f1":     round(f1[0],   4),
        "mono_prec":       round(prec[1], 4),
        "mono_recall":     round(rec[1],  4),
        "mono_f1":         round(f1[1],   4),
        "macro_f1":        round(f1_macro,    4),
        "weighted_f1":     round(f1_weighted, 4),
        "tp": int(cm[1,1]), "tn": int(cm[0,0]),
        "fp": int(cm[0,1]), "fn": int(cm[1,0]),
        "val_acc": round(val_acc_final, 4),
    }, log_path="C:/repos/Blood-Cell-Classification/experiment_log.xlsx",
       notes="")  # <-- fill in notes per run

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train(CONFIG)