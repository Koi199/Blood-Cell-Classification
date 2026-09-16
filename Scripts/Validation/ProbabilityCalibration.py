"""
kfold_inference_runner_with_plots.py

Runs k-fold inference for all stages and automatically generates
reliability diagrams (global + per-class) for each model checkpoint.

FIX (this version): each stage now writes to its OWN separate CSV file,
instead of every stage appending to one shared CalibrationRun_All.csv.
Previously, plots generated for e.g. clustered_binary were reading the
entire multi-stage accumulated file and grouping by 'true_class' across
ALL stages ever run -- silently mixing in Unusable/Usable (stage1),
Clustered/Unclustered (stage2), and both RBC-binary stages' identically-
named Has_RBC/No_RBC classes into what should have been a single stage's
2-class breakdown. Per-stage CSVs make that bug structurally impossible.

As a second safety net, generate_plots_for_model also accepts a stage_key
and will filter defensively even if a shared CSV is ever passed by mistake.
"""

import ast
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from K_foldValidation import CONFIGS


def plot_reliability_diagram(
    csv_path,
    n_bins: int = 10,
    save_path=None,
    title: str = "Reliability Diagram",
    dpi: int = 300,
    figsize: tuple = (6, 6),
):
    """
    Plot a reliability diagram from a calibration CSV, and compute
    Expected Calibration Error.

    Args:
        csv_path:   path to a calibration CSV (must have 'confidence'
                    and 'correct' columns). Can also be a DataFrame directly.
        n_bins:     number of confidence bins (default 10, i.e. deciles)
        save_path:  if given, saves the figure
        title:      plot title
        dpi:        resolution for saved raster formats
        figsize:    figure size in inches

    Returns:
        dict with:
            ece: Expected Calibration Error (weighted avg |confidence - accuracy|)
            bin_confidence, bin_accuracy, bin_counts: per-bin arrays
            fig, ax: matplotlib objects
    """
    df = csv_path if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)

    required = {"confidence", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required column(s): {missing}")

    probs = df["confidence"].to_numpy()
    correct = df["correct"].to_numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges[1:-1])  # bins 0..n_bins-1

    bin_ids_kept = []
    bin_confidence = []
    bin_accuracy = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue
        bin_ids_kept.append(b)
        bin_confidence.append(probs[mask].mean())
        bin_accuracy.append(correct[mask].mean())
        bin_counts.append(count)

    bin_confidence = np.array(bin_confidence)
    bin_accuracy = np.array(bin_accuracy)
    bin_counts = np.array(bin_counts)

    total = bin_counts.sum()
    ece = float(np.sum((bin_counts / total) * np.abs(bin_accuracy - bin_confidence))) if total else float("nan")

    # Fixed, evenly-spaced bar positions -- one center per actual bin index,
    # NOT the data's true mean confidence (that's what the red line/dots show).
    fixed_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_positions = fixed_bin_centers[bin_ids_kept]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    bin_width = (bin_edges[1] - bin_edges[0]) * 0.9
    ax.bar(
        bar_positions, bin_accuracy,
        width=bin_width, alpha=0.65, color="#4C72B0",
        edgecolor="white", label="Observed accuracy",
        align="center",
    )
    ax.plot(bin_confidence, bin_accuracy, "o-", color="#C44E52",
            linewidth=1.5, markersize=5, label="Model calibration curve")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence", fontsize=11)
    ax.set_ylabel("Observed accuracy", fontsize=11)
    ax.set_title(f"{title}\nECE = {ece:.4f}  |  n = {total}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return {
        "ece": ece,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
        "fig": fig,
        "ax": ax,
    }


def plot_reliability_by_class(
    csv_path,
    n_bins: int = 10,
    save_path=None,
    dpi: int = 300,
    ncols: int = 3,
):
    """
    Plot a separate reliability diagram per TRUE class -- useful for
    checking whether calibration quality differs across cell types.

    IMPORTANT: pass a csv_path (or DataFrame) already scoped to a SINGLE
    stage. If it contains rows from multiple stages, 'true_class' values
    from different stages will be plotted side by side as if they were
    all classes of the same model, which is misleading.

    Args:
        csv_path:  path to a calibration CSV, or a DataFrame directly
        n_bins:    number of confidence bins per subplot
        save_path: if given, saves the combined grid figure
        dpi:       resolution for saved raster formats
        ncols:     number of columns in the grid

    Returns:
        dict mapping class_name -> {ece, bin_confidence, bin_accuracy, bin_counts}
    """
    df = csv_path if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
    classes = sorted(df["true_class"].unique())
    n = len(classes)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    results = {}

    for i, cls in enumerate(classes):
        ax = axes[i]
        sub = df[df["true_class"] == cls]
        probs = sub["confidence"].to_numpy()
        correct = sub["correct"].to_numpy()

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(probs, bin_edges[1:-1])

        bin_confidence, bin_accuracy, bin_counts = [], [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            bin_confidence.append(probs[mask].mean())
            bin_accuracy.append(correct[mask].mean())
            bin_counts.append(mask.sum())

        bin_confidence = np.array(bin_confidence)
        bin_accuracy = np.array(bin_accuracy)
        bin_counts = np.array(bin_counts)
        total = bin_counts.sum() if len(bin_counts) else 0
        ece = float(np.sum((bin_counts / total) * np.abs(bin_accuracy - bin_confidence))) if total else float("nan")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.plot(bin_confidence, bin_accuracy, "o-", color="#C44E52", markersize=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{cls}\nECE={ece:.3f}, n={total}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        results[cls] = {
            "ece": ece,
            "bin_confidence": bin_confidence,
            "bin_accuracy": bin_accuracy,
            "bin_counts": bin_counts,
        }

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    results["_fig"] = fig
    results["_axes"] = axes
    return results


class FlatFolderDataset(Dataset):
    """
    Loads every image directly inside `root` (no subfolders expected),
    all labelled with a single class name you provide explicitly.
    """
    def __init__(self, root, class_name: str, transform=None):
        self.root = Path(root)
        self.class_name = class_name
        self.transform = transform

        self.samples = [
            str(p) for p in sorted(self.root.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ]
        if not self.samples:
            raise RuntimeError(f"No image files found directly inside {self.root}")

        self.classes = [class_name]
        self.class_to_idx = {class_name: 0}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # single class, index 0


# ─────────────────────────────────────────────
# MODEL LOADING -- adjust to match your actual architecture
# ─────────────────────────────────────────────
def build_model(num_classes: int, in_channels: int = 1, device: str = "cuda"):
    """
    ASSUMPTION: torchvision convnext_tiny with a resized classifier head.
    """
    model = models.convnext_tiny(weights=None)

    if in_channels != 3:
        old_conv = model.features[0][0]
        new_conv = torch.nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        model.features[0][0] = new_conv

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.to(device)
    return model


def load_checkpoint(model, checkpoint_path: str, device: str = "cuda"):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────
# TRANSFORMS -- adjust to match training preprocessing
# ─────────────────────────────────────────────
def get_transforms(in_channels: int = 1):
    """
    ASSUMPTION: 224x224 input, grayscale by default. Confirm mean/std
    against your actual training pipeline before trusting results --
    a mismatch here silently produces miscalibrated-looking output that
    is actually a preprocessing bug, not a model property.
    """
    if in_channels == 1:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# INFERENCE + CSV APPEND
# ─────────────────────────────────────────────
def run_calibration_inference(
    model_path: str,
    data_dir: str,
    output_csv: str,
    fold,
    class_names: list,
    true_class: str,
    in_channels: int = 1,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    log_fn=print,
):
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if true_class not in class_names:
        raise ValueError(f"true_class '{true_class}' not found in class_names {class_names}")

    if not torch.cuda.is_available() and device == "cuda":
        log_fn("⚠️ CUDA not available, falling back to CPU.")
        device = "cpu"

    transform = get_transforms(in_channels=in_channels)
    dataset = FlatFolderDataset(root=str(data_dir), class_name=true_class, transform=transform)
    log_fn(f"[fold={fold}] Found {len(dataset)} images in {data_dir}, all labelled '{true_class}'")

    if len(dataset) == 0:
        log_fn(f"⚠️ No images found under {data_dir} -- nothing to do.")
        return output_csv

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=len(class_names), in_channels=in_channels, device=device)
    model = load_checkpoint(model, model_path, device=device)

    true_idx = class_names.index(true_class)

    rows = []
    sample_paths = [s[0] for s in dataset.samples]
    idx = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            pred_indices = probs.argmax(dim=1)
            pred_confidences = probs.max(dim=1).values

            for i in range(images.size(0)):
                filepath = sample_paths[idx]
                pred_idx = pred_indices[i].item()

                row = {
                    "fold": fold,
                    "filepath": filepath,
                    "true_class": true_class,
                    "predicted_class": class_names[pred_idx],
                    "confidence": float(pred_confidences[i].item()),
                    "correct": int(true_idx == pred_idx),
                }
                for c_idx, c_name in enumerate(class_names):
                    row[f"prob_{c_name}"] = float(probs[i, c_idx].item())

                rows.append(row)
                idx += 1

            if (batch_idx + 1) % 10 == 0:
                log_fn(f"[fold={fold}] Processed {idx}/{len(dataset)} images...")

    fieldnames = ["fold", "filepath", "true_class", "predicted_class", "confidence", "correct"] + \
                 [f"prob_{c}" for c in class_names]

    write_header = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    overall_acc = sum(r["correct"] for r in rows) / len(rows)
    log_fn(f"\n✅ [fold={fold}] Done. {len(rows)} predictions appended to {output_csv}")
    log_fn(f"   Accuracy on this run's data: {overall_acc:.4f}")

    return output_csv


# ───────────────────────────────────────────────────────────────
# Parse uploaded fold list
# ───────────────────────────────────────────────────────────────
def parse_heldout_slides(csv_path: str, stage_key: str):
    """
    Works for:
        - comma-separated CSV
        - tab-separated text
        - semicolon-separated CSV
        - Excel-exported CSV
    """
    folds = {}
    delimiters = [",", "\t", ";"]

    for delim in delimiters:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)

            for row in reader:
                if len(row) < 4:
                    continue

                foldinfo = row[0].strip()
                stage = row[-1].strip()

                if stage != stage_key:
                    continue
                if not foldinfo.startswith("Fold"):
                    continue

                tokens = foldinfo.split()
                fold_label = " ".join(tokens[:2])

                start = foldinfo.find("[")
                end = foldinfo.find("]")
                if start == -1 or end == -1:
                    continue

                list_text = foldinfo[start:end + 1]

                try:
                    slides = ast.literal_eval(list_text)
                except Exception:
                    continue

                folds[fold_label] = slides

        if folds:
            return folds

    print(f"⚠ No folds found for stage {stage_key}. Check delimiter or file format.")
    return folds


# ───────────────────────────────────────────────────────────────
# Plotting helper
# ───────────────────────────────────────────────────────────────
def generate_plots_for_model(model_name: str, csv_path: str, output_dir: str, stage_key: str = None):
    """
    Creates:
        - reliability diagram (ECE)
        - per-class reliability grid

    Saves:
        model_name_reliability.pdf / .png
        model_name_byclass.pdf / .png

    stage_key: if given, defensively filters csv_path down to only rows
    whose 'fold' column contains this stage_key before plotting. This is
    a safety net -- with the per-stage-CSV fix in main(), csv_path should
    already contain only one stage's data, but this guards against any
    future accidental reuse of a shared/combined CSV.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if stage_key is not None:
        stage_mask = df["fold"].astype(str).str.contains(f" - {stage_key} - ", regex=False)
        n_before = len(df)
        df = df[stage_mask]
        if df.empty:
            print(f"⚠ No rows found for stage {stage_key} in {csv_path} (had {n_before} total rows) -- skipping plots.")
            return
        if len(df) != n_before:
            print(f"ℹ Filtered {csv_path} from {n_before} to {len(df)} rows for stage {stage_key}.")

    # Global reliability diagram
    diag = plot_reliability_diagram(
        df,
        save_path=str(out_dir / f"{model_name}_reliability.pdf")
    )
    diag["fig"].savefig(out_dir / f"{model_name}_reliability.png", dpi=300)

    # Per-class reliability
    byclass = plot_reliability_by_class(
        df,
        save_path=str(out_dir / f"{model_name}_byclass.pdf")
    )
    byclass["_fig"].savefig(out_dir / f"{model_name}_byclass.png", dpi=300)

    print(f"📊 Saved reliability plots for {model_name}")


# Helper to extract fold number
def extract_fold_number(fold_label: str) -> str:
    parts = fold_label.split()
    if len(parts) < 2:
        raise ValueError(f"Cannot extract fold number from: {fold_label}")

    num = "".join(ch for ch in parts[1] if ch.isdigit())
    if not num:
        raise ValueError(f"Cannot extract fold number from: {fold_label}")

    return num


# ───────────────────────────────────────────────────────────────
# Stage inference helper
# ───────────────────────────────────────────────────────────────
def run_stage_inference(stage_key: str, fold_name: str, heldout_slides, output_csv, device):
    cfg = CONFIGS[stage_key]
    raw_folders = list(cfg["folder_map"].keys())

    fold_num = extract_fold_number(fold_name)
    checkpoint = Path(cfg["checkpoint_dir"]) / f"{cfg['name']}_capped3.0x_fold{fold_num}_v2.pth"

    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint} -- skipping {stage_key} / {fold_name}")
        return

    for raw in raw_folders:
        true_class = cfg["class_names"][cfg["folder_map"][raw]]

        for slide in heldout_slides:
            d = Path(cfg["data_dir"]) / raw / slide

            if not d.exists():
                print(f"⚠ Skipping missing folder: {d}")
                continue

            image_files = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
            if len(image_files) == 0:
                print(f"⚠ Skipping empty folder: {d} (no image files found)")
                continue

            run_calibration_inference(
                model_path=str(checkpoint),
                data_dir=str(d),
                output_csv=output_csv,
                fold=f"{fold_name} - {stage_key} - {raw} - {slide}",
                class_names=cfg["class_names"],
                true_class=true_class,
                in_channels=1,
                device=device,
            )


# ───────────────────────────────────────────────────────────────
# Main driver
# ───────────────────────────────────────────────────────────────
def main():
    csv_list_path = r"C:\repos\Blood-Cell-Classification\CalibrationRuns.csv"
    output_dir = Path(r"C:\repos\Blood-Cell-Classification")
    plot_output_dir = r"C:\repos\Blood-Cell-Classification\CalibrationPlots"

    stages = [
        "stage1_usability",
        "stage2_clustered",
        "unclustered_binary",
        "clustered_binary",
    ]

    for stage_key in stages:
        print(f"\n=== Parsing held-out slides for {stage_key} ===")

        # FIX: separate output CSV per stage, instead of one shared file
        # across all stages. This is what actually prevents the class-
        # mixing bug -- each stage's calibration data now lives in its
        # own file and can never accidentally include another stage's
        # 'true_class' values.
        output_csv = str(output_dir / f"CalibrationRun_{stage_key}.csv")

        folds = parse_heldout_slides(csv_list_path, stage_key=stage_key)

        for fold_name, slide_list in folds.items():
            print(f"\n=== Running inference for {fold_name} ({stage_key}) ===")

            run_stage_inference(
                stage_key=stage_key,
                fold_name=fold_name,
                heldout_slides=slide_list,
                output_csv=output_csv,
                device="cuda",
            )
            # stage_key passed through as a defensive filter too, even
            # though output_csv is already stage-scoped now.
            generate_plots_for_model(
                f"{stage_key}_{fold_name}", output_csv, plot_output_dir, stage_key=stage_key
            )


if __name__ == "__main__":
    main()