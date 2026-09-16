from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import convnext_tiny

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGS (your existing cascade config)
# ─────────────────────────────────────────────
_stage2_target = 2200  # placeholder if needed; keep as in your original

CONFIGS = {
    "stage1_usability": {
        "name":        "stage1_usability",
        "data_dir":    r"D:/MMA_LabelledData/training_perslide_pruned",
        "folder_map": {
            "Unusable":             0,
            "RBC alone":            0,
            "Monocyte_with_RBC":    1,
            "Monocyte_without_RBC": 1,
            "Clustered_cell":       1,
        },
        "class_names":    ["Unusable", "Usable"],
        "checkpoint_dir": r"C:/repos/Blood-Cell-Classification/checkpoints_stage1",
        "subclass_targets": {
            "Unusable":             2200,
            "RBC alone":            2200,
            "Monocyte_with_RBC":    2200,
            "Monocyte_without_RBC": 2200,
            "Clustered_cell":       2200,
        },
    },

    "stage2_clustered": {
        "name":        "stage2_clustered",
        "data_dir":    r"D:/MMA_LabelledData/training_perslide_pruned",
        "folder_map": {
            "Monocyte_with_RBC":    0,
            "Monocyte_without_RBC": 0,
            "Clustered_cell":       1,
        },
        "class_names":    ["Unclustered", "Clustered"],
        "checkpoint_dir": r"C:/repos/Blood-Cell-Classification/checkpoints_stage2",
        "subclass_targets": {
            "Unclustered": _stage2_target,
            "Clustered":   _stage2_target,
        },
    },

    "unclustered_binary": {
        "name":        "unclustered_binary",
        "data_dir":    r"D:/MMA_LabelledData/training_perslide_pruned/Unclustered_RBCCount",
        "folder_map": {
            "RBC_0": 0,
            "RBC_1": 1,
            "RBC_2": 1,
            "RBC_3": 1,
            "RBC_4": 1,
            "RBC_5": 1,
        },
        "class_names":    ["No_RBC", "Has_RBC"],
        "checkpoint_dir": r"C:/repos/Blood-Cell-Classification/checkpoints_rbc_binary",
        "subclass_targets": {
            "No_RBC":  800,
            "Has_RBC": 800,
        },
    },

    "clustered_binary": {
        "name":        "clustered_binary",
        "data_dir":    r"D:/MMA_LabelledData/training_perslide_pruned/clustered_RBCCount",
        "folder_map": {
            "RBC_0":     0,
            "RBC_1":     1,
            "RBC_2":     1,
            "RBC_3":     1,
            "RBC_4":     1,
            "RBC_5":     1,
        },
        "class_names":    ["No_RBC", "Has_RBC"],
        "checkpoint_dir": r"C:/repos/Blood-Cell-Classification/checkpoints_rbc_clustered_binary",
        "subclass_targets": {
            "No_RBC":    1600,
            "Has_RBC":   1600,
        },
    },
}

# ─────────────────────────────────────────────
# PREPROCESSING (match production cascade)
# ─────────────────────────────────────────────
IMAGE_SIZE = 256
NORM_MEAN = [0.5]
NORM_STD  = [0.25]

class NormaliseToImageNet:
    """Per-image robust intensity stretch to [0, 1] before normalisation."""
    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        tensor = TF.to_tensor(img)
        ch = tensor[0]
        lo = ch.quantile(0.01)
        hi = ch.quantile(0.99)
        if hi > lo:
            tensor[0] = ((ch - lo) / (hi - lo)).clamp(0, 1)
        return TF.to_pil_image(tensor)


preprocess_256 = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    NormaliseToImageNet(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def load_image_for_display(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=-1)  # (H, W, 3)


# ─────────────────────────────────────────────
# MODEL LOADING (ConvNeXt-Tiny grayscale ensemble)
# ─────────────────────────────────────────────
def _convert_first_layer_to_grayscale(model: torch.nn.Module) -> torch.nn.Module:
    old = model.features[0][0]
    new = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        if old.weight.shape[1] == 3:
            new.weight[:] = old.weight.mean(dim=1, keepdim=True)
        else:
            new.weight[:] = old.weight
        if old.bias is not None:
            new.bias[:] = old.bias
    model.features[0][0] = new
    return model


def load_stage_ensemble(stage_key: str, device: str):
    cfg = CONFIGS[stage_key]
    ckpt_dir = Path(cfg["checkpoint_dir"])
    class_names = cfg["class_names"]

    pattern = f"{cfg['name']}_capped3.0x_fold*_v2.pth"
    model_paths = sorted(ckpt_dir.glob(pattern))
    if len(model_paths) == 0:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir} matching {pattern}")

    models = []
    for mp in model_paths:
        model = convnext_tiny(weights=None)
        model = _convert_first_layer_to_grayscale(model)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features,
                                              len(class_names))
        state = torch.load(mp, map_location=device)
        model.load_state_dict(state)
        models.append(model.to(device).eval())

    # uniform weights across folds (you can swap in FOLD_WEIGHTS if desired)
    weights = torch.ones(len(models), dtype=torch.float32)
    weights = (weights / weights.sum()).to(device)

    return models, weights, class_names


def predict_ensemble(models: list[torch.nn.Module], weights: torch.Tensor | None,
                     img: Image.Image, device: str) -> dict[str, Any]:
    tensor = preprocess_256(img).unsqueeze(0).to(device)
    with torch.no_grad():
        all_probs = []
        for model in models:
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            all_probs.append(probs)
        stacked = torch.stack(all_probs)  # (n_folds, n_classes)
        if weights is not None:
            avg_probs = (stacked * weights.unsqueeze(1)).sum(dim=0)
        else:
            avg_probs = stacked.mean(dim=0)
        pred = int(torch.argmax(avg_probs))
        score = float(avg_probs[pred])
    return {
        "pred": pred,
        "score": score,
        "probs": avg_probs.cpu().numpy(),
        "per_fold_probs": [p.cpu().numpy() for p in all_probs],
    }


# ─────────────────────────────────────────────
# GRADCAM++ on ensemble
# ─────────────────────────────────────────────
def get_gradcam_target_layer(model):
    # last spatial block before global pooling in convnext_tiny
    return [model.features[7]]


def run_gradcam_ensemble(
    models: list[torch.nn.Module],
    img: Image.Image,
    target_class_idx: int | None,
    device: str,
    blur_kernel: int = 11,
) -> np.ndarray:
    tensor = preprocess_256(img).unsqueeze(0).to(device)
    heatmaps = []

    for model in models:
        target_layers = get_gradcam_target_layer(model)
        targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
        heatmaps.append(grayscale_cam[0])

    avg_heatmap = np.mean(heatmaps, axis=0)
    if blur_kernel > 0:
        avg_heatmap = cv2.GaussianBlur(avg_heatmap, (blur_kernel, blur_kernel), sigmaX=0)
    return avg_heatmap


# ─────────────────────────────────────────────
# SINGLE IMAGE FIGURE
# ─────────────────────────────────────────────
def plot_single(
    image_path: str,
    models: list[torch.nn.Module],
    weights: torch.Tensor,
    class_names: list[str],
    device: str,
    target_class_idx: int | None = None,
    label: str = "",
    save_path=None,
    dpi: int = 300,
    alpha: float = 0.5,
    blur_kernel: int = 11,
):
    img = Image.open(image_path).convert("L")
    img_display = load_image_for_display(image_path)

    pred_info = predict_ensemble(models, weights, img, device)
    heatmap = run_gradcam_ensemble(models, img, target_class_idx, device, blur_kernel)

    overlay = show_cam_on_image(
        img_display, heatmap,
        use_rgb=True,
        image_weight=1 - alpha,
        colormap=cv2.COLORMAP_TURBO,
    )

    saliency_class = (class_names[target_class_idx] if target_class_idx is not None
                      else class_names[pred_info["pred"]])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_display, cmap="gray")
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title(f"GradCAM++\n(saliency for: {saliency_class})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    title_parts = [label or Path(image_path).name]
    title_parts.append(
        f"Predicted: {class_names[pred_info['pred']]} ({pred_info['score']:.1%})"
    )
    prob_str = "  ".join(f"{c}: {p:.3f}" for c, p in zip(class_names, pred_info["probs"]))
    title_parts.append(prob_str)

    fig.suptitle("\n".join(title_parts), fontsize=9, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stem = save_path.stem.replace("_saliency", "")
        ext  = save_path.suffix

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved (combined):  {save_path}")

        fig_hm, ax_hm = plt.subplots(figsize=(4, 4))
        ax_hm.imshow(heatmap, cmap="turbo", vmin=0, vmax=1)
        ax_hm.axis("off")
        fig_hm.tight_layout(pad=0)
        hm_path = save_path.parent / f"{stem}_heatmap{ext}"
        fig_hm.savefig(hm_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_hm)
        print(f"  Saved (heatmap):   {hm_path}")

        fig_ov, ax_ov = plt.subplots(figsize=(4, 4))
        ax_ov.imshow(overlay)
        ax_ov.axis("off")
        fig_ov.tight_layout(pad=0)
        ov_path = save_path.parent / f"{stem}_overlay{ext}"
        fig_ov.savefig(ov_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_ov)
        print(f"  Saved (overlay):   {ov_path}")

    return fig


# ─────────────────────────────────────────────
# STAGE-AWARE ENTRY POINT
# ─────────────────────────────────────────────
def generatemaps_for_stage(
    stage: str,
    images: list[str],
    output_dir: str,
    target_class: str | None = None,
    alpha: float = 0.5,
    dpi: int = 300,
    device: str = "cuda",
    fmt: str = "png",
    blur_kernel: int = 11,
):
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU.")
        device = "cpu"

    models, weights, class_names = load_stage_ensemble(stage, device)

    target_class_idx = None
    if target_class is not None:
        if target_class not in class_names:
            raise ValueError(f"target_class '{target_class}' not in {class_names}")
        target_class_idx = class_names.index(target_class)
        print(f"Saliency target: '{target_class}' (index {target_class_idx})")
    else:
        print("Saliency target: ensemble predicted class")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage: {stage}")
    print(f"Classes: {class_names}")
    print(f"Output dir: {output_dir}")

    for i, img_path in enumerate(images):
        lbl       = Path(img_path).stem
        save_path = output_dir / f"{lbl}_saliency.{fmt}"
        print(f"[{i+1}/{len(images)}] {Path(img_path).name}")
        plot_single(
            image_path=img_path,
            models=models,
            weights=weights,
            class_names=class_names,
            device=device,
            target_class_idx=target_class_idx,
            label=lbl,
            save_path=save_path,
            dpi=dpi,
            alpha=alpha,
            blur_kernel=blur_kernel,
        )

    print(f"\n✅ Done. Figures saved to {output_dir}")


# ─────────────────────────────────────────────
# MAIN EXAMPLE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    generatemaps_for_stage(
        stage="clustered_binary",
        images=[
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_0\Slide1-1\tile_x007_y003_cell_0227_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_0\Slide1-1\tile_x007_y003_cell_0233_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_0\Slide1-1\tile_x007_y003_cell_0197_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_0\Slide1-1\tile_x007_y003_cell_0200_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_0\Slide1-1\tile_x007_y003_cell_0211_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_1\Slide1-1\tile_x007_y003_cell_0187_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_1\Slide1-1\tile_x007_y003_cell_0192_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_1\Slide1-1\tile_x007_y003_cell_0160_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_1\Slide1-1\tile_x007_y003_cell_0180_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_1\Slide1-1\tile_x007_y003_cell_0182_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_2\Slide1-1\tile_x007_y004_cell_0099_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_2\Slide1-1\tile_x007_y004_cell_0100_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_2\Slide1-1\tile_x007_y004_cell_0063_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_2\Slide1-1\tile_x007_y004_cell_0073_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_2\Slide1-1\tile_x007_y004_cell_0077_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_3\Slide1-1\tile_x007_y003_cell_0283_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_3\Slide1-1\tile_x007_y003_cell_0215_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_3\Slide1-1\tile_x007_y003_cell_0236_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_3\Slide1-1\tile_x007_y003_cell_0274_slide1_1.png",
            r"D:\MMA_LabelledData\training_perslide_pruned\Clustered_RBCCount\RBC_3\Slide1-1\tile_x007_y003_cell_0282_slide1_1.png"

        ],
        output_dir=r"C:\Users\ubcmd\Documents\Publication Figures\Saliency Maps\Clustered Binary",
        target_class=None,
        alpha=0.5,
        dpi=300,
        device="cuda",
        fmt="png",
        blur_kernel=11,
    )