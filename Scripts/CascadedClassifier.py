"""
spot_check_cell.py

Manual spot-check tool: runs a single cell image through the FULL ensemble
cascade (same ensemble averaging + FOLD_WEIGHTS + normalization as the
production pipeline) but in a simple, single-image, verbose-print format
so you can eyeball what's happening at each stage.

Uses the SAME NormaliseToImageNet (grayscale, quantile stretch,
Normalize(mean=[0.5], std=[0.25])) as the production cascade — this is
deliberate. If you were previously spot-checking with a script that used
a different normalization (e.g. std=0.5, or per-channel-before-grayscale
ordering), that mismatch alone can cause different predictions on the
same cell.

Usage:
    python spot_check_cell.py "D:\\tester\\Patient2_8\\tile_x002_y001\\tile_x002_y001_cell_0002.png"
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import convnext_tiny, convnext_base


# ─────────────────────────────────────────────────────────────────────────────
# INTENSITY NORMALISATION (identical to production cascade)
# ─────────────────────────────────────────────────────────────────────────────

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
    transforms.Resize((256, 256)),
    NormaliseToImageNet(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.25]),
])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (identical to production cascade)
# ─────────────────────────────────────────────────────────────────────────────

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


def load_ensemble(paths: list[str], num_classes: int, device: str,
                   architecture: str = "convnext_tiny") -> list[torch.nn.Module]:
    ctor = {"convnext_tiny": convnext_tiny, "convnext_base": convnext_base}[architecture]
    models = []
    for path in paths:
        model = ctor(weights=None)
        model = _convert_first_layer_to_grayscale(model)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device).eval())
    return models


def predict_ensemble(models: list[torch.nn.Module], weights: torch.Tensor | None,
                      img: Image.Image, device: str) -> dict[str, Any]:
    """Weighted-average softmax across fold models (same math as ModelNode.predict)."""
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
        "probs": avg_probs.cpu().numpy().tolist(),
        "per_fold_probs": [p.cpu().numpy().tolist() for p in all_probs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — same fold paths / weights as the production pipeline
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "MonovsNonMono": [
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\stage1_usability_capped3.0x_fold1_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\stage1_usability_capped3.0x_fold2_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\stage1_usability_capped3.0x_fold3_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\stage1_usability_capped3.0x_fold4_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\stage1_usability_capped3.0x_fold5_v2.pth",
    ],
    "Cluster": [
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\stage2_clustered_capped3.0x_fold1_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\stage2_clustered_capped3.0x_fold2_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\stage2_clustered_capped3.0x_fold3_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\stage2_clustered_capped3.0x_fold4_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\stage2_clustered_capped3.0x_fold5_v2.pth",
    ],
    "Cluster_RBCCount": [
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\clustered_binary_capped3.0x_fold1_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\clustered_binary_capped3.0x_fold2_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\clustered_binary_capped3.0x_fold3_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\clustered_binary_capped3.0x_fold4_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\clustered_binary_capped3.0x_fold5_v2.pth",
    ],
    "Unclustered_RBCCount": [
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\unclustered_binary_capped3.0x_fold1_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\unclustered_binary_capped3.0x_fold2_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\unclustered_binary_capped3.0x_fold3_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\unclustered_binary_capped3.0x_fold4_v2.pth",
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\unclustered_binary_capped3.0x_fold5_v2.pth",
    ],
}

MODEL_CLASSES = {
    "MonovsNonMono": 2,
    "Cluster": 2,
    "Cluster_RBCCount": 2,
    "Unclustered_RBCCount": 2,
}

FOLD_WEIGHTS = {
    "MonovsNonMono":        [0.8767, 0.8649, 0.8169, 0.8963, 0.8829],
    "Cluster":              [0.9288, 0.9400, 0.9295, 0.9220, 0.9427],
    "Cluster_RBCCount":     [0.9476, 0.9136, 0.9353, 0.9630, 0.9071],
    "Unclustered_RBCCount": [0.9823, 0.9943, 0.9893, 0.9929, 0.9740],
}

LABEL_MAPS = {
    "MonovsNonMono":        {0: "Unusable",    1: "Usable"},
    "Cluster":              {0: "Unclustered", 1: "Clustered"},
    "Cluster_RBCCount":     {0: "No_RBC",      1: "Has_RBC"},
    "Unclustered_RBCCount": {0: "No_RBC",      1: "Has_RBC"},
}


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE (structured like doc 3's classify(), but ensemble-aware)
# ─────────────────────────────────────────────────────────────────────────────

def classify(img_path: str, ensembles: dict[str, list[torch.nn.Module]],
             weights: dict[str, torch.Tensor], device: str) -> None:
    img = Image.open(img_path).convert("L")

    print(f"\nImage: {img_path}")
    print("─" * 70)

    def run_node(node_name: str) -> dict[str, Any]:
        out = predict_ensemble(ensembles[node_name], weights.get(node_name), img, device)
        label = LABEL_MAPS[node_name][out["pred"]]
        probs_str = "  ".join(f"{p:.3f}" for p in out["probs"])
        print(f"  {node_name:20s} → {label:12s}  score={out['score']:.3f}  "
              f"probs=[{probs_str}]")
        # Per-fold breakdown so you can see if one fold disagrees with the rest
        for i, fp in enumerate(out["per_fold_probs"], start=1):
            fold_pred = int(max(range(len(fp)), key=lambda k: fp[k]))
            fold_str = "  ".join(f"{p:.3f}" for p in fp)
            flag = "" if fold_pred == out["pred"] else "  ⚠ disagrees with ensemble"
            print(f"      fold{i}: pred={fold_pred} probs=[{fold_str}]{flag}")
        return out

    # ── Stage 1: Usable vs Unusable ──
    out = run_node("MonovsNonMono")
    if out["pred"] == 0:
        print("\n  ✗ Cell classified as Unusable — stopped.")
        return

    # ── Stage 2: Clustered vs Unclustered ──
    out = run_node("Cluster")

    # ── Stage 3: RBC count ──
    node = "Unclustered_RBCCount" if out["pred"] == 0 else "Cluster_RBCCount"
    out = run_node(node)

    label = LABEL_MAPS[node][out["pred"]]
    print("\n" + "─" * 70)
    print(f"  Final: {label}  (score={out['score']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(img_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ensembles = {}
    weights = {}
    for node_name, paths in MODEL_PATHS.items():
        ensembles[node_name] = load_ensemble(
            paths, MODEL_CLASSES[node_name], device, architecture="convnext_tiny"
        )
        w = torch.tensor(FOLD_WEIGHTS[node_name], dtype=torch.float32)
        weights[node_name] = (w / w.sum()).to(device)

    classify(img_path, ensembles, weights, device)


if __name__ == "__main__":
    # Point this at the same cell crop you're spot-checking
    main(r"D:\tester\3\tile_x010_y014\tile_x010_y014_cell_0001.png")