from pathlib import Path
from typing import Any
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import convnext_tiny, convnext_base


# ─────────────────────────────────────────────────────────────────────────────
# INTENSITY NORMALISATION
# Stretches each image's per-channel intensity to a consistent range before
# ImageNet normalisation. Compensates for exposure/contrast differences across
# imaging sessions without requiring retraining.
#
# Uses robust quantile-based min-max stretch (ignores top/bottom 1% of pixels)
# so dust, bright spots, or imaging artefacts don't skew the normalisation.
# ─────────────────────────────────────────────────────────────────────────────

class NormaliseToImageNet:
    """
    Per-image robust intensity stretch to [0, 1] before normalisation.
    Assumes grayscale input (single channel).
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        # ensure grayscale
        img = img.convert("L")
        tensor = TF.to_tensor(img)          # (1, H, W) float32 in [0, 1]
        ch = tensor[0]
        lo = ch.quantile(0.01)
        hi = ch.quantile(0.99)
        if hi > lo:
            tensor[0] = ((ch - lo) / (hi - lo)).clamp(0, 1)
        return TF.to_pil_image(tensor)



# ── Preprocessing ─────────────────────────────────────────────
preprocess_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    NormaliseToImageNet(),
    transforms.ToTensor(),                      # → (1, H, W)
    transforms.Normalize(mean=[0.5], std=[0.25])
])

# -- Convert the first conv layer of a pretrained model to accept single-channel input --
def _convert_first_layer_to_grayscale(model: torch.nn.Module) -> torch.nn.Module:
    """
    Convert ConvNeXt first conv to 1‑channel to match grayscale training.
    """
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
        # if loading pretrained RGB weights, average them; if loading from
        # scratch, this will be overwritten by checkpoint anyway
        if old.weight.shape[1] == 3:
            new.weight[:] = old.weight.mean(dim=1, keepdim=True)
        else:
            new.weight[:] = old.weight
        if old.bias is not None:
            new.bias[:] = old.bias
    model.features[0][0] = new
    return model

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PATHS
#
# Two modes per node — mutually exclusive, checked at load time:
#
#   Single model  — provide "single" key with one .pth path
#   Ensemble      — provide "folds" key with a list of .pth paths (one per fold)
#
# To switch a node from single to ensemble, comment out "single" and
# uncomment "folds", then point each entry at the fold checkpoint saved
# by kfold_trainer.py (e.g. stage1_usability_fold1.pth ... fold5.pth).
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATHS = {

    "MonovsNonMono": {
        # ── Single model (original) ──
        # "single": r"model\cellusability\stage1_usability_fold4.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellusability\stage1_usability_fold1_v2.pth",
            r"model\cellusability\stage1_usability_fold2_v2.pth",
            r"model\cellusability\stage1_usability_fold3_v2.pth",
            r"model\cellusability\stage1_usability_fold4_v2.pth",
            r"model\cellusability\stage1_usability_fold5_v2.pth",
        ],
    },

    "Cluster": {
        # ── Single model (original) ──
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\convnext_base_t1500.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellclusters\stage2_clustered_fold1_v2.pth",
            r"model\cellclusters\stage2_clustered_fold2_v2.pth",
            r"model\cellclusters\stage2_clustered_fold3_v2.pth",
            r"model\cellclusters\stage2_clustered_fold4_v2.pth",
            r"model\cellclusters\stage2_clustered_fold5_v2.pth",
        ],
    },

    "Cluster_RBCCount": {
        # ── Single model (original) ──
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\convnext_tiny.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellrbccluster\clustered_binary_fold1_v2.pth",
            r"model\cellrbccluster\clustered_binary_fold2_v2.pth",
            r"model\cellrbccluster\clustered_binary_fold3_v2.pth",
            r"model\cellrbccluster\clustered_binary_fold4_v2.pth",
            r"model\cellrbccluster\clustered_binary_fold5_v2.pth",
        ],
    },

    "Unclustered_RBCCount": {
        # ── Single model (original) ──
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\convnext_tiny.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellrbcuncluster\unclustered_binary_fold1_v2.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold2_v2.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold3_v2.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold4_v2.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold5_v2.pth",
        ],
    },
}

# ── Number of output classes per node ─────────────────────────
# MonovsNonMono outputs 2 classes:
#   0 = Unusable   (terminal — not routed forward)
#   1 = Usable     (routed → Cluster)
MODEL_CLASSES = {
    "MonovsNonMono":        2,
    "Cluster":              2,
    "Cluster_RBCCount":     3,
    "Unclustered_RBCCount": 2,
}

# ── Fold weights for weighted ensemble averaging ───────────────
# One weight per fold — order must match MODEL_PATHS["folds"].
# Weights are normalised automatically so raw macro F1 scores work fine.
# Set all to 1.0 for equal weighting.
FOLD_WEIGHTS = {
    "MonovsNonMono": [
        0.8767,   # fold 1
        0.8649,   # fold 2
        0.8169,   # fold 3
        0.8963,   # fold 4
        0.8829,   # fold 5
    ],
    "Cluster": [
        0.9288,   # fold 1
        0.9400,   # fold 2
        0.9295,   # fold 3
        0.9220,   # fold 4
        0.9427,   # fold 5
    ],
    "Cluster_RBCCount": [
        0.9476,   # fold 1
        0.9136,   # fold 2
        0.9353,   # fold 3
        0.9630,   # fold 4
        0.9071,   # fold 5
    ],
    "Unclustered_RBCCount": [
        0.9823,   # fold 1
        0.9943,   # fold 2
        0.9893,   # fold 3
        0.9929,   # fold 4
        0.9740,   # fold 5
    ],
}

# ── Default per-node confidence thresholds ─────────────────────
DEFAULT_THRESHOLDS = {
    "MonovsNonMono":        0.50,
    "Cluster":              0.50,
    "Cluster_RBCCount":     0.40,
    "Unclustered_RBCCount": 0.50,
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_convnext_tiny(path: str, num_classes: int, device: str) -> torch.nn.Module:
    model = convnext_tiny(weights=None)
    model = _convert_first_layer_to_grayscale(model)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, num_classes
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_convnext_base(path: str, num_classes: int, device: str) -> torch.nn.Module:
    model = convnext_base(weights=None)
    model = _convert_first_layer_to_grayscale(model)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, num_classes
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()



def _load_model(path: str, num_classes: int, device: str,
                architecture: str) -> torch.nn.Module:
    """Load a single checkpoint by architecture name."""
    loaders = {
        "convnext_tiny": _load_convnext_tiny,
        "convnext_base": _load_convnext_base,
    }
    if architecture not in loaders:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Supported: {list(loaders.keys())}")
    return loaders[architecture](path, num_classes, device)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL NODE
# Supports both single-model and ensemble (averaged softmax) inference.
# ─────────────────────────────────────────────────────────────────────────────

class ModelNode:
    def __init__(self, name: str, device: str,
                 models: list[torch.nn.Module],
                 fold_weights: list[float] | None = None):
        """
        Args:
            name         — node name, used for threshold lookup and result logging
            device       — torch device string
            models       — list of loaded models. Single model = list of length 1.
                           Ensemble = list of k fold models.
            fold_weights — per-fold weights for weighted averaging. Must match
                           len(models). If None or mismatched, falls back to
                           equal weights. Normalised automatically.
        """
        self.name        = name
        self.device      = device
        self.models      = models
        self.routes: dict[int, str] = {}
        self.is_ensemble = len(models) > 1

        if fold_weights is not None and len(fold_weights) == len(models):
            w            = torch.tensor(fold_weights, dtype=torch.float32)
            self.weights = (w / w.sum()).to(device)
            weight_str   = ", ".join(f"{x:.3f}" for x in self.weights.cpu())
            print(f"  {name}: ensemble of {len(models)} models "
                  f"(weights: [{weight_str}])")
        else:
            self.weights = None
            if fold_weights is not None and len(fold_weights) != len(models):
                print(f"  WARNING [{name}]: FOLD_WEIGHTS has "
                      f"{len(fold_weights)} entries but {len(models)} folds — "
                      f"falling back to equal weights.")
            if self.is_ensemble:
                print(f"  {name}: ensemble of {len(models)} models (equal weights)")
            else:
                print(f"  {name}: single model")

    def set_routes(self, route_dict: dict[int, str]):
        self.routes = route_dict

    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        """
        Run inference. For ensembles, combines softmax probabilities across
        all fold models using weighted averaging (or simple mean if no weights).
        """
        tensor = preprocess_256(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            all_probs = []
            for model in self.models:
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0]
                all_probs.append(probs)

            stacked = torch.stack(all_probs)   # (n_folds, n_classes)

            if self.weights is not None:
                avg_probs = (stacked * self.weights.unsqueeze(1)).sum(dim=0)
            else:
                avg_probs = stacked.mean(dim=0)

            pred  = int(torch.argmax(avg_probs))
            score = float(avg_probs[pred])

        return {
            "pred":  pred,
            "score": score,
            "probs": avg_probs.cpu().numpy().tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE TREE
# ─────────────────────────────────────────────────────────────────────────────

class CascadeTree:
    def __init__(self, nodes: dict[str, ModelNode], root: str):
        self.nodes = nodes
        self.root  = root

    def classify(
        self,
        pil_image: Image.Image,
        thresholds: dict[str, float] = {},
    ) -> dict[str, Any]:
        """
        Classify an image through the cascade.

        Cascade structure:
            MonovsNonMono
                ├─ pred=0 (Unusable) → terminal
                └─ pred=1 (Usable)   → Cluster
                        ├─ pred=0 (Unclustered) → Unclustered_RBCCount
                        │       ├─ pred=0 → No_RBC   (terminal)
                        │       └─ pred=1 → Has_RBC  (terminal)
                        └─ pred=1 (Clustered) → Cluster_RBCCount
                                ├─ pred=0 → No_RBC    (terminal)
                                ├─ pred=1 → Has_RBC   (terminal)
                                └─ pred=2 → RBC_alone (terminal)
        """
        path    = []
        current = self.root

        while True:
            node      = self.nodes[current]
            out       = node.predict(pil_image)
            threshold = thresholds.get(node.name, 0.0)

            path.append({
                "model":       node.name,
                "pred":        out["pred"],
                "score":       out["score"],
                "probs":       out["probs"],
                "is_ensemble": node.is_ensemble,
            })

            if out["score"] < threshold:
                return {
                    "final_pred":     out["pred"],
                    "final_score":    out["score"],
                    "path":           path,
                    "low_confidence": True,
                }

            if out["pred"] not in node.routes:
                return {
                    "final_pred":     out["pred"],
                    "final_score":    out["score"],
                    "path":           path,
                    "low_confidence": False,
                }

            current = node.routes[out["pred"]]


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

# Architecture used per node.
# NOTE: Must match the architecture the checkpoints were actually trained with.
NODE_ARCHITECTURES = {
    "MonovsNonMono":        "convnext_tiny",
    "Cluster":              "convnext_tiny",
    "Cluster_RBCCount":     "convnext_tiny",
    "Unclustered_RBCCount": "convnext_tiny",
}


def build_cascade_tree(device: str = "cuda") -> CascadeTree:
    """
    Load all model weights and assemble the CascadeTree.
    Call once at startup and reuse the returned object.

    Automatically detects whether each node uses a single model or ensemble
    based on whether MODEL_PATHS[node] contains "single" or "folds".
    """
    print(f"\nBuilding cascade tree on {device}...")
    nodes = {}

    for node_name, path_config in MODEL_PATHS.items():
        num_classes  = MODEL_CLASSES[node_name]
        architecture = NODE_ARCHITECTURES[node_name]

        has_single = "single" in path_config
        has_folds  = "folds"  in path_config

        if has_single and has_folds:
            raise ValueError(
                f"[{node_name}] Both 'single' and 'folds' are defined in MODEL_PATHS. "
                f"Comment out one to choose single vs ensemble mode."
            )
        if not has_single and not has_folds:
            raise ValueError(
                f"[{node_name}] Neither 'single' nor 'folds' found in MODEL_PATHS."
            )

        if has_single:
            loaded_models = [
                _load_model(path_config["single"], num_classes, device, architecture)
            ]
        else:
            loaded_models = [
                _load_model(p, num_classes, device, architecture)
                for p in path_config["folds"]
            ]

        nodes[node_name] = ModelNode(
            node_name, device, loaded_models,
            fold_weights=FOLD_WEIGHTS.get(node_name),
        )

    # Wire up routing
    nodes["MonovsNonMono"].set_routes({1: "Cluster"})
    nodes["Cluster"].set_routes({0: "Unclustered_RBCCount", 1: "Cluster_RBCCount"})
    # RBC count nodes are terminal — no routes needed

    print("Cascade tree ready.\n")
    return CascadeTree(nodes=nodes, root="MonovsNonMono")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(
    image_paths: list[str],
    tree: CascadeTree,
    thresholds: dict[str, float] = DEFAULT_THRESHOLDS,
    log_fn=print,
) -> list[dict[str, Any]]:
    """
    Classify a list of image paths using a pre-built CascadeTree.

    Each image is intensity-normalised via NormaliseToImageNet before
    being passed to the model, compensating for exposure and contrast
    differences across imaging sessions.

    Args:
        image_paths: List of paths to segmented crop images.
        tree:        A CascadeTree built with build_cascade_tree().
        thresholds:  Per-node confidence thresholds. Defaults to
                     DEFAULT_THRESHOLDS. Pass {} to disable thresholding.
        log_fn:      Callable for logging — pass worker.log.emit for Qt signal.

    Returns:
        List of result dicts, one per image:
        { "file", "final_pred", "final_score", "path", "low_confidence" }

    Final pred meanings per terminal node:
        Unclustered_RBCCount : 0 = No_RBC,  1 = Has_RBC
        Cluster_RBCCount     : 0 = No_RBC,  1 = Has_RBC,  2 = RBC_alone
    """
    results = []

    for i, path in enumerate(image_paths):
        p = Path(path)
        try:
            img    = Image.open(p).convert("L")
            result = tree.classify(img, thresholds=thresholds)
            result["file"] = str(p)
            results.append(result)

            flag = " ⚠ low confidence" if result.get("low_confidence") else ""
            log_fn(f"  [{i+1}/{len(image_paths)}] {p.name} → "
                   f"pred={result['final_pred']} "
                   f"({result['final_score']:.2f}){flag}")

        except Exception as e:
            log_fn(f"  ❌ Failed on {p.name}: {e}")
            continue

    low_conf = sum(1 for r in results if r.get("low_confidence"))
    log_fn(f"\n✅ Classification done — {len(results)}/{len(image_paths)} images classified.")
    if low_conf:
        log_fn(f"  ⚠ {low_conf} images flagged as low confidence.")
        log_fn(f"  Thresholds used: {thresholds}")

    return results

def run_classification_ram(
    cells: list[dict],
    tree: CascadeTree,
    thresholds: dict[str, float] = DEFAULT_THRESHOLDS,
    log_fn=print,
) -> list[dict[str, Any]]:
    """
    Classify a list of in‑RAM cell dicts produced by extract_single_cells().

    Each entry in `cells` must contain:
        {
            "image": np.ndarray (H,W,3) uint8,
            "label": int,
            "parent": str,
            "index": int,
            "bbox": (minr, minc, maxr, maxc),
            "orig_bbox": original bbox
        }
    """
    results = []

    for i, cell in enumerate(cells):
        try:
            # Convert NumPy → PIL (grayscale expected by your pipeline)
            pil_img = Image.fromarray(cell["image"]).convert("L")

            out = tree.classify(pil_img, thresholds=thresholds)

            # Attach metadata
            out["parent"]    = cell["parent"]
            out["index"]     = cell["index"]
            out["bbox"]      = cell["bbox"]
            out["orig_bbox"] = cell["orig_bbox"]

            results.append(out)

            flag = " ⚠ low confidence" if out.get("low_confidence") else ""
            log_fn(
                f"  [{i+1}/{len(cells)}] {cell['parent']}_cell_{cell['index']:04d} → "
                f"pred={out['final_pred']} ({out['final_score']:.2f}){flag}"
            )

        except Exception as e:
            log_fn(f"  ❌ Failed on cell {cell['index']} from {cell['parent']}: {e}")
            continue

    low_conf = sum(1 for r in results if r.get("low_confidence"))
    log_fn(f"\n✅ Classification done — {len(results)}/{len(cells)} cells classified.")
    if low_conf:
        log_fn(f"  ⚠ {low_conf} cells flagged as low confidence.")
        log_fn(f"  Thresholds used: {thresholds}")

    return results
