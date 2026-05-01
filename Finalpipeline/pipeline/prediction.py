from pathlib import Path
from typing import Any
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import convnext_tiny, convnext_base


# ── Preprocessing ─────────────────────────────────────────────
preprocess_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\convnext_tiny_t3000.pth",

        # ── Ensemble (uncomment and fill in fold checkpoint paths to enable) ──
        "folds": [
            r"model\cellusability\stage1_usability_fold1.pth",
            r"model\cellusability\stage1_usability_fold2.pth",
            r"model\cellusability\stage1_usability_fold3.pth",
            r"model\cellusability\stage1_usability_fold4.pth",
            r"model\cellusability\stage1_usability_fold5.pth",
        ],
    },

    "Cluster": {
        # ── Single model (original) ──
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\convnext_base_t1500.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellclusters\stage2_clustered_fold1.pth",
            r"model\cellclusters\stage2_clustered_fold2.pth",
            r"model\cellclusters\stage2_clustered_fold3.pth",
            r"model\cellclusters\stage2_clustered_fold4.pth",
            r"model\cellclusters\stage2_clustered_fold5.pth",
        ],
    },

    "Cluster_RBCCount": {
        # ── Single model (original) ──
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\convnext_tiny.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellrbccluster\clustered_binary_fold1.pth",
            r"model\cellrbccluster\clustered_binary_fold2.pth",
            r"model\cellrbccluster\clustered_binary_fold3.pth",
            r"model\cellrbccluster\clustered_binary_fold4.pth",
            r"model\cellrbccluster\clustered_binary_fold5.pth",
        ],
    },

    "Unclustered_RBCCount": {
        # ── Single model (original) ──
        # "single": r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\convnext_tiny.pth",

        # ── Ensemble ──
        "folds": [
            r"model\cellrbcuncluster\unclustered_binary_fold1.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold2.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold3.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold4.pth",
            r"model\cellrbcuncluster\unclustered_binary_fold5.pth",
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
 
# ── Default per-node thresholds ────────────────────────────────
DEFAULT_THRESHOLDS = {
    "MonovsNonMono":        0.40,
    "Cluster":              0.50,
    "Cluster_RBCCount":     0.60,
    "Unclustered_RBCCount": 0.60,
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
 
def _load_convnext_tiny(path: str, num_classes: int, device: str) -> torch.nn.Module:
    model = convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, num_classes
    )
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
 
 
def _load_convnext_base(path: str, num_classes: int, device: str) -> torch.nn.Module:
    model = convnext_base(weights=None)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, num_classes
    )
    model.load_state_dict(torch.load(path, map_location=device))
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
                 models: list[torch.nn.Module]):
        """
        Args:
            name    — node name, used for threshold lookup and result logging
            device  — torch device string
            models  — list of loaded models. Single model = list of length 1.
                      Ensemble = list of k fold models. Predictions are made
                      by averaging softmax probabilities across all models.
        """
        self.name    = name
        self.device  = device
        self.models  = models
        self.routes: dict[int, str] = {}
        self.is_ensemble = len(models) > 1
 
        if self.is_ensemble:
            print(f"  {name}: ensemble of {len(models)} models")
        else:
            print(f"  {name}: single model")
 
    def set_routes(self, route_dict: dict[int, str]):
        self.routes = route_dict
 
    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        """
        Run inference. For ensembles, averages softmax probabilities
        across all fold models before taking argmax.
        """
        tensor = preprocess_256(pil_image).unsqueeze(0).to(self.device)
 
        with torch.no_grad():
            # Collect softmax probs from each model
            all_probs = []
            for model in self.models:
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0]
                all_probs.append(probs)
 
            # Average across models (ensemble) or just use the one (single)
            avg_probs = torch.stack(all_probs).mean(dim=0)
            pred      = int(torch.argmax(avg_probs))
            score     = float(avg_probs[pred])
 
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
# Reads MODEL_PATHS, detects single vs ensemble automatically,
# and assembles the CascadeTree.
# ─────────────────────────────────────────────────────────────────────────────
 
# Architecture used per node.
# NOTE: These must match the architecture the checkpoints were actually trained with.
# The original single Cluster model was convnext_base, but the kfold ensemble
# checkpoints were trained with convnext_tiny — update here if you switch back.
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
 
        # Validate config — exactly one of "single" or "folds" must be present
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
            # Single model
            loaded_models = [
                _load_model(path_config["single"], num_classes, device, architecture)
            ]
        else:
            # Ensemble — load all fold checkpoints
            fold_paths    = path_config["folds"]
            loaded_models = [
                _load_model(p, num_classes, device, architecture)
                for p in fold_paths
            ]
 
        nodes[node_name] = ModelNode(node_name, device, loaded_models)
 
    # Wire up routing
    # MonovsNonMono: pred=0 Unusable (terminal), pred=1 Usable → Cluster
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
            img    = Image.open(p).convert("RGB")
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