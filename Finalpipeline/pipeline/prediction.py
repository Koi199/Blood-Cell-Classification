from pathlib import Path
from typing import Any
from PIL import Image
import csv
import json
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

# ── Hardcoded model paths ──────────────────────────────────────
MODEL_PATHS = {
    "MonovsNonMono":        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\convnext_tiny_t3000.pth",
    "Cluster":              r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\convnext_base_t1500.pth",
    "Cluster_RBCCount":     r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\convnext_tiny.pth",
    "Unclustered_RBCCount": r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\convnext_tiny.pth",
}

# ── Number of output classes per node ─────────────────────────
# MonovsNonMono        : 3  (NonMono=0, ?, Mono=2)
# Cluster              : 2  (Unclustered=0, Clustered=1)
# Cluster_RBCCount     : 3  (No_RBC=0, Has_RBC=1, RBC_alone=2)
# Unclustered_RBCCount : 2  (No_RBC=0, Has_RBC=1)
MODEL_CLASSES = {
    "MonovsNonMono":        3,
    "Cluster":              2,
    "Cluster_RBCCount":     3,
    "Unclustered_RBCCount": 2,
}

# ── Default per-node thresholds ────────────────────────────────
# Tune these against your validation set.
# With binary nodes the baseline is higher so thresholds can be stricter.
DEFAULT_THRESHOLDS = {
    "MonovsNonMono":        0.40,   # 3 classes, baseline 0.33
    "Cluster":              0.50,   # 2 classes, baseline 0.50
    "Cluster_RBCCount":     0.60,   # 3 classes, baseline 0.33
    "Unclustered_RBCCount": 0.60,   # 2 classes, baseline 0.50
}


# ── Loaders ───────────────────────────────────────────────────
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


# ── ModelNode ─────────────────────────────────────────────────
class ModelNode:
    def __init__(self, name: str, model: torch.nn.Module, device: str):
        self.name   = name
        self.model  = model
        self.device = device
        self.routes: dict[int, str] = {}

    def set_routes(self, route_dict: dict[int, str]):
        self.routes = route_dict

    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        tensor = preprocess_256(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
            pred   = int(torch.argmax(probs))
            score  = float(probs[pred])
        return {"pred": pred, "score": score, "probs": probs.cpu().numpy().tolist()}


# ── CascadeTree ───────────────────────────────────────────────
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
                └─ pred=2 (Mono) → Cluster
                        ├─ pred=0 (Unclustered) → Unclustered_RBCCount
                        │       ├─ pred=0 → No_RBC   (terminal)
                        │       └─ pred=1 → Has_RBC  (terminal → RBC segmentation)
                        └─ pred=1 (Clustered) → Cluster_RBCCount
                                ├─ pred=0 → No_RBC    (terminal)
                                ├─ pred=1 → Has_RBC   (terminal → RBC segmentation)
                                └─ pred=2 → RBC_alone (terminal, excluded from counts)

        Args:
            pil_image:  PIL image to classify.
            thresholds: Per-node confidence thresholds keyed by model name.
                        If a node's score falls below its threshold, the cascade
                        stops and the result is flagged with low_confidence=True.
                        Nodes not present in the dict default to 0.0 (no filtering).
                        Use DEFAULT_THRESHOLDS as a starting point.
        """
        path    = []
        current = self.root

        while True:
            node      = self.nodes[current]
            out       = node.predict(pil_image)
            threshold = thresholds.get(node.name, 0.0)

            path.append({
                "model": node.name,
                "pred":  out["pred"],
                "score": out["score"],
                "probs": out["probs"],
            })

            # Stop cascade if confidence is below this node's threshold
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


# ── Factory: build the tree once ─────────────────────────────
def build_cascade_tree(device: str = "cuda") -> CascadeTree:
    """
    Load all model weights and assemble the CascadeTree.
    Call once at startup and reuse the returned object.

    Routing:
        MonovsNonMono  : pred=2 → Cluster
        Cluster        : pred=0 → Unclustered_RBCCount
                         pred=1 → Cluster_RBCCount
        Cluster_RBCCount / Unclustered_RBCCount : terminal (no further routing)
    """
    mono = ModelNode(
        "MonovsNonMono",
        _load_convnext_tiny(MODEL_PATHS["MonovsNonMono"], MODEL_CLASSES["MonovsNonMono"], device),
        device,
    )
    cluster = ModelNode(
        "Cluster",
        _load_convnext_base(MODEL_PATHS["Cluster"], MODEL_CLASSES["Cluster"], device),
        device,
    )
    clust_rbc = ModelNode(
        "Cluster_RBCCount",
        _load_convnext_tiny(MODEL_PATHS["Cluster_RBCCount"], MODEL_CLASSES["Cluster_RBCCount"], device),
        device,
    )
    unclust_rbc = ModelNode(
        "Unclustered_RBCCount",
        _load_convnext_tiny(MODEL_PATHS["Unclustered_RBCCount"], MODEL_CLASSES["Unclustered_RBCCount"], device),
        device,
    )

    mono.set_routes({2: "Cluster"})
    cluster.set_routes({0: "Unclustered_RBCCount", 1: "Cluster_RBCCount"})
    # RBC count nodes are terminal — no routes needed

    return CascadeTree(
        nodes={
            "MonovsNonMono":        mono,
            "Cluster":              cluster,
            "Cluster_RBCCount":     clust_rbc,
            "Unclustered_RBCCount": unclust_rbc,
        },
        root="MonovsNonMono",
    )


# ── Main entry point ──────────────────────────────────────────
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
        thresholds:  Per-node confidence thresholds keyed by model name.
                     Defaults to DEFAULT_THRESHOLDS. Pass an empty dict {}
                     to disable thresholding entirely.
        log_fn:      Callable for logging — pass worker.log.emit for Qt signal.

    Returns:
        List of result dicts, one per image, each containing:
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
            log_fn(f"  [{i+1}/{len(image_paths)}] {p.name} → pred={result['final_pred']} ({result['final_score']:.2f}){flag}")
        except Exception as e:
            log_fn(f"  ❌ Failed on {p.name}: {e}")
            continue

    low_conf = sum(1 for r in results if r.get("low_confidence"))
    log_fn(f"\n✅ Classification done — {len(results)}/{len(image_paths)} images classified.")
    if low_conf:
        log_fn(f"  ⚠ {low_conf} images flagged as low confidence.")
        log_fn(f"  Thresholds used: {thresholds}")

    return results