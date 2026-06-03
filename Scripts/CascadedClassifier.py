import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import convnext_tiny, convnext_base


# ─────────────────────────────────────────────────────────────────────────────
# INTENSITY NORMALISATION
# Stretches each image's per-channel intensity to a consistent range before
# ImageNet normalisation. Compensates for exposure/contrast differences across
# imaging sessions without requiring retraining.
# ─────────────────────────────────────────────────────────────────────────────

class NormaliseToImageNet:
    """
    Per-channel robust intensity stretch to [0, 1] before ImageNet normalisation.
    Uses quantile-based min-max (ignores top/bottom 1% of pixels) so dust,
    bright spots, or imaging artefacts don't skew the normalisation.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        tensor = TF.to_tensor(img)          # (3, H, W) float32 in [0, 1]
        for c in range(tensor.shape[0]):
            ch = tensor[c]
            lo = ch.quantile(0.01)
            hi = ch.quantile(0.99)
            if hi > lo:
                tensor[c] = ((ch - lo) / (hi - lo)).clamp(0, 1)
        return TF.to_pil_image(tensor)
    
def convert_first_layer_to_grayscale(model):
    old = model.features[0][0]
    new = torch.nn.Conv2d(
        1, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None)
    )
    with torch.no_grad():
        new.weight[:] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias[:] = old.bias
    model.features[0][0] = new
    return model



# ── Preprocessing ─────────────────────────────────────────────
preprocess_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    NormaliseToImageNet(),              # ← normalise exposure before ImageNet norm
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL NODE
# ─────────────────────────────────────────────────────────────────────────────

class ModelNode:
    def __init__(self, name: str, model: torch.nn.Module, device="cpu"):
        self.name   = name
        self.model  = model.to(device)
        self.device = device
        self.routes: Dict[int, str] = {}

    def set_routes(self, route_dict: Dict[int, str]):
        self.routes = route_dict

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        tensor = preprocess_256(pil_image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
            pred   = int(torch.argmax(probs))
            score  = float(probs[pred])
        return {"pred": pred, "score": score, "probs": probs.cpu().numpy().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE TREE
# ─────────────────────────────────────────────────────────────────────────────

class CascadeTree:
    def __init__(self, nodes: Dict[str, ModelNode], root: str):
        self.nodes = nodes
        self.root  = root

    def classify(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
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
            node = self.nodes[current]
            out  = node.predict(pil_image)

            path.append({
                "model": node.name,
                "pred":  out["pred"],
                "score": out["score"],
                "probs": out["probs"],
            })

            if out["pred"] not in node.routes:
                return {
                    "final_pred":  out["pred"],
                    "final_score": out["score"],
                    "path":        path,
                }

            current = node.routes[out["pred"]]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_convnext_tiny(path: str, num_classes: int, device: str = "cuda"):
    model = convnext_tiny(weights=None)
    model = convert_first_layer_to_grayscale(model)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, num_classes
    )

    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def load_convnext_base(path: str, num_classes: int, device: str = "cuda"):
    model = convnext_base(weights=None)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features, num_classes
    )
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — spot check a single image
# ─────────────────────────────────────────────────────────────────────────────

def main(img_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load models ──
    Model_MonocytevsNonMonocyte  = load_convnext_tiny(
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\stage1_usability_fold1_v2.pth",
        num_classes=2, device=device
    )

    Model_ClusteredvsUnclustered = load_convnext_tiny(
        r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\stage2_clustered_fold1.pth",
        num_classes=2, device=device
    )
    Model_ClusteredRBCCount      = load_convnext_tiny(
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered_binary\clustered_binary_fold1.pth",
        num_classes=3, device=device
    )
    Model_UnclusteredRBCCount    = load_convnext_tiny(
        r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_binary\unclustered_binary_fold1.pth",
        num_classes=2, device=device
    )

    # ── Wrap in nodes ──
    MonovsNonMono        = ModelNode("MonovsNonMono",        Model_MonocytevsNonMonocyte,  device)
    Cluster              = ModelNode("Cluster",              Model_ClusteredvsUnclustered, device)
    Cluster_RBCCount     = ModelNode("Cluster_RBCCount",     Model_ClusteredRBCCount,      device)
    Unclustered_RBCCount = ModelNode("Unclustered_RBCCount", Model_UnclusteredRBCCount,    device)

    # ── Routing ──
    MonovsNonMono.set_routes({1: "Cluster"})
    Cluster.set_routes({0: "Unclustered_RBCCount", 1: "Cluster_RBCCount"})

    # ── Build cascade ──
    tree = CascadeTree(
        nodes={
            "MonovsNonMono":        MonovsNonMono,
            "Cluster":              Cluster,
            "Cluster_RBCCount":     Cluster_RBCCount,
            "Unclustered_RBCCount": Unclustered_RBCCount,
        },
        root="MonovsNonMono"
    )

    # ── Classify ──
    img    = Image.open(img_path).convert("RGB")
    result = tree.classify(img)

    # ── Pretty print ──
    print(f"Final pred  : {result['final_pred']}")
    print(f"Final score : {result['final_score']:.3f}")
    print(f"\nCascade path:")
    for step in result["path"]:
        probs_str = "  ".join(f"{p:.3f}" for p in step["probs"])
        print(f"  {step['model']:25s} → pred={step['pred']}  "
              f"score={step['score']:.3f}  probs=[{probs_str}]")


if __name__ == "__main__":
    main(r"D:\tester2\SingleCells\tile_x009_y007_cell_0115.png")