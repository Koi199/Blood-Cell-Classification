import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Callable, Any
from torchvision.models import convnext_tiny, convnext_base

# -------------------------
# Shared preprocessing (256x256)
# -------------------------
from torchvision import transforms

preprocess_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# -------------------------
# Model Node
# -------------------------
class ModelNode:
    def __init__(self, name: str, model: torch.nn.Module, device="cpu"):
        self.name = name
        self.model = model.to(device)
        self.device = device
        self.routes: Dict[int, str] = {}   # class → next model name

    def set_routes(self, route_dict: Dict[int, str]):
        """
        Example:
            {0: "ModelB", 1: "ModelC"}
        """
        self.routes = route_dict

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        tensor = preprocess_256(pil_image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs))
            score = float(probs[pred])

        return {
            "pred": pred,
            "score": score,
            "probs": probs.cpu().numpy().tolist()
        }

# -------------------------
# Cascade Tree
# -------------------------
class CascadeTree:
    def __init__(self, nodes: Dict[str, ModelNode], root: str):
        self.nodes = nodes
        self.root = root

    def classify(self, pil_image: Image.Image) -> Dict[str, Any]:
        path = []
        current = self.root

        while True:
            node = self.nodes[current]
            out = node.predict(pil_image)

            path.append({
                "model": node.name,
                "pred": out["pred"],
                "score": out["score"],
                "probs": out["probs"]
            })

            # If no route for this class → leaf node
            if out["pred"] not in node.routes:
                return {
                    "final_pred": out["pred"],
                    "final_score": out["score"],
                    "path": path
                }

            # Otherwise route to next model
            current = node.routes[out["pred"]]

def load_convnext_tiny(path, num_classes, device="cuda"):
    # 1. Instantiate architecture
    model = convnext_tiny(weights=None)   # no pretrained weights
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features,
                                          num_classes)

    # 2. Load state_dict
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    # 3. Move to device
    return model.to(device)

def load_convnext_base(path, num_classes, device="cuda"):
    model = convnext_base(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features,
                                          num_classes)

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    return model.to(device)

def main(img_path):
    # Load your models
    Model_MonocytevsNonMonocyte = load_convnext_tiny(r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\convnext_tiny_t3000.pth", 3) 
    Model_ClusteredvsUnclustered = load_convnext_base(r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\convnext_base_t1500.pth", 2)
    Model_ClusteredRBCCount = load_convnext_tiny(r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered\convnext_tiny.pth", 5)
    Model_UnclusteredRBCCount = load_convnext_base(r"C:\repos\Blood-Cell-Classification\checkpoints_rbc\convnext_base.pth", 4)

    # Wrap them
    MonovsNonMono = ModelNode("MonovsNonMono", Model_MonocytevsNonMonocyte, device="cuda")
    Cluster = ModelNode("Cluster", Model_ClusteredvsUnclustered, device="cuda")
    Cluster_RBCCount = ModelNode("Cluster_RBCCount", Model_ClusteredRBCCount, device="cuda")
    Unclustered_RBCCount = ModelNode("Unclustered_RBCCount", Model_UnclusteredRBCCount, device="cuda")

    # Define routing logic
    MonovsNonMono.set_routes({
        2: "Cluster"    # If A predicts class 2 → go to B (monocyte)
    })

    Cluster.set_routes({
        0: "Unclustered_RBCCount",  # Unclustered
        1: "Cluster_RBCCount" # clustered
    })

    # Build cascade
    tree = CascadeTree(
        nodes={"MonovsNonMono": MonovsNonMono, "Cluster": Cluster, "Cluster_RBCCount": Cluster_RBCCount, "Unclustered_RBCCount": Unclustered_RBCCount},
        root="MonovsNonMono"
    )

    # Run on an image
    img = Image.open(img_path)
    result = tree.classify(img)
    print(result)

if __name__ == "__main__":
    main(r"D:\MMA_batch1\contrast_1.0_Sliced\Slide 1-1\tile_x001_y004_cell_0028.png")