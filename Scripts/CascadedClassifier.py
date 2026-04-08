import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Callable, Any

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

def main(img_path):
    # Load your models
    Model_MonocytevsNonMonocyte = torch.load(r"C:\repos\Blood-Cell-Classification\checkpoints_stage1\convnext_tiny_t3000.pth") 
    Model_ClusteredvsUnclustered = torch.load(r"C:\repos\Blood-Cell-Classification\checkpoints_stage2\convnext_base_t1500.pth")
    Model_ClusteredRBCCount = torch.load(r"C:\repos\Blood-Cell-Classification\checkpoints_rbc_clustered\convnext_tiny.pth")
    Model_UnclusteredRBCCount = torch.load(r"C:\repos\Blood-Cell-Classification\checkpoints_rbc\convnext_tiny.pth")

    # Wrap them
    A = ModelNode("A", Model_MonocytevsNonMonocyte, device="cuda")
    B = ModelNode("B", Model_ClusteredvsUnclustered, device="cuda")
    C = ModelNode("C", Model_ClusteredRBCCount, device="cuda")
    D = ModelNode("D", Model_UnclusteredRBCCount, device="cuda")

    # Define routing logic
    A.set_routes({
        1: "B"    # If A predicts class 1 → go to C
    })

    B.set_routes({
        0: "D",   # If B predicts class 2 → go to C
        1: "C"
    })

    # Build cascade
    tree = CascadeTree(
        nodes={"A": A, "B": B, "C": C, "D": D},
        root="A"
    )

    # Run on an image
    img = Image.open(img_path)
    result = tree.classify(img)
    print(result)

if __name__ == "__main__":
    main(r"D:\MMA_LabelledData\Clustered_RBCCount\RBC_1\tile_x001_y001_cell_0087_slide1_5.png")