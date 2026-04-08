# select_for_labeling.py
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ---------- CONFIG ----------
MODEL_CHECKPOINT = "checkpoints_stage1/convnext_tiny_t3000.pth"
UNLABELED_DIR = "D:\MMA_batch1\contrast_1.0_Sliced"          # images only, any substructure is fine
OUTPUT_ROUND_DIR = "active_learning/round_01"  # will create to_label/ inside
NUM_TO_QUERY = 200                       # how many images to select
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# You can change this per model
def build_model(num_classes: int):
    from torchvision.models import convnext_tiny
    model = convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    return model

NUM_CLASSES = 5  # e.g. monocyte, monocyte_rbc, rbc_only, clustered, unusable
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = build_model(NUM_CLASSES)
    state = torch.load(MODEL_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def list_images(root):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
    return paths

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    ent = -(probs * log_probs).sum(dim=-1)
    return ent  # shape: [N]

@torch.no_grad()
def score_unlabeled(model, image_paths, batch_size=64):
    scores = []
    paths = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        x = torch.stack(imgs).to(DEVICE)
        logits = model(x)
        ent = entropy_from_logits(logits)  # [B]
        scores.extend(ent.cpu().tolist())
        paths.extend(batch_paths)
    return scores, paths

def main():
    os.makedirs(OUTPUT_ROUND_DIR, exist_ok=True)
    to_label_dir = Path(OUTPUT_ROUND_DIR) / "to_label"
    to_label_dir.mkdir(parents=True, exist_ok=True)

    print("Listing unlabeled images...")
    unlabeled_paths = list_images(UNLABELED_DIR)
    print(f"Found {len(unlabeled_paths)} unlabeled images.")

    print("Scoring images...")
    model = load_model()
    scores, paths = score_unlabeled(model, unlabeled_paths)

    # sort by descending entropy (most uncertain first)
    scored = list(zip(scores, paths))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:NUM_TO_QUERY]

    print(f"Selecting top {NUM_TO_QUERY} images...")
    for _, p in selected:
        dst = to_label_dir / p.name
        shutil.copy2(p, dst)

    print(f"Copied {len(selected)} images to {to_label_dir}")

if __name__ == "__main__":
    main()