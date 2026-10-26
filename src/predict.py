"""
Run the trained model on a single image and print the prediction.

Usage:
    python src/predict.py path/to/image.tif
    python src/predict.py path/to/image.jpg --dataset idb1

--dataset selects which trained model to load: "idb2" (default, single-cell
model) or "idb1" (whole-image model).
"""

import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Healthy", "ALL (blast cell)"]

MODEL_PATHS = {
    "idb2": "models/efficientnet_allidb2.pth",
    "idb1": "models/efficientnet_allidb1.pth",
}

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(weights_path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)  # add batch dimension
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return pred_idx, probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image to classify")
    parser.add_argument("--dataset", choices=["idb1", "idb2"], default="idb2",
                         help="Which trained model to use (default: idb2)")
    args = parser.parse_args()

    model = load_model(MODEL_PATHS[args.dataset])
    pred_idx, probs = predict(model, args.image)

    print(f"Image:       {args.image}")
    print(f"Model:       {MODEL_PATHS[args.dataset]}")
    print(f"Prediction:  {CLASS_NAMES[pred_idx]}")
    print(f"Confidence:  Healthy={probs[0]*100:.1f}%  ALL={probs[1]*100:.1f}%")


if __name__ == "__main__":
    main()
