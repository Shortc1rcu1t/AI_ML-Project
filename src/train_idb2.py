"""
Baseline leukemia cell classifier - ALL-IDB2 dataset.

Task: given a single, centered white blood cell image, classify it as
healthy (0) or ALL blast cell (1).

Model: EfficientNet-B0, pretrained on ImageNet, fine-tuned on our tiny
260-image dataset. EfficientNet is used instead of a bigger network like
ResNet50 because it reaches strong accuracy with far fewer parameters,
which matters a lot when the whole dataset is only 260 images.

Run from the project root:
    python src/train_idb2.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
DATA_DIR = "ALL_IDB2/ALL_IDB2/img"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15               # stage 1: train classifier head only, backbone frozen
FINE_TUNE_EPOCHS = 8      # stage 2: unfreeze last block, train with a low LR
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "models/efficientnet_allidb2.pth"
RESULTS_DIR = "results"

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
# ALL-IDB2 filenames look like "Im001_1.tif" -> label is the digit before
# the extension (0 = healthy, 1 = ALL blast cell).


class AllIdb2Dataset(Dataset):
    def __init__(self, filepaths, labels, transform):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_filepaths_and_labels():
    filepaths = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
    labels = [int(os.path.splitext(f)[0].split("_")[-1]) for f in filepaths]
    return filepaths, labels


# ImageNet normalization stats, required because EfficientNet was pretrained
# on ImageNet.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# Cells have no natural "up" direction, so flips/rotations are safe, realistic
# augmentations that help a lot on a dataset this small.
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze the pretrained convolutional backbone: with only ~180 training
    # images we don't have enough data to safely retrain millions of
    # parameters, so we only train the final classifier head.
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model.to(DEVICE)


def unfreeze_last_block(model):
    """Stage 2: unfreeze the final convolutional block so the model can
    adapt its highest-level features (not just the classifier) to blood
    cell images specifically, instead of only generic ImageNet features."""
    for param in model.features[-1].parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# 4. Train / evaluate loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


def print_metrics(name, labels, preds, probs, tag):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "specificity": specificity,
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print(f"\n--- {name} metrics ---")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 score:    {metrics['f1']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n[[{tn} {fp}]\n [{fn} {tp}]]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    disp = ConfusionMatrixDisplay(
        confusion_matrix(labels, preds), display_labels=["Healthy", "ALL"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"{name} confusion matrix")
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{tag}.png"), bbox_inches="tight")
    plt.close()

    return metrics


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    filepaths, labels = load_filepaths_and_labels()
    print(f"Found {len(filepaths)} images "
          f"({labels.count(0)} healthy, {labels.count(1)} ALL)")

    # 70% train / 15% val / 15% test, stratified so each split keeps the
    # same healthy/ALL ratio as the full dataset.
    train_fp, temp_fp, train_lbl, temp_lbl = train_test_split(
        filepaths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_fp, test_fp, val_lbl, test_lbl = train_test_split(
        temp_fp, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42
    )
    print(f"Train: {len(train_fp)}  Val: {len(val_fp)}  Test: {len(test_fp)}")

    train_ds = AllIdb2Dataset(train_fp, train_lbl, train_transform)
    val_ds = AllIdb2Dataset(val_fp, val_lbl, eval_transform)
    test_ds = AllIdb2Dataset(test_fp, test_lbl, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = build_model()
    criterion = nn.CrossEntropyLoss()

    # --- Stage 1: train only the classifier head, backbone frozen ---
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    print("\n=== Stage 1: training classifier head (backbone frozen) ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_labels, val_preds, _ = evaluate(model, val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    test_labels, test_preds, test_probs = evaluate(model, test_loader)
    print_metrics("STAGE 1 TEST", test_labels, test_preds, test_probs, tag="idb2_stage1")

    # --- Stage 2: unfreeze the last conv block and fine-tune at a low LR ---
    unfreeze_last_block(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=FINE_TUNE_LR
    )
    print("\n=== Stage 2: fine-tuning last block (low learning rate) ===")
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_labels, val_preds, _ = evaluate(model, val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch:2d}/{FINE_TUNE_EPOCHS} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    test_labels, test_preds, test_probs = evaluate(model, test_loader)
    print_metrics("STAGE 2 (fine-tuned) TEST", test_labels, test_preds, test_probs, tag="idb2_stage2")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nSaved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
