# Acute Lymphoblastic Leukemia Detection — Project Report

## 1. Problem

Acute Lymphoblastic Leukemia (ALL) is a blood cancer diagnosed by identifying
"blast" (cancerous) cells under a microscope. Manual screening is slow and
requires an expert. The goal: use deep learning to automatically classify
microscope blood images as **healthy** or **ALL**.

This project is based on the public **ALL-IDB** datasets (Univ. of Milan) and
takes inspiration from the paper *"An Efficient Detection and Classification
of Acute Leukemia Using Transfer Learning and Orthogonal Softmax Layer-Based
Model"* (Das et al., IEEE/ACM TCBB 2023), which reports 98–99% accuracy on
these same datasets using ResNet18 + a custom classifier layer.

## 2. Datasets

| Dataset | Images | Task | Label source |
|---|---|---|---|
| ALL-IDB2 | 260 (130 healthy / 130 ALL) | Classify a single, centered cell image | Filename suffix (`Im001_1.tif`) |
| ALL-IDB1 | 108 (59 healthy / 49 ALL) | Classify a whole blood-smear image (may contain many cells) | Filename suffix (`Im001_1.jpg`) |

Each dataset was split **70% train / 15% validation / 15% test**, stratified
to preserve the healthy/ALL ratio in each split.

## 3. Method

- **Model:** EfficientNet-B0, pretrained on ImageNet. Chosen over a heavier
  network (e.g. ResNet50) because it has far fewer parameters, which matters
  a lot when the entire dataset is only 100–260 images.
- **Augmentation:** random flips + rotation. Safe here because a blood cell
  has no "correct" orientation.
- **Two-stage transfer learning:**
  1. **Stage 1** — freeze the pretrained convolutional backbone, train only
     a new 2-class classifier head. This is the safe move when data is
     scarce: it avoids overfitting the large backbone to a tiny dataset.
  2. **Stage 2** — unfreeze the last convolutional block and fine-tune the
     whole thing at a low learning rate, letting the network adapt its
     highest-level features specifically to blood cell images.
- **Metrics:** accuracy, precision, recall, specificity, F1, AUC, confusion
  matrix — the same metrics used in the reference paper, for direct
  comparison.

Code: [`src/train_idb2.py`](src/train_idb2.py), [`src/train_idb1.py`](src/train_idb1.py).

## 4. Results

### ALL-IDB2 (single-cell classification)

| Metric | Stage 1 (frozen) | Stage 2 (fine-tuned) | Paper (ResNet18+OSL) |
|---|---|---|---|
| Accuracy | 87.2% | 87.2% | 98.2% |
| Precision | 100% | 100% | 98.6% |
| Recall | 73.7% | 73.7% | 97.7% |
| Specificity | 100% | 100% | 98.5% |
| F1 | 0.85 | 0.85 | 0.98 |
| AUC | 0.921 | 0.934 | 0.981 |

The model never misclassifies a healthy cell as ALL (0 false positives in
both stages), but misses ~5 of 19 ALL cells in the small test set. Fine-tuning
raised the model's confidence (AUC 0.921 → 0.934) without flipping any of the
final predictions.

### ALL-IDB1 (whole-image classification)

| Metric | Stage 1 (frozen) | Stage 2 (fine-tuned) |
|---|---|---|
| Accuracy | 100% | 100% |
| Precision / Recall / Specificity / F1 / AUC | all 1.00 | all 1.00 |

Perfect score here should be read with caution — the test set is only 17
images, so this is not statistically strong evidence, just an encouraging
early signal.

Confusion matrix plots for every run are saved in [`results/`](results).

## 5. Discussion

- The EfficientNet baseline is a **solid, honest starting point**: precise
  (no false alarms) but conservative (misses some true ALL cases on IDB2).
- The gap to the paper's 98% accuracy comes mainly from their custom
  **Orthogonal Softmax Layer (OSL)**, which replaces the standard classifier
  head and forces the weight vectors of each class to stay orthogonal to
  each other. This reduces overfitting further and sharpens the decision
  boundary — a natural next improvement beyond this baseline.
- Other likely improvements: more aggressive fine-tuning (unfreeze more
  blocks), k-fold cross-validation instead of one fixed split (the paper
  uses 10-fold CV, which is more robust on such small datasets), and
  combining IDB1 detection with IDB2-style per-cell classification.

## 6. How to reproduce

```bash
python src/train_idb2.py
python src/train_idb1.py
```

Trained models are saved to `models/`, metrics (JSON) and confusion matrix
plots to `results/`.
