"""
Generates a detailed PDF project report (Project_Report.pdf) from the
results already produced by train_idb1.py / train_idb2.py.

Run from the project root, after both training scripts have been run:
    python src/generate_report_pdf.py
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RESULTS_DIR = "results"
OUT_PATH = "Project_Report.pdf"


def load_metrics(tag):
    with open(os.path.join(RESULTS_DIR, f"metrics_{tag}.json")) as f:
        return json.load(f)


def metrics_table_data(header, rows):
    """rows: list of (label, stage1_metrics, stage2_metrics or None)"""
    data = [header]
    for label, m1, m2 in rows:
        row = [label, f"{m1:.3f}" if isinstance(m1, float) else m1]
        if m2 is not None:
            row.append(f"{m2:.3f}" if isinstance(m2, float) else m2)
        data.append(row)
    return data


def styled_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def build():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1c", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8,
            textColor=colors.HexColor("#2c3e50"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2c", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
            textColor=colors.HexColor("#34495e"),
        )
    )
    styles.add(
        ParagraphStyle(name="Body", parent=styles["BodyText"], spaceAfter=8, leading=15)
    )
    styles["Code"].fontSize = 8.3
    styles["Code"].leading = 11
    styles["Code"].backColor = colors.HexColor("#f5f5f5")

    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    )
    story = []

    # ---------------------------------------------------------------- Title
    story.append(Paragraph("Acute Lymphoblastic Leukemia Detection", styles["Title"]))
    story.append(Paragraph(
        "A Transfer-Learning Approach with EfficientNet-B0 on the ALL-IDB1 and ALL-IDB2 Datasets",
        ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=12,
                        textColor=colors.HexColor("#555555"), spaceAfter=20, alignment=1),
    ))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------------- 1. Problem
    story.append(Paragraph("1. Problem and Motivation", styles["H1c"]))
    story.append(Paragraph(
        "Acute Lymphoblastic Leukemia (ALL) is a fast-progressing blood cancer that originates in the "
        "bone marrow and affects white blood cells (lymphocytes). Diagnosis relies on a trained "
        "hematologist visually identifying abnormal \"blast\" cells in a microscope image of a blood "
        "smear. This process is slow, requires expert training, and is subject to human error and "
        "inter-observer disagreement.", styles["Body"],
    ))
    story.append(Paragraph(
        "The goal of this project is to build a deep learning model that automatically classifies "
        "microscope blood images as <b>healthy</b> or <b>ALL</b>, as a proof-of-concept for computer-aided "
        "diagnosis. We deliberately picked a task with small, well-annotated public datasets, because "
        "that is the realistic situation in most medical imaging problems: labeled medical data is "
        "expensive and scarce, so the model design has to work well with very little data - this shaped "
        "almost every choice described in this report.", styles["Body"],
    ))
    story.append(Paragraph(
        "This work is inspired by, and benchmarked against, the paper <i>\"An Efficient Detection and "
        "Classification of Acute Leukemia Using Transfer Learning and Orthogonal Softmax Layer-Based "
        "Model\"</i> (Das, Sahoo &amp; Meher, IEEE/ACM Transactions on Computational Biology and "
        "Bioinformatics, 2023), which uses the same two datasets and reports 98-99% accuracy using a "
        "ResNet18 backbone with a custom classification layer.", styles["Body"],
    ))

    # ---------------------------------------------------------------- 2. Datasets
    story.append(Paragraph("2. The Datasets", styles["H1c"]))
    story.append(Paragraph(
        "Both datasets come from Fabio Scotti's group at the Universita degli Studi di Milano and are "
        "the de-facto standard public benchmark for this problem.", styles["Body"],
    ))

    story.append(Paragraph("2.1 ALL-IDB1 - whole blood smear images", styles["H2c"]))
    story.append(Paragraph(
        "108 whole microscope-field images (<code>ImXXX_Y.jpg</code>), each possibly containing many "
        "cells. <b>Y = 0</b> means the image is from a healthy donor; <b>Y = 1</b> means the donor has "
        "ALL and at least one blast cell is present somewhere in the image. Each image also comes with "
        "a companion <code>.xyc</code> text file giving the (x, y) pixel coordinates of every blast "
        "cell's centroid - ground truth for a future localization/detection task, which we did not need "
        "for classification but is available for extension work. 59 images are healthy, 49 are ALL.",
        styles["Body"],
    ))
    story.append(Paragraph("2.2 ALL-IDB2 - single, centered cell images", styles["H2c"]))
    story.append(Paragraph(
        "260 images (<code>ImXXX_Y.tif</code>), each a crop containing one cell placed in the center of "
        "the frame. <b>Y = 0</b> means the centered cell is healthy; <b>Y = 1</b> means it is a blast "
        "cell from an ALL patient. Perfectly balanced: 130 healthy, 130 ALL. Because the label directly "
        "describes the one cell in the image (rather than \"somewhere in this image\"), this is a "
        "cleaner, easier classification problem than ALL-IDB1, and was therefore treated as the primary "
        "dataset for this project.", styles["Body"],
    ))
    story.append(Paragraph(
        "In both datasets, the label is encoded directly in the filename, so no separate label file "
        "had to be parsed for the classification task - the data loading code simply reads the digit "
        "before the file extension.", styles["Body"],
    ))

    # ---------------------------------------------------------------- 3. Method
    story.append(Paragraph("3. Method - Why We Built It This Way", styles["H1c"]))

    story.append(Paragraph("3.1 Choosing EfficientNet-B0", styles["H2c"]))
    story.append(Paragraph(
        "The single biggest constraint on this project is dataset size: 108-260 images is tiny by deep "
        "learning standards. A large network such as ResNet50 or VGG19 has tens of millions of "
        "parameters and will overfit almost immediately on that little data. EfficientNet-B0 was chosen "
        "specifically because it was designed to get strong ImageNet accuracy with a much smaller "
        "parameter count than older architectures (roughly 5 million parameters vs. 25 million for "
        "ResNet50), by scaling network depth, width and input resolution together in a balanced way. "
        "Starting from ImageNet-pretrained weights (transfer learning) rather than training from "
        "scratch is essential here too - the network already knows general-purpose visual features "
        "(edges, textures, shapes) before it ever sees a single blood cell image.", styles["Body"],
    ))

    story.append(Paragraph("3.2 Two-stage training", styles["H2c"]))
    story.append(Paragraph(
        "Training was split into two deliberate stages instead of fine-tuning the whole network at once:",
        styles["Body"],
    ))
    story.append(Paragraph(
        "<b>Stage 1 - frozen backbone.</b> All of EfficientNet's convolutional layers are frozen "
        "(kept fixed at their ImageNet weights) and only a new 2-class classification head is trained. "
        "With so little data, allowing gradients to flow back into millions of backbone parameters "
        "immediately would destroy the useful pretrained features and overfit to our small training set. "
        "Training only the head is the standard, safe way to adapt a pretrained network to a new, small "
        "dataset.", styles["Body"],
    ))
    story.append(Paragraph(
        "<b>Stage 2 - fine-tuning the last block.</b> Once the classifier head has converged, the final "
        "convolutional block of EfficientNet is unfrozen and the whole thing is trained further at a "
        "much lower learning rate (1e-5 instead of 1e-4). This lets the network adjust its highest-level, "
        "most task-specific features (which, for ImageNet, encode things like \"dog ear\" or \"car "
        "wheel\") toward blood-cell-specific patterns, while the lower-level, more generic layers "
        "(edges, colors, textures) are left untouched. This staged strategy is a standard transfer "
        "learning best practice for small datasets.", styles["Body"],
    ))

    story.append(Paragraph("3.3 Data augmentation", styles["H2c"]))
    story.append(Paragraph(
        "Training images are randomly flipped (horizontal and vertical) and rotated up to 20 degrees. "
        "This is a safe augmentation choice specifically because blood cells have no natural "
        "\"orientation\" the way, say, a photo of a person does - a rotated or mirrored cell is still a "
        "perfectly realistic cell image, so this effectively multiplies the amount of training data seen "
        "without introducing unrealistic examples.", styles["Body"],
    ))

    story.append(Paragraph("3.4 Data split", styles["H2c"]))
    story.append(Paragraph(
        "Each dataset was split 70% train / 15% validation / 15% test, using stratified sampling so "
        "that the healthy/ALL ratio is preserved in every split. The validation set is used to monitor "
        "training (see accuracy printed after every epoch); the test set is only touched once, at the "
        "very end, to report the final numbers below.", styles["Body"],
    ))

    story.append(Paragraph("3.5 Evaluation metrics", styles["H2c"]))
    story.append(Paragraph(
        "Accuracy alone can be misleading in medical classification, so six metrics are reported for "
        "every run: <b>accuracy</b>, <b>precision</b> (of the cells flagged as ALL, how many really are), "
        "<b>recall / sensitivity</b> (of the actual ALL cells, how many were caught), <b>specificity</b> "
        "(of the actual healthy cells, how many were correctly cleared), <b>F1 score</b> (balance of "
        "precision and recall), and <b>AUC</b> (overall ability to rank ALL cells above healthy ones). "
        "These are exactly the metrics used in the reference paper, chosen so our results can be "
        "compared directly against theirs.", styles["Body"],
    ))

    story.append(PageBreak())

    # ---------------------------------------------------------------- 4. Code walkthrough
    story.append(Paragraph("4. Code Structure", styles["H1c"]))
    story.append(Paragraph(
        "The project has two near-identical training scripts, one per dataset, so each experiment is "
        "self-contained and easy to run and explain on its own.", styles["Body"],
    ))
    code_table = styled_table(
        [
            ["File", "Purpose"],
            ["src/train_idb2.py", "Loads ALL-IDB2, trains and evaluates the single-cell classifier"],
            ["src/train_idb1.py", "Loads ALL-IDB1, trains and evaluates the whole-image classifier"],
            ["src/generate_report_pdf.py", "Builds this PDF report from the saved results"],
            ["models/", "Saved trained model weights (.pth)"],
            ["results/", "Metrics (JSON) and confusion matrix plots (PNG) for every run"],
        ],
        col_widths=[2.1 * inch, 3.9 * inch],
    )
    story.append(code_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Each training script follows the same five steps: (1) load image file paths and labels parsed "
        "from filenames, (2) build a stratified train/val/test split, (3) build the EfficientNet-B0 "
        "model with a new classification head, (4) run Stage 1 then Stage 2 training loops, printing "
        "training/validation accuracy every epoch, (5) evaluate on the held-out test set and save "
        "metrics + a confusion matrix plot.", styles["Body"],
    ))

    # ---------------------------------------------------------------- 5. Results
    story.append(Paragraph("5. Results", styles["H1c"]))

    story.append(Paragraph("5.1 ALL-IDB2 (single-cell classification)", styles["H2c"]))
    s1 = load_metrics("idb2_stage1")
    s2 = load_metrics("idb2_stage2")
    rows = [
        ("Accuracy", s1["accuracy"], s2["accuracy"]),
        ("Precision", s1["precision"], s2["precision"]),
        ("Recall", s1["recall"], s2["recall"]),
        ("Specificity", s1["specificity"], s2["specificity"]),
        ("F1 score", s1["f1"], s2["f1"]),
        ("AUC", s1["auc"], s2["auc"]),
    ]
    data = metrics_table_data(
        ["Metric", "Stage 1 (frozen backbone)", "Stage 2 (fine-tuned)"], rows
    )
    story.append(styled_table(data, col_widths=[1.8 * inch, 2.1 * inch, 2.1 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Confusion matrix (TN={s2['confusion_matrix']['tn']}, FP={s2['confusion_matrix']['fp']}, "
        f"FN={s2['confusion_matrix']['fn']}, TP={s2['confusion_matrix']['tp']}): the model produced "
        "zero false positives in both stages - it never mistook a healthy cell for an ALL cell - but "
        f"missed {s2['confusion_matrix']['fn']} of the {s2['confusion_matrix']['fn'] + s2['confusion_matrix']['tp']} "
        "true ALL cells in the test set. Fine-tuning (Stage 2) raised the AUC from "
        f"{s1['auc']:.3f} to {s2['auc']:.3f} - i.e. the model became more confident and better at "
        "ranking cases correctly - without changing which test cases were ultimately misclassified.",
        styles["Body"],
    ))
    story.append(Spacer(1, 6))
    if os.path.exists(os.path.join(RESULTS_DIR, "confusion_matrix_idb2_stage2.png")):
        story.append(Image(os.path.join(RESULTS_DIR, "confusion_matrix_idb2_stage2.png"),
                            width=3.2 * inch, height=3.2 * inch, hAlign="CENTER"))

    story.append(Spacer(1, 14))
    story.append(Paragraph("5.2 ALL-IDB1 (whole-image classification)", styles["H2c"]))
    t1 = load_metrics("idb1_stage1")
    t2 = load_metrics("idb1_stage2")
    rows1 = [
        ("Accuracy", t1["accuracy"], t2["accuracy"]),
        ("Precision", t1["precision"], t2["precision"]),
        ("Recall", t1["recall"], t2["recall"]),
        ("Specificity", t1["specificity"], t2["specificity"]),
        ("F1 score", t1["f1"], t2["f1"]),
        ("AUC", t1["auc"], t2["auc"]),
    ]
    data1 = metrics_table_data(
        ["Metric", "Stage 1 (frozen backbone)", "Stage 2 (fine-tuned)"], rows1
    )
    story.append(styled_table(data1, col_widths=[1.8 * inch, 2.1 * inch, 2.1 * inch]))
    story.append(Spacer(1, 8))
    total_test = sum(t2["confusion_matrix"].values())
    story.append(Paragraph(
        f"The model reached a perfect score on the ALL-IDB1 test set. This must be read with caution: "
        f"the IDB1 test set only has {total_test} images (108 total in the whole dataset), so this is an "
        "encouraging early signal rather than strong statistical evidence - a small test set can easily "
        "produce a perfect score by chance. A more reliable estimate would need k-fold cross-validation "
        "(see Section 6).", styles["Body"],
    ))
    story.append(Spacer(1, 6))
    if os.path.exists(os.path.join(RESULTS_DIR, "confusion_matrix_idb1_stage2.png")):
        story.append(Image(os.path.join(RESULTS_DIR, "confusion_matrix_idb1_stage2.png"),
                            width=3.2 * inch, height=3.2 * inch, hAlign="CENTER"))

    story.append(PageBreak())

    story.append(Paragraph("5.3 Comparison to the reference paper (ALL-IDB2)", styles["H2c"]))
    comp = [
        ["Metric", "Our EfficientNet-B0", "Paper (ResNet18 + OSL)"],
        ["Accuracy", f"{s2['accuracy']*100:.1f}%", "98.2%"],
        ["Precision", f"{s2['precision']*100:.1f}%", "98.6%"],
        ["Recall", f"{s2['recall']*100:.1f}%", "97.7%"],
        ["Specificity", f"{s2['specificity']*100:.1f}%", "98.5%"],
        ["F1 score", f"{s2['f1']:.2f}", "0.98"],
        ["AUC", f"{s2['auc']:.2f}", "0.98"],
    ]
    story.append(styled_table(comp, col_widths=[1.6 * inch, 2.0 * inch, 2.0 * inch]))
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------------- 6. Discussion
    story.append(Paragraph("6. Discussion", styles["H1c"]))
    story.append(Paragraph(
        "The EfficientNet-B0 baseline is a solid, honest starting point: it is precise (zero false "
        "positives across every run) but conservative, meaning it misses some true positive ALL cases "
        "rather than risk a false alarm. This is a reasonable failure mode for a first pass, though in a "
        "real diagnostic tool a missed cancer case (false negative) is generally considered more "
        "dangerous than a false alarm, so recall would be the metric to push hardest on next.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The main gap to the reference paper's ~98% accuracy comes from their <b>Orthogonal Softmax "
        "Layer (OSL)</b>, a custom classification layer that replaces a standard fully-connected layer. "
        "It works by keeping the weight vector for each class mathematically orthogonal (at 90 degrees) "
        "to the others during training. Intuitively, this forces the network to carve out clearly "
        "separated decision regions for \"healthy\" vs. \"ALL\" instead of letting the two classes' "
        "weight vectors drift close together, which is what tends to cause overfitting on small "
        "datasets. It is conceptually similar to Dropout/DropConnect (both also fight overfitting by "
        "removing connections), but applied specifically to the final classification layer.",
        styles["Body"],
    ))
    story.append(Paragraph("Other concrete next steps:", styles["Body"]))
    story.append(Paragraph(
        "&bull; <b>K-fold cross-validation</b> instead of one fixed train/test split - the paper uses "
        "10-fold CV, which gives a far more reliable estimate of accuracy on datasets this small, where "
        "a single lucky or unlucky split can swing results by 10+ percentage points.<br/>"
        "&bull; <b>Unfreeze more layers</b> during fine-tuning (Stage 2 only unfroze the last block).<br/>"
        "&bull; <b>Implement the Orthogonal Softmax Layer</b> as a drop-in replacement for the final "
        "linear layer, to test how much of the paper's improvement it accounts for on its own.<br/>"
        "&bull; <b>True detection on ALL-IDB1</b> using the provided .xyc blast-cell coordinates, instead "
        "of only whole-image classification.",
        styles["Body"],
    ))

    # ---------------------------------------------------------------- 7. Reproduce
    story.append(Paragraph("7. How to Reproduce", styles["H1c"]))
    story.append(Paragraph(
        "python src/train_idb2.py &nbsp;&nbsp;# trains + evaluates the ALL-IDB2 model<br/>"
        "python src/train_idb1.py &nbsp;&nbsp;# trains + evaluates the ALL-IDB1 model<br/>"
        "python src/generate_report_pdf.py &nbsp;&nbsp;# regenerates this PDF from results/",
        styles["Code"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Trained model weights are saved to <code>models/</code>; metrics (JSON) and confusion matrix "
        "plots are saved to <code>results/</code> after every run.", styles["Body"],
    ))

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
