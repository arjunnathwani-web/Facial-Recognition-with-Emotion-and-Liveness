"""
Margin Sensitivity Analysis
---------------------------
Jaspreet Singh | Student ID: 105342118
COS30082 Applied Machine Learning - Innovative Feature

This script investigates how different triplet loss margin values affect
face verification performance on the validation set.

The trained model's embeddings are fixed at this point, so retraining
from scratch at every margin value is not practical. Instead, a post-hoc
scaling approach is used to approximate the effect of different margins
on the similarity scores, without touching the model weights.

Steps:
  1. Load the trained metric learning model (face_metric.pth)
  2. Extract embeddings for all verification pairs in one pass
  3. Apply margin-aware score scaling to simulate different margin settings
  4. Compute AUC at each margin value and plot the results
  5. Save everything to margin_analysis.csv for reference

Run with:
    python tune_margin.py

Outputs:
    margin_sensitivity.png  - AUC vs margin line plot
    margin_analysis.csv     - full results table
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models import FaceEmbeddingNet
from datasets import VerificationDataset, face_val_transform
import config


# range of margin values to test - covers both relaxed and strict settings
# around the default of 0.5 used during training
MARGINS_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]


def load_metric_model(device):
    """Load the trained triplet loss model from saved_models/."""
    model = FaceEmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=None,
        pretrained=False
    ).to(device)

    model_path = os.path.join(config.MODEL_SAVE_DIR, 'face_metric.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Metric model not found at {model_path}. "
            "Run train_metric.py first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded metric model from {model_path}")
    return model


def extract_embeddings(model, device):
    """
    Run all verification pairs through the model and collect embeddings.
    This only needs to happen once - we reuse the same embeddings for
    every margin value tested, which is what makes this approach fast.
    """
    dataset = VerificationDataset(
        config.VERIFICATION_PAIRS,
        config.VERIFICATION_DIR,
        face_val_transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    emb1_list, emb2_list, label_list = [], [], []

    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2 = img1.to(device), img2.to(device)
            e1 = model(img1)
            e2 = model(img2)
            emb1_list.append(e1.cpu())
            emb2_list.append(e2.cpu())
            label_list.append(label)

    emb1 = torch.cat(emb1_list)
    emb2 = torch.cat(emb2_list)
    labels = torch.cat(label_list).numpy()

    print(f"Extracted embeddings for {len(labels)} verification pairs.")
    return emb1, emb2, labels


def margin_scaled_scores(emb1, emb2, margin):
    """
    Apply a margin-aware scaling to cosine similarity scores.

    In triplet loss training, a larger margin pushes the model to create
    more separation between positive and negative pairs. We approximate
    this post-hoc using:

        scaled_score = cosine_sim - margin * (1 - cosine_sim)

    The effect is that low-confidence matches get penalised more heavily
    as the margin increases, while high-confidence matches (cosine_sim
    close to 1) are barely affected. This simulates what a stricter
    margin would have produced during training without actually retraining.

    Note: embeddings are already L2 normalised, so the dot product
    gives us cosine similarity directly.
    """
    cos_sim = (emb1 * emb2).sum(dim=1).numpy()
    scaled = cos_sim - margin * (1.0 - cos_sim)
    return scaled


def run_margin_analysis(emb1, emb2, labels):
    """Compute AUC at each margin value and collect results."""
    results = []

    print(f"\n{'Margin':>10} {'AUC':>10}")
    print("-" * 25)

    for margin in MARGINS_TO_TEST:
        scores = margin_scaled_scores(emb1, emb2, margin)
        auc = roc_auc_score(labels, scores)
        results.append({'margin': margin, 'auc': auc})
        print(f"{margin:>10.2f} {auc:>10.4f}")

    return results


def save_results_csv(results, output_path='margin_analysis.csv'):
    """Save the margin vs AUC table to a CSV file for reference."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['margin', 'auc'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


def plot_results(results, output_path='margin_sensitivity.png'):
    """
    Plot AUC vs margin and save the figure.
    The training margin is marked with a dashed line so it is easy to
    see how the default value compares to the rest of the range.
    """
    margins = [r['margin'] for r in results]
    aucs = [r['auc'] for r in results]

    best_idx = int(np.argmax(aucs))
    best_margin = margins[best_idx]
    best_auc = aucs[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(margins, aucs, marker='o', color='#1f77b4',
            linewidth=2, markersize=7, label='AUC')

    # mark the margin that was actually used during training
    default_margin = config.MARGIN
    ax.axvline(x=default_margin, color='grey', linestyle='--',
               linewidth=1.2, label=f'Training margin ({default_margin})')

    # highlight whichever margin gave the best AUC
    ax.scatter([best_margin], [best_auc], color='#d62728', zorder=5,
               s=100, label=f'Best AUC={best_auc:.4f} at margin={best_margin}')

    ax.set_xlabel('Margin Value', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Margin Sensitivity Analysis - Triplet Loss Face Verification', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # keep the y-axis tight around the actual range so differences are visible
    ax.set_ylim(max(0, min(aucs) - 0.02), min(1.0, max(aucs) + 0.02))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Plot saved to {output_path}")

    return best_margin, best_auc


def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Testing margins: {MARGINS_TO_TEST}")
    print(f"Default training margin: {config.MARGIN}\n")

    # load model and extract embeddings for all pairs
    model = load_metric_model(device)
    emb1, emb2, labels = extract_embeddings(model, device)

    # run the margin analysis and save outputs
    results = run_margin_analysis(emb1, emb2, labels)
    save_results_csv(results)
    best_margin, best_auc = plot_results(results)

    print(f"\nSummary:")
    print(f"  Default margin (used in training): {config.MARGIN}")
    print(f"  Best margin from analysis:         {best_margin}")
    print(f"  Best AUC:                          {best_auc:.4f}")

    if best_margin != config.MARGIN:
        print(f"\n  Note: margin={best_margin} outperforms the training margin.")
        print(f"  Retraining with this margin could improve verification performance.")
    else:
        print(f"\n  The training margin of {config.MARGIN} appears well-suited to this dataset.")


if __name__ == '__main__':
    main()
