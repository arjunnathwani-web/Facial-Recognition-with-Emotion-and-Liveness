"""
Evaluate trained face verification models.
Computes ROC curve and AUC for:
  - Classification-based model
  - Metric learning model
  - Both cosine similarity and Euclidean distance metrics
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import FaceEmbeddingNet
from datasets import VerificationDataset, face_val_transform
import config


def get_embeddings(model, loader, device):
    """Run all pairs through model and collect embeddings + labels."""
    model.eval()
    emb1_list = []
    emb2_list = []
    label_list = []

    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2 = img1.to(device), img2.to(device)
            e1 = model(img1)
            e2 = model(img2)
            emb1_list.append(e1.cpu())
            emb2_list.append(e2.cpu())
            label_list.append(label)

    return (torch.cat(emb1_list),
            torch.cat(emb2_list),
            torch.cat(label_list).numpy())


def cosine_scores(emb1, emb2):
    # embeddings are already l2 normalised, so dot product == cosine similarity
    return (emb1 * emb2).sum(dim=1).numpy()


def euclidean_scores(emb1, emb2):
    # negate distance so that higher score = more similar
    dist = torch.sum((emb1 - emb2) ** 2, dim=1)
    return -dist.numpy()


def evaluate(model_path, model_name, num_classes=None):
    device = config.DEVICE

    dataset = VerificationDataset(
        config.VERIFICATION_PAIRS,
        config.VERIFICATION_DIR,
        face_val_transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # load model - use strict=False to handle cases where classifier head is missing
    model = FaceEmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    emb1, emb2, labels = get_embeddings(model, loader, device)

    cos_sim = cosine_scores(emb1, emb2)
    euc_sim = euclidean_scores(emb1, emb2)

    auc_cos = roc_auc_score(labels, cos_sim)
    auc_euc = roc_auc_score(labels, euc_sim)

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"  Cosine Similarity AUC:   {auc_cos:.4f}")
    print(f"  Euclidean Distance AUC:  {auc_euc:.4f}")

    fpr_cos, tpr_cos, _ = roc_curve(labels, cos_sim)
    fpr_euc, tpr_euc, _ = roc_curve(labels, euc_sim)

    return fpr_cos, tpr_cos, auc_cos, fpr_euc, tpr_euc, auc_euc


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    results = {}

    # evaluate classification model
    cls_model_path = os.path.join(config.MODEL_SAVE_DIR, 'face_classification.pth')
    meta_path = os.path.join(config.MODEL_SAVE_DIR, 'classification_meta.json')

    num_classes = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            num_classes = meta.get('num_classes')

    if os.path.exists(cls_model_path):
        r = evaluate(cls_model_path, 'Classification (Softmax)', num_classes=num_classes)
        results['Classification'] = r
    else:
        print("Classification model not found, skipping.")

    # evaluate metric learning model
    metric_model_path = os.path.join(config.MODEL_SAVE_DIR, 'face_metric.pth')
    if os.path.exists(metric_model_path):
        r = evaluate(metric_model_path, 'Metric Learning (Triplet Loss)', num_classes=None)
        results['Metric Learning'] = r
    else:
        print("Metric learning model not found, skipping.")

    # plot all ROC curves together
    if results:
        plt.figure(figsize=(8, 6))
        colors = {'Classification': ('#1f77b4', '#aec7e8'),
                  'Metric Learning': ('#d62728', '#f5a9a9')}

        for name, (fpr_cos, tpr_cos, auc_cos, fpr_euc, tpr_euc, auc_euc) in results.items():
            c1, c2 = colors.get(name, ('blue', 'red'))
            plt.plot(fpr_cos, tpr_cos, color=c1, label=f'{name} - Cosine (AUC={auc_cos:.4f})')
            plt.plot(fpr_euc, tpr_euc, color=c2, linestyle='--', label=f'{name} - Euclidean (AUC={auc_euc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Face Verification')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=150)
        plt.show()
        print("\nROC curve saved to roc_curves.png")


if __name__ == '__main__':
    main()
