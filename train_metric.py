"""
Train the face embedding network using triplet loss (metric learning).
The model learns to map same-person faces close together in embedding space
and different-person faces far apart.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import FaceEmbeddingNet, TripletLoss
from datasets import TripletDataset, face_train_transform
import config


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        loss = criterion(emb_anchor, emb_positive, emb_negative)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    train_set = TripletDataset(config.DATA_DIR, transform=face_train_transform)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # no classification head for metric learning
    model = FaceEmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=None,
        pretrained=True
    ).to(device)

    criterion = TripletLoss(margin=config.MARGIN)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{config.EPOCHS} | Triplet Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            save_path = os.path.join(config.MODEL_SAVE_DIR, 'face_metric.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model (loss: {best_loss:.4f})")

    print(f"\nDone. Best triplet loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
