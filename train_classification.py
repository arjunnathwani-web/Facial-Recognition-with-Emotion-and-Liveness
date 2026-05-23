"""
Train the face embedding network using classification loss (softmax cross-entropy).
After training, the embedding layer is used as the face descriptor for verification.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import FaceEmbeddingNet
from datasets import ClassificationDataset, face_train_transform, face_val_transform
import config


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # return logits not embedding
        logits = model(imgs, return_embedding=False)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, return_embedding=False)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    train_set = ClassificationDataset(config.DATA_DIR, split='train', transform=face_train_transform)
    val_set = ClassificationDataset(config.DATA_DIR, split='val', transform=face_val_transform)

    num_classes = len(train_set.classes)
    print(f"Classes found: {num_classes}")
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = FaceEmbeddingNet(
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # only update params that need gradients (frozen layers excluded)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # save num_classes so evaluate.py knows what architecture to load
    with open(os.path.join(config.MODEL_SAVE_DIR, 'classification_meta.json'), 'w') as f:
        json.dump({'num_classes': num_classes, 'embedding_dim': config.EMBEDDING_DIM}, f)

    best_val_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.MODEL_SAVE_DIR, 'face_classification.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model (val acc: {best_val_acc:.4f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
