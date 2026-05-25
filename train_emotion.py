"""
Train the emotion detection model.
7-class classification: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
Uses FER2013 dataset. Weighted cross-entropy handles class imbalance.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import EmotionNet
from datasets import EmotionDataset, emotion_train_transform, emotion_val_transform
import config


def compute_class_weights(dataset, num_classes, device):
    counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        counts[label] += 1
    # inverse frequency weighting
    weights = counts.sum() / (num_classes * counts)
    print("Class weights:")
    for i, (cls, w) in enumerate(zip(dataset.classes, weights)):
        print(f"  {cls}: count={int(counts[i])}, weight={w:.4f}")
    return weights.to(device)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
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
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    train_set = EmotionDataset(config.EMOTION_DATA_DIR, split='train', transform=emotion_train_transform)
    val_set = EmotionDataset(config.EMOTION_DATA_DIR, split='val', transform=emotion_val_transform)

    print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # compute weights from actual dataset counts to handle class imbalance
    weights = compute_class_weights(train_set, config.NUM_EMOTIONS, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = EmotionNet(num_classes=config.NUM_EMOTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    best_val_acc = 0.0
    EPOCHS = 50

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.MODEL_SAVE_DIR, 'emotion.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model (val acc: {best_val_acc:.4f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
