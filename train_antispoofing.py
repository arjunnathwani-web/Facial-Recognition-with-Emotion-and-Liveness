"""
Train the liveness detection (anti-spoofing) model.
Binary classification: real face vs spoofed/fake face.
Dataset should be sourced from a liveness benchmark like LCC FASD or CelebA-Spoof.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import LivenessNet
from datasets import LivenessDataset, face_train_transform, face_val_transform
import config


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

    train_set = LivenessDataset(config.LIVENESS_DATA_DIR, split='train', transform=face_train_transform)
    val_set = LivenessDataset(config.LIVENESS_DATA_DIR, split='val', transform=face_val_transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = LivenessNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
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
            save_path = os.path.join(config.MODEL_SAVE_DIR, 'liveness.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model (val acc: {best_val_acc:.4f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
