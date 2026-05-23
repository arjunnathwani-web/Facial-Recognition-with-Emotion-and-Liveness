import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# transforms for face recognition (training includes augmentation)
face_train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transforms for emotion (grayscale 48x48)
emotion_train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

emotion_val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def load_img(path, mode='RGB'):
    return Image.open(path).convert(mode)


class ClassificationDataset(Dataset):
    """
    Loads face images for classification-based training.
    Expects: root_dir/train_data/<person_id>/*.jpg
    """

    def __init__(self, root_dir, split='train', transform=None):
        folder = 'train_data' if split == 'train' else (
            'val_data' if split == 'val' else 'test_data'
        )
        self.root = os.path.join(root_dir, folder)
        self.transform = transform if transform else face_train_transform

        self.classes = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_img(path)
        if self.transform:
            img = self.transform(img)
        return img, label


class TripletDataset(Dataset):
    """
    Generates random triplets (anchor, positive, negative) for metric learning.
    Positive = same person, Negative = different person.
    """

    def __init__(self, root_dir, transform=None):
        self.root = os.path.join(root_dir, 'train_data')
        self.transform = transform if transform else face_train_transform

        # build mapping: person_id -> list of image paths
        self.class_imgs = {}
        for cls in sorted(os.listdir(self.root)):
            cls_path = os.path.join(self.root, cls)
            if not os.path.isdir(cls_path):
                continue
            imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) >= 2:
                self.class_imgs[cls] = imgs

        self.classes = list(self.class_imgs.keys())
        print(f"Triplet dataset: {len(self.classes)} identities")

    def __len__(self):
        # arbitrary; each epoch samples new triplets
        return len(self.classes) * 30

    def __getitem__(self, idx):
        # randomly pick anchor/positive class and negative class
        anc_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != anc_class])

        anc_path, pos_path = random.sample(self.class_imgs[anc_class], 2)
        neg_path = random.choice(self.class_imgs[neg_class])

        anchor = self._load(anc_path)
        positive = self._load(pos_path)
        negative = self._load(neg_path)
        return anchor, positive, negative

    def _load(self, path):
        img = load_img(path)
        if self.transform:
            img = self.transform(img)
        return img


class VerificationDataset(Dataset):
    """
    Loads image pairs from the verification_pairs_val.txt file.
    Returns (img1, img2, label) where label=1 means same person.
    """

    def __init__(self, pairs_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else face_val_transform
        self.pairs = []
        self.labels = []

        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    self.pairs.append((parts[0], parts[1]))
                    self.labels.append(int(parts[2]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        img1 = load_img(os.path.join(self.data_dir, p1))
        img2 = load_img(os.path.join(self.data_dir, p2))
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, self.labels[idx]


class EmotionDataset(Dataset):
    """
    Emotion dataset (FER2013 or similar).
    Expects: root_dir/train/<emotion_class>/*.jpg
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.root = os.path.join(root_dir, split)
        self.transform = transform if transform else emotion_train_transform

        self.classes = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"Emotion classes: {self.classes}")

        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_img(path, mode='L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


class LivenessDataset(Dataset):
    """
    Liveness dataset for anti-spoofing.
    Expects: root_dir/train/real/*.jpg and root_dir/train/fake/*.jpg
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.root = os.path.join(root_dir, split)
        self.transform = transform if transform else face_train_transform
        self.samples = []

        for label, cls in enumerate(['fake', 'real']):
            cls_path = os.path.join(self.root, cls)
            if not os.path.exists(cls_path):
                print(f"Warning: {cls_path} not found")
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, fname), label))

        print(f"Liveness dataset ({split}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_img(path)
        if self.transform:
            img = self.transform(img)
        return img, label
