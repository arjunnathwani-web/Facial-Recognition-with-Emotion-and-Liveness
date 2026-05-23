import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _load_mobilenet(pretrained):
    # torchvision >= 0.13 uses weights= instead of pretrained=
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v2(weights=weights)
    except AttributeError:
        return models.mobilenet_v2(pretrained=pretrained)


def _load_resnet18(pretrained):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=pretrained)


class FaceEmbeddingNet(nn.Module):
    """
    CNN backbone for face verification using MobileNetV2.
    Used for both classification-based and metric learning approaches.
    Early layers are frozen, later layers are fine-tuned.
    """

    def __init__(self, embedding_dim=128, num_classes=None, pretrained=True):
        super(FaceEmbeddingNet, self).__init__()

        # load pretrained mobilenet v2
        self.backbone = _load_mobilenet(pretrained)

        # freeze the first 10 feature blocks to preserve low-level features
        for i, layer in enumerate(self.backbone.features):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False

        # get feature size before the classifier
        # mobilenet v2 outputs 1280-dim vector after adaptive avg pool
        in_features = self.backbone.classifier[1].in_features  # 1280

        # replace original classifier with identity so we just get the feature vector
        self.backbone.classifier = nn.Identity()

        # our embedding head maps to a compact face embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # classification head is only used during classification-based training
        self.num_classes = num_classes
        if num_classes is not None:
            self.fc_classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=True):
        # pass through backbone (includes adaptive avg pool + flatten internally)
        features = self.backbone(x)
        embedding = self.embedding_head(features)

        # l2 normalize so cosine similarity == dot product
        embedding = F.normalize(embedding, p=2, dim=1)

        if self.num_classes is not None and not return_embedding:
            return self.fc_classifier(embedding)

        return embedding


class EmotionNet(nn.Module):
    """
    Simple CNN for 7-class emotion detection.
    Input is 48x48 grayscale face crops.
    Architecture inspired by standard FER2013 benchmarks.
    """

    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 48 -> 24
            nn.Dropout2d(0.25),

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 24 -> 12
            nn.Dropout2d(0.25),

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 12 -> 6
            nn.Dropout2d(0.25),
        )

        # after 3 maxpool(2): 48 -> 6, so 128 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LivenessNet(nn.Module):
    """
    Binary classifier for liveness detection (real face vs spoofed face).
    Uses ResNet18 backbone with partial fine-tuning.
    """

    def __init__(self, pretrained=True):
        super(LivenessNet, self).__init__()

        self.backbone = _load_resnet18(pretrained)

        # freeze layer1 and layer2, fine-tune the rest
        freeze_layers = ['layer1', 'layer2', 'conv1', 'bn1']
        for name, param in self.backbone.named_parameters():
            if any(l in name for l in freeze_layers):
                param.requires_grad = False

        # replace final fc with binary classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)   # 0 = fake, 1 = real
        )

    def forward(self, x):
        return self.backbone(x)


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    Pushes anchor and positive together, anchor and negative apart.
    """

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        # only penalise violations (hard triplets)
        loss = torch.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
