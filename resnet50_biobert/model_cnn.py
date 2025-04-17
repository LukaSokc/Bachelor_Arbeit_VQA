"""
src/model_cnn.py

ResNet50 (ohne FC) + 2D-CNN + Dense => (B, l_i, 512)
Analogie zum Paper: "Image Feature Extraction"
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ImageFeatureExtractor(nn.Module):
    def __init__(self, dropout=0.1, out_channels=512):
        super().__init__()
        base = models.resnet50(pretrained=True)
        # Entferne die letzten Schichten (AvgPool + FC)
        layers = list(base.children())[:-2]  # => (B, 2048, 7, 7)
        self.backbone = nn.Sequential(*layers)

        self.conv = nn.Conv2d(2048, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(out_channels, 512)

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        return: (B, 49, 512) => 7*7=49
        """
        feats = self.backbone(x)       # (B, 2048, 7, 7)
        feats = self.conv(feats)       # (B, 512, 7, 7)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        feats = feats.flatten(start_dim=2)  # (B, 512, 49)
        feats = feats.transpose(1, 2)       # (B, 49, 512)
        feats = self.linear(feats)         # (B, 49, 512)
        return feats
