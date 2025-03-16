# src/model_vit.py

import torch
import torch.nn as nn
from transformers import ViTModel

class VisionTransformerExtractor(nn.Module):
    """
    Lädt ein vortrainiertes ViT-Base-Patch16-224 (google/vit-base-patch16-224)
    und projiziert die Features ggf. auf out_dim (z.B. 512).
    """
    def __init__(self, model_name="google/vit-base-patch16-224", out_dim=512, dropout=0.1):
        super().__init__()
        # 1) Vortrainiertes ViT-Base-Patch16-224 laden
        self.vit = ViTModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size  # Bei ViT-Base: 768

        # 2) Linear + Dropout, um die Features (768) auf out_dim (z.B. 512) zu bringen
        self.linear = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.out_dim = out_dim

    def forward(self, pixel_values):
        """
        Args:
          pixel_values: (B, 3, 224, 224) normalisierte Bilder
        Return:
          feats: (B, seq_len, out_dim)
        """
        # Vorwärtsdurchlauf durch das ViT
        outputs = self.vit(pixel_values=pixel_values)
        # outputs.last_hidden_state => (B, seq_len, hidden_size=768)

        # Projektion => (B, seq_len, out_dim)
        feats = self.linear(outputs.last_hidden_state)
        feats = self.dropout(feats)

        # feats hat Dimension (B, seq_len=197, out_dim)
        return feats
