"""
src/xai.py

Funktionen für Erklärbarkeit (XAI):
- Grad-CAM für ResNet50-Bilder
- SHAP für BioBERT-Text
"""

import os
import torch
import numpy as np
import cv2
import shap

from torch.nn import functional as F


def apply_grad_cam(model, dataloader, device, save_dir="logs/gradcam", num_examples=4):
    """
    Führt Grad-CAM auf einigen Beispielen aus dem DataLoader aus.
    Speichert Heatmaps als PNG in save_dir.

    Args:
        model: Dein TraPVQA-Modell (ResNet50 als CNN-Backbone).
        dataloader: DataLoader mit "image", "input_ids", ...
        device: "cuda" oder "cpu"
        save_dir: Ordner, in dem die Ergebnisse gespeichert werden
        num_examples: Anzahl der Bilder, die wir visualisieren
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1) Wir nehmen ein paar Beispiele (z.B. das erste Batch)
    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    # 2) Finde das CNN-Modul in unserem Modell
    #    Annahme: model.image_encoder = ImageFeatureExtractor
    #    Falls dein Code anders heißt, anpassen
    cnn_module = model.image_encoder.backbone  # ResNet50 ohne FC

    # 3) Grad-CAM-Hilfsfunktion
    def forward_and_grad(images_tensor):
        """
        Wir berechnen Feature Maps und Gradienten.
        - images_tensor: (B, 3, H, W) muss grad-fähig sein.
        """
        images_tensor.requires_grad = True
        feats = cnn_module(images_tensor)  # z.B. (B, 2048, 7, 7)
        # Ein sehr vereinfachtes Grad-CAM: wir nehmen den Mittelwert aller Features
        score = feats.mean()
        # Rückwärts
        model.zero_grad()
        score.backward(retain_graph=True)
        return images_tensor.grad, feats

    # 4) Gradienten und Feature-Maps
    grad, feats = forward_and_grad(images)

    # 5) Erzeuge Heatmaps
    # feats z.B. (B, 2048, 7, 7) -> mitteln => (B, 7, 7)
    # raw_img => (H, W, 3)
    for i in range(min(images.size(0), num_examples)):
        raw_img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        raw_img = (raw_img * 255).astype(np.uint8)

        # Feature Map => Mittelung über Channel
        feat_map = feats[i].mean(dim=0).detach().cpu().numpy()  # (7, 7) z.B. ResNet50
        feat_map = np.maximum(feat_map, 0)  # ReLU
        feat_map = cv2.resize(feat_map, (raw_img.shape[1], raw_img.shape[0]))  # Resize auf Originalgröße

        # Normalisierung [0..1]
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        feat_map = np.uint8(255 * feat_map)
        heatmap = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)

        # Überlagern
        superimposed = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        out_path = os.path.join(save_dir, f"gradcam_example_{i}.png")
        cv2.imwrite(out_path, superimposed)
        print(f"Grad-CAM gespeichert: {out_path}")


def explain_text_shap(model, tokenizer, texts, device, save_dir="logs/shap"):
    """
    SHAP-Erklärung für BioBERT-Text.
    Zeigt, welche Tokens den größten Einfluss auf das Modell haben.

    Annahme:
      - model.text_encoder ist dein BioBERT+BiLSTM-Encoder
      - Du hast ggf. ein Klassifikations-Logit oder Score, das wir SHAP geben.
      - Hier: Wir machen nur ein Dummy-Scoring (z.B. mittlerer Feature-Wert).

    Args:
        model: Dein TraPVQA-Modell
        tokenizer: z.B. AutoTokenizer(BioBERT)
        texts: Liste von Strings (Sätzen/Fragen)
        device: "cuda" oder "cpu"
        save_dir: Ordner für SHAP-Visualisierung
    """
    os.makedirs(save_dir, exist_ok=True)

    # Wir definieren eine Funktion f, die SHAP erklären soll.
    # f: List[str] -> np.array shape (B,)
    def f(texts_list):
        # 1) Tokenisieren
        enc = tokenizer(texts_list, return_tensors="pt", padding=True, truncation=True).to(device)
        # 2) Vorwärts
        with torch.no_grad():
            # Dein Text-Encoder => (B, seq_len, 512)
            text_feats = model.text_encoder(enc["input_ids"], enc["attention_mask"])
            # Wir nehmen den Mittelwert pro Batch => (B,)
            # In deinem Fall könntest du stattdessen "Klassifikationslogit" nehmen.
            score = text_feats.mean(dim=(1, 2))  # Mittelwert über seq_len und hidden_dim
        return score.detach().cpu().numpy()

    # 2) SHAP-Explainer
    explainer = shap.Explainer(f, tokenizer)
    # 3) SHAP-Werte berechnen
    shap_values = explainer(texts)

    # 4) Visualisierung
    # Zeigen wir nur die erste Text-Erklärung
    shap.plots.text(shap_values[0], display=True)

    # 5) Speichern als HTML
    shap_save_path = os.path.join(save_dir, "shap_text_explanation.html")
    shap.plots.save_html(shap_values[0], shap_save_path)
    print(f"SHAP-Text-Explanation gespeichert: {shap_save_path}")
