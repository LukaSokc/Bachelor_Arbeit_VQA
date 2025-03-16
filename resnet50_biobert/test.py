"""
src/test.py

Testskript für TraP-VQA (Arrow-Version):
- Lädt Config
- Lädt dein bestes Modell (BioBERT+BiLSTM, ResNet50)
- Lädt Test-Dataset + DataLoader (aus .arrow)
- Führt Test-Loop aus
- Speichert Testergebnisse (Loss, Accuracy, etc.) in docs/test_results.csv
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet50_biobert.config import Config
from resnet50_biobert.dataset import PathVQADataset
from resnet50_biobert.model_cnn import ImageFeatureExtractor
from resnet50_biobert.model_text import BioBERTBiLSTM
from resnet50_biobert.model_transformer import TraPVQA
from transformers import AutoTokenizer

def test_loop(model, dataloader, criterion, device):
    """
    Test-Schleife: keine Backprop, nur Forward + Metriken (Loss, z. B. Accuracy)
    Hier nutzen wir Teacher Forcing (decoder_input_ids), um denselben Pfad
    wie im Training zu durchlaufen und einen Token-Level-Loss zu messen.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)

            # Wichtig: Wir geben decoder_input_ids=decoder_input_ids mit, NICHT None!
            logits = model(images, input_ids, attention_mask, decoder_input_ids=decoder_input_ids)
            # => logits: (B, seq_len, vocab_size)

            # CrossEntropyLoss: vergleicht pro Token
            loss = criterion(logits.transpose(1, 2), decoder_input_ids)
            total_loss += loss.item()

            # Beispiel: "korrekte Tokens" - wir nehmen argmax pro Token
            preds = logits.argmax(dim=-1)  # (B, seq_len)
            # Dummy: wir zählen, wie oft die gesamte Sequenz stimmt
            # (Achtung: in einem echten generativen Modell oft SHIFT nötig)
            correct = (preds == decoder_input_ids).all(dim=1).sum().item()
            total_correct += correct
            total_samples += images.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    tokenizer = AutoTokenizer.from_pretrained(cfg.BIOBERT_MODEL)

    # Lade Testdaten
    test_arrow_dir = cfg.TEST_ARROW_DIR  # z. B. "data/raw/test"
    image_dir = cfg.DATA_DIR

    test_ds = PathVQADataset(
        arrow_dir=test_arrow_dir,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=cfg.MAX_QUESTION_LEN
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Modell initialisieren
    text_encoder = BioBERTBiLSTM(
        biobert_model=cfg.BIOBERT_MODEL,
        lstm_hidden=256,
        dropout=cfg.DROPOUT,
        max_len=cfg.MAX_QUESTION_LEN
    )
    image_encoder = ImageFeatureExtractor(
        dropout=cfg.DROPOUT,
        out_channels=512
    )

    model = TraPVQA(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        vocab_size=cfg.VOCAB_SIZE,
        nhead=cfg.NHEAD,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        dim_feedforward=cfg.DIM_FEEDFORWARD,
        dropout=cfg.DROPOUT
    ).to(device)

    # Lade Bestes Modell
    best_model_path = os.path.join(cfg.MODEL_DIR, "resnet50_biobert_2025-03-16_17-50-04.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint {best_model_path} nicht gefunden!")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Modell {best_model_path} geladen.")

    # Criterion (z. B. CrossEntropy)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Test Loop
    test_loss, test_acc = test_loop(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Ergebnisse als CSV speichern
    os.makedirs("docs", exist_ok=True)
    test_csv_path = os.path.join("docs", "test_results.csv")

    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_loss", "test_accuracy"])
        writer.writerow([test_loss, test_acc])

    print(f"Test-Ergebnisse gespeichert in {test_csv_path}")

if __name__ == "__main__":
    main()
