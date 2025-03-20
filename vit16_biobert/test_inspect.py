# test_inspect.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vit16_biobert.config import Config
from vit16_biobert.dataset import PathVQADataset
from vit16_biobert.model_vit import VisionTransformerExtractor
from vit16_biobert.model_text import BioBERTBiLSTM
from vit16_biobert.model_transformer import TraPVQA


def main():
    # Konfiguration und Gerät initialisieren
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(cfg.BIOBERT_MODEL)

    # Test-Datensatz laden (verwende den in der Config definierten Test-Pfad)
    test_ds = PathVQADataset(
        arrow_dir=cfg.TEST_ARROW_DIR,
        image_dir=cfg.DATA_DIR,
        tokenizer=tokenizer,
        max_length=cfg.MAX_QUESTION_LEN
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    # Modelle initialisieren – gleiche Architektur wie beim Training
    text_encoder = BioBERTBiLSTM(
        biobert_model=cfg.BIOBERT_MODEL,
        lstm_hidden=256,
        dropout=cfg.DROPOUT,
        max_len=cfg.MAX_QUESTION_LEN
    )
    image_encoder = VisionTransformerExtractor(
        model_name="google/vit-base-patch16-224",
        out_dim=512,
        dropout=cfg.DROPOUT
    )

    # Gesamtmodell zusammenbauen
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

    # Lade deinen gespeicherten Checkpoint (verwende den Pfad deines trainierten Modells)
    checkpoint_path = os.path.join(cfg.MODEL_DIR, "vit16_biobert_2025-03-16_22-34-37.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Checkpoint {checkpoint_path} wurde geladen.")
    else:
        print(f"Checkpoint {checkpoint_path} wurde nicht gefunden. Bitte überprüfe den Pfad.")
        return

    # Schalte das Modell in den Evaluierungsmodus
    model.eval()

    # Anzahl der zu inspizierenden Beispiele (anpassen, falls gewünscht)
    num_examples = 500
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_examples:
                break

            # Lade Batch-Daten
            images = batch["image"].to(device)  # (1, 3, 224, 224)
            input_ids = batch["input_ids"].to(device)  # (1, lQ)
            attention_mask = batch["attention_mask"].to(device)  # (1, lQ)
            gold_answer = batch["answer_text"][0]  # Richtige Antwort als String

            # Inferenz: Ohne decoder_input_ids wird der Inferenzmodus genutzt
            outputs = model(images, input_ids, attention_mask)
            # outputs hat Dimension (B, max_len) mit B=1
            pred_tokens = outputs[0].tolist()
            # Token in lesbaren Text umwandeln (skip_special_tokens entfernt Sondertokens)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)

            print(f"Beispiel {i + 1}:")
            print(f"Frage: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
            print(f"Modell-Antwort: {pred_text}")
            print(f"Richtige-Antwort: {gold_answer}")
            print("-" * 50)


if __name__ == "__main__":
    main()
