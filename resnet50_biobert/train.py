import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime

from resnet50_biobert.config import Config
from resnet50_biobert.dataset import PathVQADataset
from resnet50_biobert.model_cnn import ImageFeatureExtractor
from resnet50_biobert.model_text import BioBERTBiLSTM
from resnet50_biobert.model_transformer import TraPVQA
from transformers import AutoTokenizer

def train_loop(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask, decoder_input_ids)
        loss = criterion(logits.transpose(1, 2), decoder_input_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Train Batch {batch_idx}/{len(dataloader)} | Batch Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def val_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)

            logits = model(images, input_ids, attention_mask, decoder_input_ids)
            loss = criterion(logits.transpose(1, 2), decoder_input_ids)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    tokenizer = AutoTokenizer.from_pretrained(cfg.BIOBERT_MODEL)

    train_arrow_dir = cfg.TRAIN_ARROW_DIR
    val_arrow_dir   = cfg.VAL_ARROW_DIR
    image_dir = cfg.DATA_DIR

    train_ds = PathVQADataset(
        arrow_dir=train_arrow_dir,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=cfg.MAX_QUESTION_LEN
    )
    val_ds = PathVQADataset(
        arrow_dir=val_arrow_dir,
        image_dir=image_dir,
        tokenizer=tokenizer,
        max_length=cfg.MAX_QUESTION_LEN
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

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

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    # Listen für Logs
    train_logs = []
    val_logs = []

    best_val_loss = float("inf")
    for epoch in range(cfg.EPOCHS):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_loss   = val_loop(model,   val_loader,   criterion, device)

        print(f"[Epoch {epoch+1}/{cfg.EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Logs sammeln
        train_logs.append([epoch+1, train_loss])
        val_logs.append([epoch+1, val_loss])

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_name = f"resnet50_biobert_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
            save_path = os.path.join(cfg.MODEL_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at {save_path}")

    # CSV-Logs speichern in docs/
    os.makedirs("docs", exist_ok=True)

    train_csv_path = os.path.join("docs", "train_logs.csv")
    val_csv_path   = os.path.join("docs", "val_logs.csv")

    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        writer.writerows(train_logs)
    print(f"Train logs of resnet50_biobert saved to {train_csv_path}")

    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss"])
        writer.writerows(val_logs)
    print(f"Validation logs of resnet50_biobert saved to {val_csv_path}")

if __name__ == "__main__":
    main()
