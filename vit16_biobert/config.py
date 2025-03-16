"""
src/config.py

Globale Konfigurationseinstellungen für dein Projekt:
- Pfade
- Hyperparameter
- Trainings-Settings
"""

import os
from dataclasses import dataclass

@dataclass
class Config:
    # Pfade
    DATA_DIR: str = os.path.join("data")
    PROCESSED_DATA_DIR: str = os.path.join("data")
    LOG_DIR: str = "logs"
    MODEL_DIR: str = "models"
    DOCS_DIR: str = "docs"

    # Neuer Arrow-Pfad
    TRAIN_ARROW_DIR: str = os.path.join("data", "train")  # z. B. data/raw/train/
    VAL_ARROW_DIR: str = os.path.join("data", "validation")  # z. B. data/raw/validation/
    TEST_ARROW_DIR: str = os.path.join("data", "test")

    # Trainings-Hyperparameter
    EPOCHS: int = 5
    BATCH_SIZE: int = 8
    LR: float = 1e-4
    SEED: int = 42

    # Modell-Hyperparameter
    MAX_QUESTION_LEN: int = 40
    NHEAD: int = 8
    NUM_ENCODER_LAYERS: int = 2
    NUM_DECODER_LAYERS: int = 2
    DIM_FEEDFORWARD: int = 2048
    DROPOUT: float = 0.1

    # BioBERT
    BIOBERT_MODEL: str = "dmis-lab/biobert-v1.1"

    # Decoder-Vokabular
    VOCAB_SIZE: int = 30522  # BERT-Base Size oder eigenes Vokabular
