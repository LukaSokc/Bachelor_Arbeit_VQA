"""
src/dataset.py

Definiert das Dataset für PathVQA, wobei die Daten
als .arrow-Dateien (Hugging Face Arrow-Dataset) vorliegen.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_from_disk

class PathVQADataset(Dataset):
    def __init__(self, arrow_dir, image_dir, tokenizer, max_length=40):
        dataset_or_dict = load_from_disk(arrow_dir)
        self.dataset = dataset_or_dict  # falls es ein single dataset ist
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Bild
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)

        # Frage
        question = str(sample["question"])
        encoded_q = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        input_ids = torch.tensor(encoded_q["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded_q["attention_mask"], dtype=torch.long)

        # Antwort
        answer = str(sample["answer"])
        encoded_a = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        # "Gold"-Label für Teacher Forcing
        decoder_input_ids = torch.tensor(encoded_a["input_ids"], dtype=torch.long)

        return {
            "image": image,                  # (3, 224, 224)
            "input_ids": input_ids,          # (lQ)
            "attention_mask": attention_mask,# (lQ)
            "decoder_input_ids": decoder_input_ids,  # (lA)
            "answer_text": answer
        }
