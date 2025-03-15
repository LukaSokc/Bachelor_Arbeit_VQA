"""
src/dataset.py

Definiert das Dataset für PathVQA, wobei die Daten
als .arrow-Dateien (Hugging Face Arrow-Dataset) vorliegen.

Benötigt:
 - arrow_dir (Pfad zum Disk-basierten Dataset oder DatasetDict)
 - image_dir (Ordner mit den Bildern)
 - tokenizer (BioBERT)
 - max_length (max. Token-Länge)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Wichtig: installiere 'datasets' (pip install datasets)
from datasets import load_from_disk

class PathVQADataset(Dataset):
    def __init__(self, arrow_dir, image_dir, tokenizer, max_length=40):
        """
        Args:
            arrow_dir (str): Pfad zum Ordner mit Arrow-Dateien (z.B. data/raw/train/)
                             oder zum gesamten DatasetDict.
            image_dir (str): Ordner mit Bildern (z.B. data/raw/images/)
            tokenizer: BERT-Tokenizer (z.B. BioBERT)
            max_length (int): maximale Token-Länge für die Frage
        """

        # 1) Dataset laden (Arrow)
        #    Falls arrow_dir ein einzelner Datensatz ist, bekommst du "Dataset" zurück.
        #    Falls arrow_dir ein Ordner mit train/val/test, bekommst du "DatasetDict".
        #    Hier nehmen wir an, dass arrow_dir = "data/raw/train/" etc. => Single Dataset
        dataset_or_dict = load_from_disk(arrow_dir)

        # Falls du ein DatasetDict hast, wähle hier den Split aus. Beispiel:
        # if isinstance(dataset_or_dict, DatasetDict):
        #     self.dataset = dataset_or_dict["train"]
        # else:
        #     self.dataset = dataset_or_dict

        # Wir gehen hier von einem einzelnen Dataset aus:
        self.dataset = dataset_or_dict

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Bild-Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # sample["image"] ist ein PIL-Bild (evtl. RGBA)
        image = sample["image"]

        # Bild in RGB konvertieren (falls es nicht schon RGB ist)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Jetzt anwenden wir self.transform (Resize, ToTensor, etc.)
        if self.transform is not None:
            image = self.transform(image)

        question = str(sample["question"])
        answer = str(sample["answer"])

        # Frage tokenisieren
        encoded = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

        # Dummy Decoder Input
        decoder_input_ids = torch.tensor([2, 101, 102, 3], dtype=torch.long)

        return {
            "image": image,  # => (3, 224, 224) dank Convert("RGB")
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "answer_text": answer
        }

