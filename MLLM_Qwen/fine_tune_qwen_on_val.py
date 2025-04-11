import csv
import os
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from pathlib import Path
from transformers import BitsAndBytesConfig  # Falls BitsAndBytes verwendet wird

# Parameter
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Vortrainiertes Modell laden (mit 4-Bit Quantisierung falls CUDA verfügbar)
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # oder torch.float16 als Fallback
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=True
    )
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        use_cache=True
    )

# 2. Prozessor laden und konfigurieren
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# 3. Adapter laden (finetuned Adapter werden z.B. unter "./output" gespeichert)
print(f"Before adapter parameters: {model.num_parameters()}")
model.load_adapter("./output_first_fine_tune_qwen")
print(f"After adapter parameters: {model.num_parameters()}")

# 4. Lade den Validation-Datensatz
project_root = Path.cwd().parent