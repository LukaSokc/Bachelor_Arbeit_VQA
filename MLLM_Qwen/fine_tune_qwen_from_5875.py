import csv
import os
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from pathlib import Path
from transformers import BitsAndBytesConfig

# Parameter
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# System Message für medizinische Bilddiagnose
system_message = """You are a medical pathology expert. Your task is to answer medical questions based solely on the visual information in the provided pathology image. Focus only on what is visible in the image — do not rely on prior medical knowledge, assumptions, or external information. Your responses should be short, factual, and medically precise, using appropriate terminology. Do not include any explanations, reasoning, or additional text. Use a consistent format, without punctuation, and avoid capitalisation unless medically required. Only return the exact answer."""

# Funktion zum Formatieren des Prompts
def format_prompt(sample):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]}
            ]
        }
    ]

# 1. Modell laden (mit optionaler 4-Bit-Quantisierung)
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
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

# 2. Prozessor laden
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

# 3. Finetuned Adapter laden
print(f"Before adapter parameters: {model.num_parameters()}")
model.load_adapter("./output_first_fine_tune_qwen")
print(f"After adapter parameters: {model.num_parameters()}")

# 4. Validation-Datensatz von Hugging Face laden
dataset = load_dataset("dein-username/dein-datensatzname", split="validation")  # <-- HIER ersetzen

# 5. Ergebnis-CSV vorbereiten
project_root = Path.cwd().parent
csv_file = project_root / "docs" / "docs" / "results_finetuned_qwen_val_with_system_message_from_5875_05_04_25.csv"
csv_file.parent.mkdir(parents=True, exist_ok=True)

# Header anlegen, falls CSV noch nicht existiert
if not csv_file.exists():
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

# 6. Inferenz starten ab Index 5875
start_index = 5875
last_id = start_index  # IDs starten ab 5875

for idx in range(start_index, len(dataset)):
    sample = dataset[idx]
    image = sample["image"]
    question = sample["question"]
    correct_answer = sample["answer"]

    # Baue Prompt mit System-Message
    messages = format_prompt(sample)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Eingabe vorbereiten
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Antwort generieren
    generated_ids = model.generate(**inputs, max_new_tokens=96)

    # Antwort extrahieren
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    model_output = output_text[0]

    # Ergebnis speichern
    last_id += 1
    row = {
        "ID": last_id,
        "question": question,
        "correct_answer": correct_answer,
        "model_output": model_output
    }
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL)
        writer.writerow(row)

    print(f"Sample {idx} verarbeitet und in CSV abgespeichert.")
