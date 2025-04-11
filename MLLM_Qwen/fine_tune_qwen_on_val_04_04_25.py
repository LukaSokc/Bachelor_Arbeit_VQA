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
data_path = project_root / "data" / "validation"
dataset = load_from_disk(str(data_path))

# 5. Definiere den Pfad zur Ergebnis-CSV-Datei
csv_file = project_root / "docs" / "docs" / "results_finetuned_qwen_val.csv"
csv_file.parent.mkdir(parents=True, exist_ok=True)

# Bestimme den Startwert der ID (falls die CSV bereits existiert)
if csv_file.exists():
    try:
        existing_df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL)
        if "ID" in existing_df.columns:
            last_id = int(existing_df["ID"].max())
        else:
            last_id = len(existing_df)
    except Exception as e:
        last_id = 0
else:
    last_id = 0
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
            fieldnames=["ID", "question", "correct_answer", "model_output"],
            quoting=csv.QUOTE_ALL)
        writer.writeheader()

# 6. Iteriere über den Validierungsdatensatz, generiere Antworten und speichere sie in der CSV
for idx, sample in enumerate(dataset):
    image = sample["image"]
    question = sample["question"]
    correct_answer = sample["answer"]

    # Erstelle den Prompt: Hier wird nur eine User-Message mit Bild und Frage genutzt (ohne System Message)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]

    # Erstelle den finalen Prompt-Text
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Verarbeite den Prompt, um Eingabetensoren zu erhalten
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generiere die Antwort (hier max_new_tokens=96, anpassbar)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # Entferne den Prompt-Teil (falls notwendig)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    model_output = output_text[0]

    # Erhöhe die ID und speichere das Ergebnis
    last_id += 1
    row = {
        "ID": last_id,
        "question": question,
        "correct_answer": correct_answer,
        "model_output": model_output
    }
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
            fieldnames=["ID", "question", "correct_answer", "model_output"],
            quoting=csv.QUOTE_ALL)
        writer.writerow(row)

    print(f"Sample {idx} verarbeitet und in CSV abgespeichert.")

# Beispiel für die Verwendung einer Textgenerierungsfunktion
# (Diese Funktion musst du entsprechend implementieren, um einen einzelnen Sample-Datensatz zu testen.)
# generated_text, actual_answer = text_generator(sample_data)
# print(f"Generated Answer: {generated_text}")
# print(f"Actual Answer: {actual_answer}")
