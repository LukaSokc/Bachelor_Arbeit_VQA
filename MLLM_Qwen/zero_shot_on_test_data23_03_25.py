import csv
import os
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from pathlib import Path


def main():
    project_root = Path.cwd().parent
    data_path = project_root / "data" / "validation"

    # Lade den Datensatz
    dataset = load_from_disk(str(data_path))

    # Modell und Prozessor laden
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Definiere den relativen Pfad zur CSV-Datei (z.B. project_root/docs/docs/results_zero_shot_qwen_test.csv)
    csv_file = project_root / "docs" / "docs" / "results_zero_shot_qwen_val_without_system_message.csv"

    # Stelle sicher, dass der Zielordner existiert
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Bestimme die aktuelle letzte ID, falls die Datei bereits existiert,
    # andernfalls initialisiere sie mit 0 und schreibe den Header.
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
            writer = csv.DictWriter(
                f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL
            )
            writer.writeheader()

    # Iteriere über alle Samples im Testdatensatz und speichere nach jeder Antwort in die CSV
    for idx, sample in enumerate(dataset):
        image = sample["image"]
        question = sample["question"]
        correct_answer = sample["answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=96)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        model_output = output_text[0]

        # Erhöhe die ID und schreibe die Zeile in die CSV-Datei
        last_id += 1
        row = {
            "ID": last_id,
            "question": question,
            "correct_answer": correct_answer,
            "model_output": model_output
        }

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL
            )
            writer.writerow(row)

        print(f"Sample {idx} verarbeitet und in CSV abgespeichert.")


if __name__ == '__main__':
    main()
