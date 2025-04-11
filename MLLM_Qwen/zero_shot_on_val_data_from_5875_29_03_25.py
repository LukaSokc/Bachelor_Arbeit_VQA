import csv
import os
import torch
import pandas as pd
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from pathlib import Path


def main():
    # Setze das Environment-Variable, um Fragmentierung zu vermeiden
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    project_root = Path.cwd().parent
    data_path = project_root / "data" / "validation"

    # Definiere den Pfad zur CSV-Datei
    csv_file = project_root / "docs" / "docs" / "results_zero_shot_qwen_validation_29_03_25.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Bestimme die letzte gespeicherte ID
    if csv_file.exists():
        try:
            existing_df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL)
            if "ID" in existing_df.columns:
                last_id = int(existing_df["ID"].max())
            else:
                last_id = len(existing_df)
        except Exception as e:
            print(f"Fehler beim Lesen der CSV-Datei: {e}")
            last_id = 0
    else:
        last_id = 0
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL
            )
            writer.writeheader()

    # Lazy-Loading des Datensatzes: Nur ab der letzten ID laden, um RAM zu sparen
    dataset = load_from_disk(str(data_path))
    if last_id < len(dataset):
        dataset = dataset.select(range(last_id, len(dataset)))
    else:
        print("Alle Daten wurden bereits verarbeitet. 🎉")
        return

    # Modell und Prozessor laden
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    model.eval()  # Setze in den Evaluierungsmodus
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Iteriere über alle Samples ab dem gespeicherten Index
    for i, sample in enumerate(dataset):
        current_id = last_id + i + 1  # Fortlaufende ID
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

        try:
            # Erzeuge den Texteingabe-String
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Verarbeite Bild- und Video-Inputs
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inferenz ohne Gradientenberechnung
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=96)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            model_output = output_text[0]

            # Schreibe das Ergebnis in die CSV
            row = {
                "ID": current_id,
                "question": question,
                "correct_answer": correct_answer,
                "model_output": model_output
            }
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL
                )
                writer.writerow(row)

            print(f"✅ Sample {current_id} verarbeitet und gespeichert.")

            # Lösche nicht mehr benötigte Variablen und räume den GPU-Speicher auf
            del inputs, generated_ids, generated_ids_trimmed, output_text
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Fehler bei Sample {current_id}: {e}")
            continue


if __name__ == '__main__':
    main()

