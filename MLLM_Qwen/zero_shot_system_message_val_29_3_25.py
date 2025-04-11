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

    # 1) Datensatz laden
    dataset = load_from_disk(str(data_path))

    # 2) Modell und Prozessor laden
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Optional: System Message
    system_message = (
        "You are a medical pathology expert. Your task is to answer medical questions "
        "based solely on the visual information in the provided pathology image. "
        "Focus only on what is visible in the image — do not rely on prior medical knowledge, "
        "assumptions, or external information. Your responses should be short, factual, "
        "and medically precise, using appropriate terminology. "
        "Do not include any explanations, reasoning, or additional text. "
        "Use a consistent format, without punctuation, and avoid capitalisation unless medically required. "
        "Only return the exact answer."
    )

    # 3) Pfad zur CSV-Datei definieren
    csv_file = project_root / "docs" / "docs" / "results_zero_shot_qwen_val_with_system_message_29_3_2025.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # 4) Prüfen, ob die Datei bereits existiert.
    if csv_file.exists():
        # Falls ja: wir hängen neue Einträge an (Modus "a"), ohne Header
        file_mode = "a"
        write_header = False
        # Letzte ID ermitteln:
        try:
            existing_df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL)
            if "ID" in existing_df.columns:
                last_id = int(existing_df["ID"].max())
            else:
                last_id = len(existing_df)
        except Exception:
            last_id = 0
    else:
        # Falls nein: neue Datei erstellen (Modus "w"), mit Header
        file_mode = "w"
        write_header = True
        last_id = 0

    # 5) Datei 1x öffnen und am Ende schließen
    with open(csv_file, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "question", "correct_answer", "model_output"],
            quoting=csv.QUOTE_ALL
        )

        # Bei neuer Datei: Header schreiben
        if write_header:
            writer.writeheader()

        # 6) Schleife über alle Samples, direkt row schreiben
        for idx, sample in enumerate(dataset):
            image = sample["image"]
            question = sample["question"]
            correct_answer = sample["answer"]

            # Nachricht (system + user)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_message}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=96)
            # Prompt entfernen
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            model_output = output_text[0]

            # Schreibe direkt in CSV
            last_id += 1
            row = {
                "ID": last_id,
                "question": question,
                "correct_answer": correct_answer,
                "model_output": model_output
            }
            writer.writerow(row)

            print(f"Sample {idx} verarbeitet -> CSV: ID={last_id}")

    print(f"Fertig! Alle neuen Einträge direkt in '{csv_file}' geschrieben.")

if __name__ == "__main__":
    main()
