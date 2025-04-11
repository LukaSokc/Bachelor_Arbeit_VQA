import csv
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from pathlib import Path

def main():
    # Lade den Datensatz (Trainings- und Validierungssplit)
    dataset = load_dataset("flaviagiammarino/path-vqa")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Definiere die Few‑Shot-Indizes (Beispiele aus dem Trainingsdatensatz)
    few_shot_indices = [10658, 18497, 8273, 16324, 10392, 9073, 4623, 10336]
    few_shot_examples = []
    for idx in few_shot_indices:
        sample = train_dataset[idx]
        few_shot_examples.append({
            "question": sample["question"],
            "answer": sample["answer"],
            "image": sample["image"]
        })

    # Erstelle das Verzeichnis für die Ergebnisse und die CSV-Datei
    results_dir = Path("results_qwen")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "results_few_shot_qwen_in_context_validation_all_06_04_25.csv"

    # Schreibe den Header in die CSV-Datei (überschreibt vorhandene Datei)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

    # Lade das Modell und den Prozessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    model.eval()  # Modell in den Evaluierungsmodus schalten
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Definiere die System Message
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

    # Iteriere über alle Samples im Validierungsdatensatz
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        image = sample["image"]
        question = sample["question"]
        correct_answer = sample["answer"]

        # Erstelle die Nachrichtenliste
        # Zuerst die System Message
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            }
        ]

        # Füge alle Few‑Shot-Beispiele hinzu – Bild, Frage und Antwort in einer Nachricht
        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": "question: " + ex["question"] + "\nanswer: " + ex["answer"]}
                ]
            })

        # Füge das aktuelle Validierungssample hinzu (nur Bild und Frage, damit das Modell eine Antwort generiert)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "question: " + question},
                {"type": "text", "text": "Answer: "}
            ]
        })

        try:
            # Erzeuge den Text-Prompt aus der Nachrichtenliste
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Verarbeite die Vision-Informationen (Bilder und Videos)
            image_inputs, video_inputs = process_vision_info(messages)
            # Erstelle die finalen Inputs für das Modell
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inferenz ohne Gradientenberechnung
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=96)
                # Entferne den Input-Teil, um nur die generierte Antwort zu erhalten
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            model_output = output_text[0]

            # Verwende als ID den aktuellen Index + 1
            current_id = idx + 1

            row = {
                "ID": current_id,
                "question": question,
                "correct_answer": correct_answer,
                "model_output": model_output
            }

            # Hänge das Ergebnis an die CSV-Datei an
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["ID", "question", "correct_answer", "model_output"], quoting=csv.QUOTE_ALL)
                writer.writerow(row)

            print(f"✅ Sample {current_id} verarbeitet und gespeichert.")

        except Exception as e:
            print(f"❌ Fehler bei Sample {idx}: {e}")
            continue

if __name__ == '__main__':
    main()
