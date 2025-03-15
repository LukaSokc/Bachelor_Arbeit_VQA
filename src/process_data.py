"""
src/process_data.py

Lädt deine Arrow-Dateien (Hugging Face Dataset) aus dem Ordner "raw"
(train/validation/test), verarbeitet sie und speichert sie in "processed"
(als Arrow oder CSV).
"""

import argparse
import os
from datasets import load_from_disk, DatasetDict


def clean_text(text: str) -> str:
    """
    Beispiel-Funktion für Textbereinigung. Passe sie an deine Bedürfnisse an.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Optional: text = text.lower()
    # Hier kannst du Regex-Filter, Token-Cleaning etc. einbauen.
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Pfad zum Verzeichnis mit train/validation/test (Arrow-Dateien)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Zielverzeichnis für die verarbeiteten Daten"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="arrow",
        choices=["arrow", "csv"],
        help="Ausgabeformat: 'arrow' (Hugging Face Dataset) oder 'csv'"
    )
    args = parser.parse_args()

    # 1. Lade das DatasetDict von 'raw'
    #    Wenn dein Ordner "raw" direkt die Unterordner train/validation/test enthält,
    #    kannst du das gesamte Verzeichnis laden:
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Eingabeverzeichnis existiert nicht: {args.input_dir}")

    # load_from_disk lädt entweder ein einzelnes Dataset
    # oder ein DatasetDict, wenn train/validation/test vorhanden sind.
    dataset = load_from_disk(args.input_dir)  # dataset ist vom Typ DatasetDict

    if not isinstance(dataset, DatasetDict):
        raise ValueError("Die geladene Struktur ist kein DatasetDict (train/validation/test).")

    print(f"DatasetDict geladen: {list(dataset.keys())}")

    # Erwartet: ['train', 'validation', 'test'] oder ein Teil davon

    # 2. Verarbeitung/Mapping
    #    Beispiel: Textbereinigung in den Spalten "question" und "answer"
    def preprocess_function(example):
        if "question" in example:
            example["question"] = clean_text(example["question"])
        if "answer" in example:
            example["answer"] = clean_text(example["answer"])
        return example

    # Wir wenden preprocess_function auf alle Splits an (train, val, test)
    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].map(preprocess_function)
        print(f"Split '{split_name}' hat jetzt {len(dataset[split_name])} Einträge.")

    # 3. Erzeuge Ausgabeverzeichnis
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Speichern je nach Format
    if args.save_format == "arrow":
        # Speichert den gesamten DatasetDict als Arrow
        dataset.save_to_disk(args.output_dir)
        print(f"Gesamtes DatasetDict als Arrow in '{args.output_dir}' gespeichert.")
    else:
        # CSV-Ausgabe: Für jeden Split eine CSV-Datei erstellen
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            out_csv = os.path.join(args.output_dir, f"{split_name}.csv")
            df.to_csv(out_csv, index=False)
            print(f"Split '{split_name}' als CSV gespeichert: {out_csv}")


if __name__ == "__main__":
    main()
