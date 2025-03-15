from datasets import load_dataset
from pathlib import Path

current_file = Path(__file__).resolve()

project_root = current_file.parent.parent  # -> Bachelor_Arbeit_VQA/

save_path = project_root / "data"

dataset = load_dataset("flaviagiammarino/path-vqa")

dataset.save_to_disk(str(save_path))

