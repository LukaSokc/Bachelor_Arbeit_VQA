from huggingface_hub import login
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
import torch
import csv

# 🔐 Login and model setup


model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id, use_fast = True)

# 🔍 Dtype depending on hardware support
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float

# 📁 Load your PathVQA data
project_root = Path.cwd().parent
data_path = project_root / "Bachelor_Arbeit_VQA" / "data" / "validation"  # or "val"
dataset = load_from_disk(str(data_path))

# 📄 Output CSV setup
output_file = "data/llm_answers/gemma_system_prompt.csv"
fieldnames = ["ID", "question", "correct_answer", "model_output"]

with open(output_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # 🔁 Process each question-image pair
    for idx in tqdm(range(len(dataset[:0]))):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]
        qid = sample.get("id", idx + 1)  # fallback to index if no ID

        try:
            # 🧠 Prepare chat input
            messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a medical pathology expert. Your task is to answer medical questions based solely on the visual information in the provided pathology image. Focus only on what is visible in the image — do not rely on prior medical knowledge, assumptions, or external information. Your responses should be short, factual, and medically precise, using appropriate terminology. Do not include any explanations, reasoning, or additional text. Use a consistent format, without punctuation, and avoid capitalisation unless medically required. Only return the exact answer."
}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }
]

            # 🧃 Tokenize with chat template
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=dtype)

            input_len = inputs["input_ids"].shape[-1]

            # ✨ Generate response
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                output = output[0][input_len:]

            # 📝 Decode LLM output
            llm_answer = processor.decode(output, skip_special_tokens=True).strip()

            # ✅ Write to file immediately
            writer.writerow({
                "ID": qid,
                "question": question,
                "correct_answer": ground_truth,
                "model_output": llm_answer
            })

        except Exception as e:
            print(f"❌ Error on sample {qid}: {e}")
            continue

print("✅ All results saved to gemma_system_prompt.csv")
