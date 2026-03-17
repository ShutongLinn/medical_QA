# filename: inference_base.py
import os
import torch
import json
from datasets import load_from_disk
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ================= setting =================
DATA_DIR = "./gsm8k_processed_data"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_FILE = "./results/results_base.json"
MAX_SAMPLES = -1  # -1 load all data
MAX_LENGTH = 512
# ===========================================

def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"error: cannot found: {DATA_DIR}")
    
    test_dataset = load_from_disk(os.path.join(DATA_DIR, "test"))
    limit = len(test_dataset) if MAX_SAMPLES == -1 else min(MAX_SAMPLES, len(test_dataset))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=False,
        fast_inference=True,
        gpu_memory_utilization=0.9,
    )
    model.eval()

    results = []

    for i in range(limit):
        item = test_dataset[i]
        prompt = item["prompt"]
        target = item["target"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        results.append({
            "id": i,
            "prompt": prompt,
            "target": target,
            "model_output": output_text,
            "model_type": "base"
        })

        if (i + 1) % 10 == 0:
            print(f"processed: {i+1}/{limit}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"saved in: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()