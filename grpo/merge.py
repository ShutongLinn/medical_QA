from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
LORA_PATH = "grpo_lora_path"
MERGE_PATH = "grpo_merged_model"

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)

model = PeftModel.from_pretrained(
    base,
    LORA_PATH
)

model = model.merge_and_unload()

model.save_pretrained(MERGE_PATH)