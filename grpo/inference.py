"""### 1.5 Inference"""
import os
import torch
import unsloth
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel, is_bfloat16_supported

## load test datasets
data_dir = "./gsm8k_processed_data"
base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_path = "grpo_lora"  # Directory where LoRA weights and tokenizer are saved

if os.path.exists(data_dir):
    loaded_dataset = load_from_disk(data_dir)
    test_dataset = loaded_dataset["test"]
else:
    print("error: cannot found processed dataset...")


def load_model_tokenizer_ft(base_model_id, adapter_path):
  """ base_model_id = id of the based model on HF
      adapter_path = saved lora weights
  """
  # Load the base model
  model, _ = FastLanguageModel.from_pretrained(
      model_name=base_model_id,
      max_seq_length=1024,         # Same as during training
      load_in_4bit=False,           # Load in 4-bit for memory efficiency
      fast_inference=True,         # Enable fast inference
      gpu_memory_utilization=0.8,  # Adjust based on your GPU memory
  )

  # Apply LoRA weights (need to be the same as before)
  model = FastLanguageModel.get_peft_model(
      model,
      r=16,  # Same rank as during training
      target_modules=[
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj",
      ],
      lora_alpha=32,  # Same alpha as during training
      use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing if needed
      random_state=3407,  # Same random state as during training
  )
  # Load the saved LoRA weights
  model.load_adapter(adapter_path,adapter_name="default")
  # Set the model to evaluation mode
  model.eval()

  # Load the tokenizer
  tokenizer = AutoTokenizer.from_pretrained(adapter_path)
  return model, tokenizer

model,tokenizer=load_model_tokenizer_ft(base_model_id, adapter_path)

def inference(data, i):
  """ Inference on the i-th sample of the test data"""

  # load model and tokenizer

  # prompt from test data
  prompt=data[i]["prompt"]
  target=data[i]["target"]

  # tokenize the prompt
  input_ids=tokenizer(prompt,return_tensors="pt").input_ids
  input_ids=input_ids.to(model.device)
  input_length=input_ids.shape[1]

  with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=512,  # Adjust based on your needs
        temperature=0.1,  # Controls randomness (lower = more deterministic)
        top_p=0.9,        # Take the top logits with probs summing up to 0.9
        do_sample=True
    )

  output_ids_generate = output_ids[:, input_length:]
  outputs_text=tokenizer.decode(output_ids_generate[0], skip_special_tokens=True)
  full_text=prompt+outputs_text
  print(full_text)
  print("Target: ",target)

inference(test_dataset, 200)