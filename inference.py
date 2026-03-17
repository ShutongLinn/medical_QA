from unsloth import FastLanguageModel
import torch
import os
import unsloth
from datasets import load_from_disk
from transformers import AutoTokenizer

data_dir = "./medical_o1_reasoning_SFT_processed"
model_path = "grpo_merged_model" # Directory where LoRA weights and tokenizer are saved

if os.path.exists(data_dir):
    test_path = os.path.join(data_dir, "test")
    ds_test = load_from_disk(test_path)
else:
    print("error: cannot found processed dataset...")

"""### 2.3 Inference"""

def load_model_tokenizer_ft(mode_path):
  """ base_model_id = id of the based model on HF
      adapter_path = saved lora weights
  """
  # Load the base model
  model, _ = FastLanguageModel.from_pretrained(
      model_name=model_path,
      max_seq_length=1024,         # Same as during training
      load_in_4bit=False,           # Load in 4-bit for memory efficiency
      fast_inference=True,         # Enable fast inference
      gpu_memory_utilization=0.6,  # Adjust based on your GPU memory
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
#   model.load_adapter(adapter_path,adapter_name="default")
  # Set the model to evaluation mode
  model.eval()

  # Load the tokenizer
  tokenizer = AutoTokenizer.from_pretrained(mode_path)
  return model, tokenizer

model,tokenizer=load_model_tokenizer_ft(model_path)

# inference function used earlier
def inference(data, i):
  """ Inference on the i-th sample of the test data"""

  # prompt from test data: separate the answer
  sample=data[i]['prompt']
  question=sample.split("###Question###:")[1].strip()
  question=question.split("###Response###")[0].strip()

  # correct response (for comparison)
  correct_response="<think>"+sample.split("<|im_start|>assistant")[1]

  # add back <|im_start|>assistant for the model to generate
  prompt=sample.split("<|im_start|>assistant")[0]+"<|im_start|>assistant"

  # tokenize the prompt: set add_special_tokens=False because we already did it
  input_ids=tokenizer(prompt,return_tensors="pt", add_special_tokens=False).input_ids
  input_ids=input_ids.to(model.device)
  input_length=input_ids.shape[1]

  with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=1024,  # Adjust based on your needs
        temperature=0.7,  # Controls randomness (lower = more deterministic)
        top_p=0.9,        # Take the top logits with probs summing up to 0.9
        do_sample=True
    )

  output_ids_generate = output_ids[:, input_length:]
  outputs_text=tokenizer.decode(output_ids_generate[0], skip_special_tokens=True)

  # prepend <think> token
  model_response="<think>"+outputs_text

  print("Question:\n", question)
  print("-"*100)
  print("Model response:\n", model_response)
  print("-"*100)
  print("Target response:\n", correct_response)
  print("-"*100)



inference(ds_test, 0)

