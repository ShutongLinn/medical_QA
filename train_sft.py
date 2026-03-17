from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import os
from datasets import load_from_disk

## load test datasets
data_dir = "./medical_o1_reasoning_SFT_processed"
base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_path = "./grpo/grpo_lora"  # Directory where LoRA weights and tokenizer are saved

if os.path.exists(data_dir):
    loaded_dataset = load_from_disk(data_dir)
    
    ds_train = loaded_dataset["train"]
    ds_test = loaded_dataset["test"]
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
      gpu_memory_utilization=0.4,  # Adjust based on your GPU memory
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

# put model to train mode
model.train()
print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

training_args=SFTConfig(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, # gradient accumulation
    per_device_eval_batch_size=16, # should set to 8 for faster speed!!!
    logging_steps=50,              # log train loss every 50 steps
    evaluation_strategy="steps",   # Evaluate at the end of each epoch
    eval_steps=50,                 # Evaluate every 50 steps
    num_train_epochs = 2,          # or set max_steps=200
    warmup_ratio=0.03,             # follow QLoRA paper
    learning_rate=3e-5,
    bf16=True,        # train in bf16 for higher precision
    optim="adamw_torch_fused",
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    output_dir="qwen-grpo-medical",
    dataset_text_field="prompt",
    report_to="none",
    seed=42, # for reproducibility
)


trainer = SFTTrainer(
    model=model,
    processing_class =tokenizer,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    args=training_args,
)
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
trainer.train()

# save model and tokenizer
trainer.model.save_lora(training_args.output_dir)
trainer.tokenizer.save_pretrained(training_args.output_dir)