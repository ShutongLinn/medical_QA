
from reward import correctness_reward, format_reward
from datasets import load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import os
import unsloth

data_dir = "./gsm8k_processed_data"
if os.path.exists(data_dir):
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")
    
    train_dataset = load_from_disk(train_path)
    test_dataset = load_from_disk(test_path)
else:
    print("error: cannot found processed dataset...")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-4B-Instruct-2507",
    max_seq_length = 1024, # Can increase for longer reasoning traces
    load_in_4bit = False,   # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,      # Larger rank = smarter, but slower. Suggested values 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


### Train
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 4,                  # log information every 4 steps
    bf16 = is_bfloat16_supported(),     # determine the data type used for training
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,  # Increase to 4 for smoother training
    num_generations = 2,              # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 256,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 400,
    save_steps = 400,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "grpo_lora",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        correctness_reward,
        format_reward,
    ],
    args = training_args,
    train_dataset = train_dataset,
)
trainer.train()

# save model and tokenizer
trainer.model.save_lora(training_args.output_dir)
trainer.tokenizer.save_pretrained(training_args.output_dir)