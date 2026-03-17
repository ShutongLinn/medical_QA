import os
import unsloth
from datasets import load_dataset
from transformers import AutoTokenizer

data_dir = "./medical_o1_reasoning_SFT_processed"
model_path = "./grpo/grpo_merged_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

system_prompt = (
    "You a medical expert with advanced knowledge in clinical reasoning, diagonstics, and treatment planning."
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer."
)

user_prompt= (
    "Below is a question that describes the condition of a patient. Write a response that appropriately answers the question."
    "Show your reasoning in <think> </think> tags. And return the final response in <answer> </answer> tags.\n"
    "###Question###:\n{question}\n"
    "###Response###:\n<think>"
)

def format_sample(sample):
    question = sample["Question"]
    cot = sample["Complex_CoT"]
    response = sample["Response"]

    # Format the prompt using the tokenizer's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question)},
            {"role": "assistant", "content": f"{cot}\n</think>\n<answer>\n{response}\n</answer>\n"},
        ],
        tokenize=False,  # Do not tokenize yet
        continue_final_message=True
    )

    return {"prompt": formatted_prompt}


def gen_dataset(num_samples=10000):
    """ Return:
            A generator on train data "train_gen"
    """
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split = "train",trust_remote_code=True)

    # take 10000 data, shuffle with a fixed seed for reproducibility
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    dataset=dataset.map(format_sample)

    return dataset


dataset=gen_dataset()
# split train and test data
dataset=dataset.train_test_split(test_size=0.1)
ds_train, ds_test=dataset["train"], dataset["test"]

ds_train.save_to_disk(os.path.join(data_dir, "train"))
ds_test.save_to_disk(os.path.join(data_dir, "test"))