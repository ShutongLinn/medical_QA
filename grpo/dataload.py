"""### 1.1. Data"""

from datasets import load_dataset
from transformers import AutoTokenizer
import os

data_dir = "./gsm8k_processed_data"


def extract_hash_answer(text):
  # the final answer comes after ####
  if "####" not in text:
    return None

  return text.split("####")[1].strip()

system_prompt = (
    "You are an expert in solving high school math word problems. "
    "You first think through the reasoning process step-by-step in your mind and then provide the answer."
)

user_prompt= (
  "Solve the following question:\n{question}.\n"
  "Show your work in <think> </think> tags. And return the final answer as an integer in <answer> </answer> tags."
)

def format_prompt(tokenizer,sample):
  prompt=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt.format(question=sample["question"])},
      {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
  ]
  return {
      "prompt": tokenizer.apply_chat_template(prompt,tokenize=False, continue_final_message=True),
      "target": extract_hash_answer(sample["answer"]),
  }

def get_questions(tokenizer):
  """ To train with GRPO, each data sample needs to have fields "prompt" and "target"
      In the original dataset, there are 2 fields "question" and "answer"
  """
  ds = load_dataset("openai/gsm8k", "main")
  ds = ds.map(lambda x: format_prompt(tokenizer,x))
  # Optional: remove redundant columns
  # ds = ds.remove_columns(['question', 'answer'])
  return ds

# extract answer from the model to verify correctness
def extract_xml_answer(text):
  if "<answer>" not in text or "</answer>" not in text:
    return None
  answer=text.split("<answer>")[1].strip()    # strip white spaces (\ or \n) both front and back
  answer=answer.split("</answer>")[0].strip()
  return answer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

dataset=get_questions(tokenizer)
train_dataset, test_dataset=dataset["train"], dataset["test"]

train_dataset.save_to_disk(os.path.join(data_dir, "train"))
test_dataset.save_to_disk(os.path.join(data_dir, "test"))