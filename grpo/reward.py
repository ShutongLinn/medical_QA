"""### 1.2 Reward functions"""

import re
from dataload import extract_xml_answer

def format_reward(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

    Returns:
        list[float]: Reward scores
    """
    rewards=[]
    for completion,gt_ans in zip(completions,target):
      try: # add synthetic <think> as it is part of the prompt
        completion = "<think>"+completion
        # define regular expression to check if the format is correct
        # allow nested tags like <think><other_tag>...</other_tag></think> but not nested <think> or </think>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match=re.search(regex,completion,re.DOTALL)
        if match is None or len(match.groups())!=2:
          rewards.append(0.0)
        else:
          # (Optional) print to monitor model's response
          # print('-'*100, f"\nModel Response:\n{completion}", f"\n#### Answer: {gt_ans}")
          rewards.append(1.0)
      except Exception:
        rewards.append(0.0)

    return rewards


def correctness_reward(completions,target,**kwargs) -> list[float]:
  """
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers, each is an integer, but read a string

    Returns:
        list[float]: Reward scores
  """
  rewards=[]
  for completion, gt_ans in zip(completions,target):
    model_ans=extract_xml_answer(completion)
    # Check if model_ans is None (i.e., no <answer> tag found)
    if model_ans is None:
        rewards.append(0.0)  # No answer found, so reward is 0
        continue
    try:
        # Compare the model's answer to the ground truth
        if abs(float(model_ans) - float(gt_ans)) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    except Exception: # Handle cases where model_ans or gt_ans cannot be converted to float
        rewards.append(0.0)
  return rewards