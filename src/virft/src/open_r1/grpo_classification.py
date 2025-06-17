# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pdb
import sys
print(f"--- SCRIPT EXECUTED BY: {sys.executable} ---")

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass  # Continue to next verification method if this fails

#         # If symbolic verification failed, try string matching
#         if reward == 0.0:
#             try:
#                 # Extract as before…
#                 sol_match = re.search(r'<answer>(.*?)</answer>', sol, flags=re.IGNORECASE)
#                 ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
#                 content_match = re.search(r'<answer>(.*?)</answer>', content, flags=re.IGNORECASE)
#                 student_answer = content_match.group(1).strip() if content_match else content.strip()

#                 # Normalize to lower‐case (or use .casefold() for full Unicode)
#                 gt_norm = ground_truth.lower()
#                 sa_norm = student_answer.lower()

#                 if gt_norm == sa_norm:
#                     reward = 1.0
#                 else:
#                     from difflib import SequenceMatcher
#                     def text_similarity(a, b):
#                         return SequenceMatcher(None, a, b).ratio()
    
#                     reward = 0.6 * text_similarity(gt_norm, sa_norm)
#             except Exception:
#                 pass

#         rewards.append(reward)
#         # import pdb; pdb.set_trace()
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             # local_rank = int(os.getenv("LOCAL_RANK", 0))
#             with open(log_path, "a", encoding='utf-8') as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"content: {content}\n")
#                 f.write(f"sol: {sol}\n")
#     return rewards

# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass  # Continue to next verification method if this fails

#         # If symbolic verification failed, try string matching
#         if reward == 0.0:
#             try:
#                 # Extract as before…
#                 ground_truth = sol.replace("<answer>", "").replace("</answer>", "")
#                 student_answer = content
#                 gt_norm = ground_truth.lower()
#                 sa_norm = student_answer.lower()
#                 if gt_norm == sa_norm:
#                     reward = 1.0
#                 else:
#                     from difflib import SequenceMatcher
#                     def text_similarity(a, b):
#                         return SequenceMatcher(None, a, b).ratio()
#                     reward = 0.6 * text_similarity(gt_norm, sa_norm)
#             except Exception:
#                 pass
#         rewards.append(reward)
#         # import pdb; pdb.set_trace()
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             # local_rank = int(os.getenv("LOCAL_RANK", 0))
#             with open(log_path, "a", encoding='utf-8') as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"content: {content}\n")
#                 f.write(f"sol: {ground_truth}\n")
#     return rewards


def accuracy_reward(completions, solution, supclass, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, sc in zip(contents, solution, supclass):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract as before…
                ground_truth = sol.replace("<answer>", "").replace("</answer>", "")
                student_answer = content
                gt_norm = ground_truth.lower()
                sa_norm = student_answer.lower()
                m = re.match(r'\s*(.*?)\s*\|\s*(.+)', sa_norm)
                if m:
                    super_class_prediction = m.group(1)
                    class_prediction = m.group(2)
                else:
                    reward = 0.0
                if super_class_prediction.lower() == sc.lower() and class_prediction == gt_norm:
                    reward = 1.0
                elif super_class_prediction.lower() == sc.lower():
                    reward = 0.5
                else:
                    reward = 0.0
            except Exception:
                pass
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {ground_truth}\n")
    return rewards

# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         # Try symbolic verification first
#         try:
#             answer = parse(content)
#             if float(verify(answer, parse(sol))) > 0:
#                 reward = 1.0
#         except Exception:
#             pass

#         if reward == 0.0:
#             try:
#                 ground_truth = sol.replace("<answer>", "").replace("</answer>", "")
#                 student_answer = content
#                 gt_norm = ground_truth.lower()
#                 sa_norm = student_answer.lower()
#                 if gt_norm == sa_norm:
#                     reward = 1.0
#                 else:
#                     reward = 0.0
#             except Exception:
#                 pass
#         rewards.append(reward)
#         # import pdb; pdb.set_trace()
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             # local_rank = int(os.getenv("LOCAL_RANK", 0))
#             with open(log_path, "a", encoding='utf-8') as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"content: {content}\n")
#                 f.write(f"sol: {ground_truth}\n")
#     return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    # "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    # import pdb; pdb.set_trace()
    # script_args.reward_funcs = ['accuracy','format']
    script_args.reward_funcs = ['accuracy']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # import pdb; pdb.set_trace()


    ## lzy modified
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    
    # from datasets import load_from_disk
    # dataset = load_from_disk(script_args.dataset_name)
    # from datasets import load_dataset
    # # dataset = load_dataset("parquet", data_files="/map-vepfs/datasets/ViRFT_CLS_flower_4_shot/data/train-00000-of-00001.parquet")
    # dataset = load_dataset("parquet", data_files=script_args.dataset_name)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                # {
                #     "role":    "system",
                #     "content": [
                #         {"type": "text", "text": SYSTEM_PROMPT}
                #     ],
                # },
                {
                    "role":    "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ]
        }
    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    import os

    # 1) 定义一个 debug wrapper，带上 idx，并检查你怀疑出问题的字段
    def debug_make_conversation_image(example, idx):
        out = make_conversation_image(example)
        # 假设问题出在 out["some_field"] 上，请替换成你真正怀疑的列名
        val = out.get("some_field")
        # 如果既不是 list，也不是 None，就打印出来看看
        if val is not None and not isinstance(val, list):
            print(f"[BUG] idx={idx}, some_field type={type(val)}, value={val!r}")
        return out

    # 2) 在 map 时用 with_indices=True 拿到 idx，用 num_proc 保持并行
    split = script_args.dataset_train_split
    if "image" in dataset[split].features:
        print("has image in dataset")
        dataset = dataset.map(
            make_conversation_image,
        )

    # if "image" in dataset[script_args.dataset_train_split].features:
    #     print("has image in dataset")
    #     dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    #     # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    # pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
