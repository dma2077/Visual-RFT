import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ANSI颜色定义
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

def plot_images(image_paths):
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# # 全局变量，后续由命令行参数赋值
# MODEL_PATH = None
# TEST_FILE = None

import json
import math
import re
import torch
import logging
from tqdm import tqdm
# 假设已导入模型和处理器相关的模块，例如：
# from transformers import AutoProcessor
# from your_model_library import Qwen2_5_VLForConditionalGeneration, process_vision_info

import json
import math
import re
import torch
import logging
from tqdm import tqdm
# 假设以下模块已导入：
# from transformers import AutoProcessor
# from your_model_library import Qwen2_5_VLForConditionalGeneration, process_vision_info

def run(rank, world_size, MODEL_PATH, TEST_FILE):
    logger = logging.getLogger(__name__)
    
    # ------------------ 模型与处理器加载 ------------------
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 将模型加载到对应设备上（这里假设 rank 对应 CUDA 设备）
    model = model.to(torch.device(rank))
    model = model.eval()
    
    # ------------------ 类别文件加载 ------------------
    food251_class_file = '/llm_reco/dehua/data/food_data/FoodX-251/annot/class_list.txt'
    food101_class_file = '/llm_reco/dehua/data/food_data/food-101/meta/classes.txt'
    food172_class_file = '/llm_reco/dehua/data/food_data/VireoFood172/SplitAndIngreLabel/FoodList.txt'
    
    if 'foodx251' in TEST_FILE:
        with open(food251_class_file, 'r') as file:
            lines = file.readlines()
        categories = []
        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                category = parts[1]
            else:
                category = parts[0]
            category = category.replace('_', ' ')
            categories.append(category)
        print("类别数量:", len(categories))
    elif 'food172' in TEST_FILE:
        with open(food172_class_file, 'r') as file:
            lines = file.readlines()
        categories = []
        for line in lines:
            category = line.strip().replace('_', ' ')
            categories.append(category)
        print("类别数量:", len(categories))
    else:
        with open(food101_class_file, 'r') as file:
            lines = file.readlines()
        categories = [line.strip().replace('_', ' ') for line in lines]
        print("类别数量:", len(categories))
    # ------------------ 测试数据加载 ------------------
    with open(TEST_FILE, 'r', encoding='utf-8') as file:
        val_set = []
        for line in file.readlines():
            data = json.loads(line)
            # 存储格式为 {image_path: label_index}
            val_set.append({data["image"]: int(categories.index(data["groundtruth"]))})
    
    # 根据 world_size 划分当前进程数据
    split_length = math.ceil(len(val_set) / world_size)
    logger.info("Split Chunk Length: " + str(split_length))
    split_images = val_set[int(rank * split_length): int((rank + 1) * split_length)]
    logger.info("当前进程处理数量: " + str(len(split_images)))
    
    
    # ------------------ 设置 batch size 并进行批量推理 ------------------
    total_samples = len(split_images)
    error_count = 0
    right_count = 0

    # 用 tqdm 显示样本处理进度（total_samples 为全部样本数）
    for image in tqdm(split_images, total=total_samples): 
        ### 获取图片信息
        for k,v in image.items():
            image_path = k
            image_label = v
        image_cate = categories[image_label]   
        # plot_images([image_path])
    
        question = (
        "This is an image containing an food. Please identify the category of the food based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>category name</answer>\n"
        "Please strictly follow the format."
        )
    
        image_path = image_path.replace("/map-vepfs/dehua/data/data/", "/llm_reco/dehua/data/food_data/")
        query = "<image>\n"+question
        # print(RED+query+RESET)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        # print("\033[92m" + response + "\033[0m")
    
        try:
            match = re.search(r"<answer>(.*?)</answer>", response)
            answer_content = match.group(1)
            image_cate_proc = image_cate.replace(' ', '').replace('_', '').lower()
            answer_content_proc = answer_content.replace(' ', '').replace('_', '').lower()
            if image_cate_proc in answer_content_proc or answer_content_proc in image_cate_proc:
                print('yes')
                right_count += 1
                logger.info('Local Right Number: ' + str(right_count))
            else:
                print('no')
        except Exception as e:
            error_count += 1
            
    return [error_count, right_count]




def main():
    parser = ArgumentParser(description="多进程图像推理")
    parser.add_argument("--model_path", type=str, default="/llm_reco/dehua/model/transfer_model/Qwen2.5-VL-7B-Instruct_GRPO_foodx251_all_shot/Qwen2.5-VL-7B-Instruct_GRPO_foodx251_all_shot/checkpoint-2900",
                        help="模型路径")
    parser.add_argument("--test_file", type=str, default="/llm_reco/dehua/data/transfer_data/questions/foodx251/attribute.jsonl",
                        help="测试数据文件路径")
    args = parser.parse_args()

    # global MODEL_PATH, TEST_FILE
    MODEL_PATH = args.model_path
    TEST_FILE = args.test_file
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn', force=True)
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, MODEL_PATH=MODEL_PATH, TEST_FILE=TEST_FILE)
            result_lists = pool.map(func, range(world_size))

        global_count_error = sum(int(r[0]) for r in result_lists)
        global_count_right = sum(r[1] for r in result_lists)
            
        logger.info('Error number: ' + str(global_count_error))  
        logger.info('Total Right Number: ' + str(global_count_right))
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
