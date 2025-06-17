#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datasets import Dataset, DatasetDict, Features, Value, Image as HfImage
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

dataset_name = "veg200"
def parse_args():
    parser = argparse.ArgumentParser(
        description="将 JSON 文件转换为 HuggingFace DatasetDict 并缩放图像后保存到磁盘"
    )
    parser.add_argument(
        "--json", "-j", default=f'/llm_reco/dehua/data/questions/{dataset_name}_question.json',
        help="输入 JSON 文件路径（包含 images & conversations 字段）"
    )
    parser.add_argument(
        "--output_path", "-o", default=f'/llm_reco/dehua/code/Visual-RFT/share_data/{dataset_name}',
        help="输出 DatasetDict 保存目录"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"🚩 读取 JSON：{args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 初始化字段
    images, problems, solutions, categories, supclasses = [], [], [], [], []

    print("🚩 加载并处理图像中…")
    for item in tqdm(raw):
        try:
            # 图像路径映射
            image_path = item["images"][0].replace(
                "/map-vepfs/dehua/data/data/",
                "/llm_reco/dehua/data/food_data/"
            ).replace("vegfru-dataset/", "")
            # 加载 + resize
            img = Image.open(image_path).convert("RGB").resize((224, 224), resample=Image.LANCZOS)
            images.append(img)

            cat = item["conversations"][1]["value"]
            problems.append("<image>What is the dish?")
            solutions.append(f"<answer>{cat}</answer>")
            categories.append(cat)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as e:
            print(f"⚠️ 跳过图像：{image_path}, 错误：{e}")
            continue

    print(f"✅ 成功处理图像数：{len(images)}")

    # 构造 Dataset（使用 datasets.Image 类型）
    features = Features({
        "image":    HfImage(),  # 正确声明为 Image 类型
        "problem":  Value("string"),
        "solution": Value("string"),
        "category": Value("string"),
    })

    dataset = Dataset.from_dict({
        "image": images,
        "problem": problems,
        "solution": solutions,
        "category": categories,
    }, features=features)

    dataset_dict = DatasetDict({"train": dataset})

    # 保存到磁盘
    os.makedirs(args.output_path, exist_ok=True)
    print(f"💾 保存数据集到：{args.output_path}")
    dataset_dict.save_to_disk(args.output_path)
    print("✅ 完成保存")

if __name__ == "__main__":
    main()
