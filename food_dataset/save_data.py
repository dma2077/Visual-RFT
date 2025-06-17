#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset, DatasetDict, Features, Value, Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='基于 food101_question.json 构造 few-shot HF Dataset'
    )
    parser.add_argument(
        '--json', type=str,
        default='/llm_reco/dehua/data/questions/foodx251_question.json',
        help='原始 JSON 文件路径（含 images & conversations）'
    )
    parser.add_argument(
        '--k', type=int, default=10000,
        help='每个类别保留前 K 条样本'
    )
    parser.add_argument(
        '--workers', type=int, default=64,
        help='并行处理线程数'
    )
    parser.add_argument(
        '--output_path', type=str,
        default='/llm_reco/dehua/code/Visual-RFT/share_data/food251_all_dataset_nocot',
        help='输出 HF DatasetDict 路径，推荐使用绝对路径'
    )
    return parser.parse_args()

def load_raw(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def select_k_shot(data, k_shot):
    selected, counts = [], {}
    for d in data:
        cat = d['conversations'][1]['value']
        if counts.get(cat, 0) >= k_shot:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        selected.append(d)
    print(f"■ 原始共 {len(data)} 条，few-shot 后 {len(selected)} 条；各类分布：{counts}")
    return selected

def process_image_path(d):
    raw = d["images"][0]
    image_path = raw.replace(
        "/map-vepfs/dehua/data/data/",
        "/llm_reco/dehua/data/food_data/"
    )

    # 检查并确保 image_path 存在
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Image path 不存在 - {image_path}")

    sol_txt = d['conversations'][1]['value']

    # 强制sol_txt为字符串，避免类型混合
    if isinstance(sol_txt, list):
        sol_txt = " ".join(sol_txt)
        print(sol_txt)
    elif sol_txt is None or not isinstance(sol_txt, str):
        sol_txt = ""
        print("None")
    sol_txt = sol_txt.strip()  # 去除前后空格

    return {
        'image_path': image_path,
        'problem': "<image>\nWhat is the category of the food?",
        'solution': f"<answer>{sol_txt}</answer>",
        'category': sol_txt
    }

def build_dataset_dict(records, save_path):
    image_paths = [r['image_path'] for r in records]
    problems    = [r['problem']    for r in records]
    solutions   = [r['solution']   for r in records]
    categories  = [r['category']   for r in records]

    features = Features({
        'image':    Image(),
        'problem':  Value('string'),
        'solution': Value('string'),
        'category': Value('string')
    })

    ds = Dataset.from_dict(
        {
            'image':    image_paths,
            'problem':  problems,
            'solution': solutions,
            'category': categories
        },
        features=features
    )

    ds_dict = DatasetDict({'train': ds})

    # 安全写入文件并捕获异常
    try:
        ds_dict.save_to_disk(save_path)
        print(f"✅ DatasetDict 已保存到 {save_path}")
    except Exception as e:
        print(f"❌ 保存DatasetDict失败：{e}")

    return ds_dict

def main():
    args = parse_args()
    print("🚩 脚本开始：", time.asctime())

    raw = load_raw(args.json)
    sel = select_k_shot(raw, args.k)

    print("🚩 并行处理路径与文本…")
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        records = list(tqdm(
            exe.map(process_image_path, sel),
            total=len(sel),
            desc="处理数据",
            unit="条"
        ))

    def check_records(records):
        for idx, record in enumerate(records):
            for key, value in record.items():
                if not isinstance(value, str):
                    print(f"[类型问题] 在第 {idx} 条数据，字段 '{key}' 不是字符串，实际值: {value}，类型: {type(value)}", flush=True)
    print("🔍 检查记录字段类型...", flush=True)
    check_records(records)
    # 构造并保存 HF DatasetDict
    ds_dict = build_dataset_dict(records, args.output_path)

    print("🚩 脚本结束：", time.asctime())
    # 示例加载数据:
    # from datasets import DatasetDict
    # loaded = DatasetDict.load_from_disk(args.output_path)

if __name__ == '__main__':
    main()
