import json
from datasets import Dataset, DatasetDict
from PIL import Image
import io
import argparse
import concurrent.futures
import os
from tqdm import tqdm

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='处理食物图像数据并保存为 Dataset 格式')
parser.add_argument('--k', type=int, default=100000, help='每个类别保存的样本数量')
parser.add_argument('--workers', type=int, default=os.cpu_count(), help='工作线程数量')
parser.add_argument('--output_path', type=str, default='./share_data/food101_dataset_all_shot', help='保存数据集的路径')
args = parser.parse_args()

k_shot = args.k
num_workers = args.workers
output_path = args.output_path

print(f"每个类别将保存 {k_shot} 个样本")
print(f"使用 {num_workers} 个工作线程")
print(f"输出路径: {output_path}")

# -----------------------------
# Step 1: 读取 JSON 文件中的数据
# -----------------------------
json_filename = '/map-vepfs/dehua/code/visual-memory/questions/food101/clip_train_5_fewshot4_old.json'
with open(json_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# -----------------------------
# Step 2: 定义图像预处理函数
# -----------------------------
def process_image(item):
    d, category_counts_local = item
    # 获取图片路径，假设图片路径位于 "images" 数组的第一个元素
    image_path = d["images"][0]
    
    # 从对话中获取类别名称（同时作为 solution_text）
    solution_text = d['conversations'][1]['value']
    category = solution_text  # 用于分类统计，可选存储
    
    # 构造 solution 字段，外层套 <answer> 标签
    solution = f"<answer>{solution_text}</answer>"
    
    # 构造 problem 字段，提供图片任务说明
    problem = (
        "This is an image containing a food. Please identify the categories of the food based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. "
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>category name</answer>\n"
        "Please strictly follow the format."
    )
    
    # 读取并处理图像
    try:
        with open(image_path, "rb") as f:
            original_bytes = f.read()
        img = Image.open(io.BytesIO(original_bytes))
        
        # 若图像模式不是 RGB，则转换
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 如果图像较大，则调整尺寸（保持长宽比，最大边不超过800像素）
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
    except Exception as e:
        print(f"Warning: 无法处理图片 {image_path}. 错误: {e}")
        img = None

    return {
        'image': img,
        'problem': problem,
        'solution': solution,
        'category': category  # 可选字段，用于后续统计
    }

# -----------------------------
# Step 3: 根据每个类别只选取 k 个样本的原则进行预处理
# -----------------------------
print("预处理样本数据...")
selected_items = []
category_counts = {}

for d in data:
    solution_text = d['conversations'][1]['value']
    category = solution_text
    # 检查该类别是否已经达到 k 个样本
    if category_counts.get(category, 0) >= k_shot:
        continue
    selected_items.append((d, category_counts.copy()))
    category_counts[category] = category_counts.get(category, 0) + 1

print(f"选择了 {len(selected_items)} 个样本进行处理")

# -----------------------------
# Step 4: 多线程处理图像数据
# -----------------------------
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    for result in tqdm(executor.map(process_image, selected_items), total=len(selected_items), desc="多线程处理图像", unit="img"):
        results.append(result)

# -----------------------------
# Step 5: 整理处理结果，构造 Dataset 数据
# -----------------------------
images = []
problems = []
solutions = []
categories = []  # 可选字段

for res in results:
    images.append(res['image'])
    problems.append(res['problem'])
    solutions.append(res['solution'])
    categories.append(res['category'])

# 创建数据集字典
dataset_dict = {
    'image': images,
    'problem': problems,
    'solution': solutions,
    'category': categories  # 如果不需要可去除
}

# 用 datasets 库构造 Dataset 对象，然后包装为 DatasetDict
dataset = Dataset.from_dict(dataset_dict)
dataset_dict_hf = DatasetDict({
    'train': dataset
})

print(f"\n总共收集了 {len(dataset)} 个样本")
print(f"类别分布: {category_counts}")

# -----------------------------
# Step 6: 保存数据集到磁盘
# -----------------------------
print("保存 DatasetDict 到磁盘...")
dataset_dict_hf.save_to_disk(output_path)
print(f"数据集已保存到 {output_path}")
