import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from PIL import Image
import io
import re
import argparse
import concurrent.futures
import os

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='处理食物图像数据并保存为Parquet格式')
parser.add_argument('--k', type=int, default=4, help='每个类别保存的样本数量')
parser.add_argument('--workers', type=int, default=os.cpu_count(), help='工作线程数量')
args = parser.parse_args()

# 设置k值 - 每个类别保存的样本数量
k_shot = args.k
num_workers = args.workers
print(f"每个类别将保存 {k_shot} 个样本")
print(f"使用 {num_workers} 个工作线程")

# -----------------------------
# Step 1: 读取原始文件及其元数据
# -----------------------------
original_file = "/map-vepfs/datasets/ViRFT_CLS_flower_4_shot/data/train-00000-of-00001.parquet"
# 读取原始 parquet 文件，保留所有元数据
original_table = pq.read_table(original_file)
print("原始文件的 schema:")
print(original_table.schema)

# 提取原始文件的元数据
original_metadata = original_table.schema.metadata

# -----------------------------
# Step 2: 读取 JSON 文件中的新数据，并转换图像为 PNG 格式
# -----------------------------
json_filename = '/map-vepfs/dehua/code/visual-memory/questions/food172/clip_train_5_fewshot4_old.json'
with open(json_filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 定义多线程处理函数
def process_image(item):
    d, category_counts_local = item
    
    # 获取图片路径
    image_path = d["images"][0]
    
    # 获取类别名称直接用solution_text
    solution_text = d['conversations'][1]['value']
    category = solution_text  # 直接用作类别
    
    # 获取solution字段
    solution = f"<answer>{solution_text}</answer>"
    
    # 读取图片的二进制数据并转换为压缩格式
    with open(image_path, "rb") as f:
        original_bytes = f.read()
    
    # 使用Pillow处理图像并压缩
    img = Image.open(io.BytesIO(original_bytes))
    
    # 调整图像尺寸以减小大小 (保持宽高比)
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # 使用JPEG格式保存，可以控制质量
    output_buffer = io.BytesIO()
    if img.mode == 'RGBA':
        # JPEG不支持透明通道，转换为RGB
        img = img.convert('RGB')
    img.save(output_buffer, format="JPEG", quality=85, optimize=True)
    compressed_bytes = output_buffer.getvalue()
    
    # 输出压缩信息
    compression_ratio = len(original_bytes) / len(compressed_bytes)
    # print(f"图像压缩: {len(original_bytes)/1024:.1f}KB -> {len(compressed_bytes)/1024:.1f}KB (比率: {compression_ratio:.1f}x)")
    
    # 定义problem字段
    problem = (
        "This is an image containing a food. Please identify the categories of the food based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. "
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>category name</answer>\n"
        "Please strictly follow the format."
    )
    
    return {
        'image_bytes': compressed_bytes,
        'image_path': None,
        'problem': problem,
        'solution': solution,
        'category': category
    }

# 准备数据
print("预处理样本数据...")
processed_items = []
category_counts = {}

# 第一步：选择要处理的项目
selected_items = []
for d in data:
    solution_text = d['conversations'][1]['value']
    category = solution_text
    
    # 检查该类别是否已达到k个样本
    if category_counts.get(category, 0) >= k_shot:
        continue  # 如果已经有k个该类别的样本，则跳过
    
    selected_items.append((d, category_counts.copy()))
    category_counts[category] = category_counts.get(category, 0) + 1

print(f"选择了 {len(selected_items)} 个样本进行处理")

# 第二步：多线程处理图像
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    # 使用tqdm显示进度
    results = list(tqdm(
        executor.map(process_image, selected_items),
        total=len(selected_items),
        desc="多线程处理图像"
    ))

# 第三步：整理处理结果
image_bytes_list = []
image_path_list = []
problem_list = []
solution_list = []
final_categories = {}

for result in results:
    image_bytes_list.append(result['image_bytes'])
    image_path_list.append(result['image_path'])
    problem_list.append(result['problem'])
    solution_list.append(result['solution'])
    
    category = result['category']
    final_categories[category] = final_categories.get(category, 0) + 1

# 输出统计信息
print(f"\n总共收集了 {len(image_bytes_list)} 个样本，来自 {len(final_categories)} 个不同类别")
print(f"类别分布: {final_categories}")

# -----------------------------
# Step 3: 构造 PyArrow Table
# -----------------------------
# 提取原始schema中的字段类型
image_field_type = original_table.schema.field('image').type
bytes_field_type = image_field_type.field('bytes').type
path_field_type = image_field_type.field('path').type

# 创建数组 - 分批处理以避免ChunkedArray问题
batch_size = 500  # 每批处理的样本数
batches = []

for i in range(0, len(image_bytes_list), batch_size):
    end = min(i + batch_size, len(image_bytes_list))
    batch_bytes = image_bytes_list[i:end]
    batch_paths = image_path_list[i:end]
    batch_problems = problem_list[i:end]
    batch_solutions = solution_list[i:end]
    
    # 创建每批的数组
    image_bytes_array = pa.array(batch_bytes, type=bytes_field_type)
    image_path_array = pa.array(batch_paths, type=path_field_type)
    
    # 确保这些是普通Array而不是ChunkedArray
    if isinstance(image_bytes_array, pa.ChunkedArray):
        image_bytes_array = image_bytes_array.chunk(0)
    if isinstance(image_path_array, pa.ChunkedArray):
        image_path_array = image_path_array.chunk(0)
    
    # 创建结构体数组
    image_struct_array = pa.StructArray.from_arrays(
        [image_bytes_array, image_path_array],
        names=["bytes", "path"]
    )
    
    problem_array = pa.array(batch_problems, type=pa.string())
    solution_array = pa.array(batch_solutions, type=pa.string())
    
    # 创建批次表格
    batch_table = pa.Table.from_arrays(
        [image_struct_array, problem_array, solution_array],
        names=["image", "problem", "solution"]
    )
    
    # 应用原始文件的元数据
    batch_table = batch_table.replace_schema_metadata(original_metadata)
    batches.append(batch_table)

# 合并所有批次
if batches:
    new_table = pa.concat_tables(batches)
else:
    new_table = pa.Table.from_arrays([], names=["image", "problem", "solution"])

# -----------------------------
# Step 4: 写入新文件 - 使用与原始文件兼容的选项
# -----------------------------
output_file = f"/map-vepfs/datasets/food172/food172-{k_shot}-shot-train.parquet"
# 使用与原始文件相同的写入选项
print(f"写入文件 {output_file}...")
pq.write_table(new_table, output_file, compression='snappy', use_dictionary=True, 
               version='2.6', write_statistics=True)

print(f"新数据已按照原始文件的格式写入 {output_file} 文件。")

# -----------------------------
# Step 5: 验证新文件 - 使用安全的读取选项
# -----------------------------
try:
    # 使用多种读取选项
    print("\n尝试不同的读取选项:")
    
    # 选项1: 普通读取
    print("选项1: 普通读取")
    new_table1 = pq.read_table(output_file)
    print("读取成功")
    
    # 选项2: 使用legacy_dataset
    print("选项2: 使用legacy_dataset")
    new_table2 = pq.read_table(output_file, use_legacy_dataset=True)
    print("读取成功")
    
    # 选项3: 禁用内存映射
    print("选项3: 禁用内存映射")
    new_table3 = pq.read_table(output_file, memory_map=False)
    print("读取成功")
    
    # 输出新文件的行数
    num_rows = len(new_table1)
    expected_rows = sum(min(count, k_shot) for category, count in final_categories.items())
    print(f"\n新文件的行数: {num_rows}")
    print(f"预期行数 (每个类别最多 {k_shot} 个样本): {expected_rows}")
    print(f"类别数量: {len(final_categories)}")
    print(f"理论最大行数 (如果每个类别都有 {k_shot} 个样本): {len(final_categories) * k_shot}")
    
    if num_rows == expected_rows:
        print("✓ 行数符合预期")
    else:
        print("✗ 行数与预期不符")
    
    print("\n新文件的 schema:")
    print(new_table1.schema)
    print("\n原始文件的 schema:")
    print(original_table.schema)
    
    # 判断两个 schema 是否一致
    if original_table.schema.equals(new_table1.schema):
        print("\n格式一致：两个文件的 schema 完全相同。")
    else:
        print("\n格式不一致：两个文件的 schema 不同。")
    
except Exception as e:
    print(f"读取新文件时出错: {e}")
    print("尝试使用更多兼容选项读取...")
    
    try:
        # 使用更多兼容性选项读取
        new_table = pq.ParquetFile(output_file).read()
        print("使用 ParquetFile 读取成功")
    except Exception as e2:
        print(f"尝试使用 ParquetFile 读取也失败: {e2}")
