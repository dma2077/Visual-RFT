import os
import yaml
import json
import random
from tqdm import tqdm
from difflib import SequenceMatcher
import argparse
from collections import defaultdict
import string
import subprocess
import ast
import re
from openai import OpenAI
import base64
from google import genai
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import io
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API调用管理器，用于限制速率
class APIRateLimiter:
    def __init__(self, max_calls_per_minute=60):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = queue.Queue(maxsize=max_calls_per_minute)
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """在需要时等待以确保不超过速率限制"""
        with self.lock:
            # 如果队列已满，弹出最早的调用时间
            if self.calls.full():
                earliest_call = self.calls.get()
                # 计算需要等待的时间
                time_to_wait = max(0, 60 - (time.time() - earliest_call))
                if time_to_wait > 0:
                    logger.info(f"限制速率：等待 {time_to_wait:.2f} 秒")
                    time.sleep(time_to_wait)
            
            # 记录当前调用
            self.calls.put(time.time())

# 图像处理常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform

def encode_image_from_pil(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices

def load_video(video_path, bound=None, input_size=224, num_segments=8):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 存储编码后的帧
    encoded_frame_list = []
    
    # 获取要提取的帧索引
    frame_indices = get_index(
        bound, fps, total_frames, first_idx=0, num_segments=num_segments
    )

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_index}")
            continue
        
        # 转换帧（BGR到RGB）
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        img_pil = Image.fromarray(img)
        transform = build_transform(input_size=input_size)
        transformed_image = transform(img_pil)
        
        # 编码为base64字符串
        encoded_image = encode_image_from_pil(img_pil)
        encoded_frame_list.append(encoded_image)

    cap.release()
    return encoded_frame_list

class Gpt():
    def __init__(
        self,
        max_num: str = 1,
        api_key=None,
        rate_limiter=None
    ):
        self.num_segments = 128
        self.max_num = max_num
        self.api_key = api_key or "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"
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1",
            timeout=60,
        )
        self.rate_limiter = rate_limiter

    def set_frame_num(self, new_num):
        self.num_segments = new_num

    def generate_until1(self, video_path, image_path, text) -> str:
        # 使用速率限制器
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
            
        try:
            # 加载视频帧和图像
            encoded_frame_list = load_video(video_path, num_segments=self.num_segments)
            image = encode_image(image_path)
            encoded_frame_list.append(image)
            
            # 准备内容
            content = [{"type": "text", "text": text}]
            for frame_base64 in encoded_frame_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                })
                
            # 调用API
            response = self._client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=2,  # 增加token数以获取完整回答
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return None

from openai import OpenAI

class Gemini():
    def __init__(
        self,
        max_num: str = 1,
        rate_limiter=None
    ):
        self.num_segments = 128
        self.max_num = max_num
        self._client = genai.Client(api_key="AIzaSyDJ__oYRy-SlSJlpjsW5G3iy5oS6QsDgq8")

        self._client = OpenAI(
            api_key="AIzaSyDJ__oYRy-SlSJlpjsW5G3iy5oS6QsDgq8",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.rate_limiter = rate_limiter

    def set_frame_num(self, new_num):
        self.num_segments = new_num

    def generate_until(self, visual, text) -> str:
            # 使用速率限制器
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        video_path = visual
        encoded_frame_list = load_video(
            video_path, num_segments=self.num_segments
        )
        content = []
        for frame in encoded_frame_list:
            content.append(frame)
        content.append(text)
        response = self._client.models.generate_content(
            #model="gemini-2.0-flash",
            #model="gemini-2.0-pro",
            model='gemini-2.0-flash-thinking-exp-01-21',
            contents = content
                )
        return response.text
    
    # def generate_until1(self, visual1, visual2, text) -> str:
    #     video_path = visual1
    #     visual2 = encode_image(visual2)
    #     encoded_frame_list = load_video(
    #         video_path, num_segments=self.num_segments
    #     )
    #     content = []
    #     for frame in encoded_frame_list:
    #         content.append(frame)
    #     content.append(visual2)
    #     content.append(text)
    #     response = self._client.models.generate_content(
    #         model="gemini-2.0-flash",
    #         contents = content
    #             )
    #     return response.text

    def generate_until1(self, video_path, image_path, text) -> str:
        # 使用速率限制器
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
            
        try:
            # 加载视频帧和图像
            encoded_frame_list = load_video(video_path, num_segments=self.num_segments)
            image = encode_image(image_path)
            encoded_frame_list.append(image)
            
            # 准备内容
            content = [{"type": "text", "text": text}]
            for frame_base64 in encoded_frame_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                })
                
            # 调用API
            response = self._client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=1000,  # 增加token数以获取完整回答
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return None

# 多选题提示
# MULTI_CHOICE_PROMPT = "Provide only the letter corresponding to the correct answer from the given choices. Do not include any explanation or analysis."

MULTI_CHOICE_PROMPT = """
Please only provide the letter corresponding to the correct answer from the given choices. After completing your analysis, output your answer in the following format:
<analysis>: [[Detailed analysis of the image and video frames based on their content]]
<answer>: [[The final answer based on your analysis]]
"""

# 提示模板
IV_PROMPT_COT1 = """
We provide you with a video that has been divided into {frame_num} evenly spaced frames across its {duration} seconds duration, followed by an image. Please answer the question based on the content from both the video frames and the image.

**Process**:
1. **Locate the content in the video based on the image**: First, analyze the image and identify key features, objects, or events it contains. Focus specifically on the **important details** such as the **person's facial features** (e.g., facial expressions, eye movements) or **the object's category or outline** (e.g., the shape and structure, not color or decoration). Avoid focusing on less important aspects like **clothing, jewelry, or color**. Then, match these elements with the corresponding video frames to locate the relevant portion in the video.
   
2. **Analyze the video frames**: Once the relevant frames have been identified, carefully analyze their content. Pay attention to the **context, actions, or interactions** taking place within these frames that are related to the image. Again, focus on **facial expressions**, **gesture movements**, or **object types and outlines**, rather than superficial features like clothing or color.

3. **Answer the question based on the analysis**: Once you have completed the analysis of both the image and the relevant video frames, formulate your answer. Ensure your answer is rooted in the **more significant features**, such as **facial characteristics** or **key object categories and shapes**, and avoid making conclusions based solely on less relevant details like clothing, accessories, or colors.

**Important**:
- **Do not give a direct answer immediately**. You must first provide a **detailed analysis** of both the image and video frames.
- Once the analysis is complete, you may then proceed to answer the question.

**Answer format**:
<analysis>: [Detailed analysis of the image and video frames based on their content, focusing on facial features, object categories, and outlines, not clothing, jewelry, or color]

<answer>: [[The final answer based on your analysis]]
"""

IV_PROMPT_VIDEO_FIRST = """
We provide you with a video that has been divided into {frame_num} evenly spaced frames across its {duration} seconds duration, followed by an image. Please answer the question based on the content from both the video frames and the image.
"""

IV_PROMPT_IMAGE_FIRST = """
We provide you with an image placed at the very beginning, followed by a video that has been divided into {frame_num} evenly spaced frames across its {duration} seconds duration. Please answer the question based on the content from both the image and the extracted video frames.
"""

def load_config(config_path):
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def format_question(question, options):
    formatted_options = "\n".join([f"{key}. {value}" for key, value in options.items()])
    return f"{question}？\n{formatted_options}"

def convert_to_multiple_choice(data):
    def shuffle_options(correct, distractors):
        options = distractors + [correct]
        random.shuffle(options)
        return options

    def get_option_letter(index):
        return chr(ord('a') + index)

    multiple_choice_questions = []

    for item in data:
        correct_answer = item['answer']
        distractors = [d for d in item['distractors'] if d.strip()]
        all_options = shuffle_options(correct_answer, distractors)

        correct_option_index = all_options.index(correct_answer)
        correct_option_letter = get_option_letter(correct_option_index)

        options = {get_option_letter(i): opt for i, opt in enumerate(all_options)}

        text = format_question(item["question"], options)
        text += "\n" + MULTI_CHOICE_PROMPT
        
        question = {
            "question": item["question"],
            "data_id": item["data_id"],
            "image_name": item["image_name"],
            "question_type": item["question_type"],
            "granularity": item["granularity"],
            "options": options,
            "text": text,
            "correct_option": correct_option_letter,
            "answer": item['answer'],
        }

        multiple_choice_questions.append(question)

    return multiple_choice_questions

def load_questions_from_jsonl(question_file):
    """读取 question.jsonl 文件并返回格式化的问题数据。"""
    all_data = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for video_key, questions_list in obj.items():
                questions_multiple_choice_list = convert_to_multiple_choice(questions_list)
                if video_key not in all_data:
                    all_data[video_key] = []
                all_data[video_key].extend(questions_multiple_choice_list)
    return all_data

def load_existing_results(output_file):
    """加载已有结果，避免重复处理。"""
    processed_ids = set()
    total_answers = 0
    valid_answers = 0
    correct_answers = 0
    
    question_type_stats = defaultdict(lambda: {"total": 0, "valid": 0, "correct": 0, "correct_rate": 0.0})
    question_type_stats["total"] = {"total": 0, "valid": 0, "correct": 0, "correct_rate": 0.0}
    
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    # 检查是否是统计数据行
                    if "total" in result and isinstance(result["total"], dict):
                        continue
                    
                    for video_key, results in result.items():
                        for item in results:
                            # 使用 (video_key, data_id) 组合判断是否已处理
                            processed_ids.add((video_key, item["data_id"]))
                            # 累加统计数据
                            if "is_true" in item and item["is_true"]:
                                correct_answers += 1
                                question_type_stats[item["question_type"]]["correct"] += 1
                                question_type_stats["total"]["correct"] += 1
                            if item.get("model_response") != "null": 
                                valid_answers += 1
                                question_type_stats[item["question_type"]]["valid"] += 1
                                question_type_stats["total"]["valid"] += 1
                            total_answers += 1
                            
                            # 更新每种 question_type 的总数
                            question_type_stats[item["question_type"]]["total"] += 1
                            question_type_stats["total"]["total"] += 1
                except Exception as e:
                    logger.error(f"解析结果文件中的行时出错: {e}")

    # 计算每个 question_type 的正确率
    for qtype, stats in question_type_stats.items():
        if stats["valid"] > 0:
            stats["correct_rate"] = stats["correct"] / stats["valid"]

    return processed_ids, total_answers, valid_answers, correct_answers, question_type_stats

def check_answer(response, answer):
    """从模型响应中提取答案并检查是否正确"""
    # 定义所有可能的选项
    all_choices = [chr(i) for i in range(ord('a'), ord('j') + 1)]
    if response is None:
        return False, random.choice(all_choices)
    
    # 清理响应文本
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response.lower() + " "  # 添加空格并转小写

    # 使用正则表达式提取 <answer>: 后面的内容
    match = re.search(r'<answer>: (\w)', response)
    if match:
        extracted_answer = match.group(1)
        return extracted_answer == answer, extracted_answer

    # 初始化候选选项
    candidates = []

    # 按从后往前的规则查找
    for choice in reversed(all_choices):
        # 检查各种格式
        if f"({choice})" in response:
            candidates.append(choice)
            continue
        if f" {choice} " in response:
            candidates.append(choice)
            continue
        if f"{choice}." in response:
            candidates.append(choice)
            continue
        
        # 检查前后不是英文字母的情况
        choice_pos = response.find(choice)
        if choice_pos != -1:
            prev_char = response[choice_pos - 1] if choice_pos > 0 else ''
            next_char = response[choice_pos + 1] if choice_pos + 1 < len(response) else ''
            if prev_char not in string.ascii_lowercase and prev_char not in string.ascii_uppercase and \
               next_char not in string.ascii_lowercase and next_char not in string.ascii_uppercase:
                candidates.append(choice)

    # 选择最后一个匹配到的或随机选择
    if candidates:
        pred_index = candidates[-1]
    else:
        pred_index = random.choice(all_choices)

    return pred_index == answer[0], pred_index

def process_question(args, model, video_dir, image_dir, video_key, question, duration_dict, processed_ids, all_options):
    """处理单个问题并返回结果"""
    # 跳过已处理的问题
    if (video_key, question["data_id"]) in processed_ids:
        logger.info(f"跳过已处理的视频和数据ID组合: {video_key}, {question['data_id']}")
        return None
    
    # 检查视频和图像文件
    video_path = os.path.join(video_dir, f"{video_key}.mp4")
    image_path = os.path.join(image_dir, question["image_name"])
    
    if not os.path.exists(video_path):
        logger.warning(f"视频文件不存在: {video_path}")
        return None
    
    if not os.path.exists(image_path):
        logger.warning(f"图像文件不存在: {image_path}")
        return None
    
    # 准备问题数据
    data_id = question["data_id"]
    q_text = question["text"]
    correct_option = question["correct_option"]
    wrong_options = all_options - {correct_option}
    answer = question["answer"]
    question_type = question["question_type"]
    
    # 获取视频时长
    duration = duration_dict.get(video_key)
    if not duration:
        logger.warning(f"未找到视频时长信息: {video_key}")
        duration = 30  # 默认值
    
    logger.info(f"处理视频: {video_key}, 问题ID: {data_id}, 时长: {duration}秒")
    
    # 根据参数准备提示文本
    # if args.reasoning_method == "cot":
    #     text = IV_PROMPT_COT1.format(duration=duration, frame_num=int(args.nframes)) + q_text
    # else:
    #     if args.image_pos == "before":
    #         text = IV_PROMPT_IMAGE_FIRST.format(duration=duration, frame_num=int(args.nframes)) + q_text
    #     else:
    text = IV_PROMPT_VIDEO_FIRST.format(duration=duration, frame_num=int(args.nframes)) + q_text
    
    # 调用模型
    try:
        if args.has_image.lower() in ["true", "1", "yes"]:
            output = model.generate_until1(video_path, image_path, text)
            logger.info("使用图像获取输出")
        else:
            # 如果你有不使用图像的API调用，请取消下行注释
            # output = model.generate_until(video_path, text)
            logger.info("不使用图像获取输出")
            output = None  # 如果没有不使用图像的API，请设置为None
    except Exception as e:
        logger.error(f"API调用时发生错误: {e}")
        output = None
    
    # 检查答案
    model_answer = None
    is_true = False
    if output:
        is_true, model_answer = check_answer(output, correct_option)
    
    # 打印结果
    logger.info(f"视频: {video_key}, 数据ID: {data_id}")
    logger.info(f"问题: {question['question']}")
    logger.info(f"模型响应: {output}")
    logger.info(f"模型回答: {model_answer}")
    logger.info(f"正确答案: {correct_option}")
    logger.info(f"是否正确: {is_true}")
    
    # 返回结果
    return {
        "data_id": data_id,
        "image_name": question["image_name"],
        "question": question["question"],
        "question_type": question_type,
        "granularity": question["granularity"],
        "choices": question["options"],
        "model_answer": model_answer,
        "correct_option": f"{correct_option}" + ". " + question["options"][correct_option],
        "is_true": is_true,
        "model_response": output
    }

def main():
    # 环境变量设置
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NCCL_IB_TIMEOUT"] = "22"
    
    def str_to_list(value):
        return ast.literal_eval(value)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="A script to process video QA with concurrent API calls.")
    
    parser.add_argument("--config_path", type=str, default="internvl2_5.yaml", 
                        help="Path to the configuration file")
    parser.add_argument("--has_image", type=str, default=None)
    parser.add_argument("--nframes", type=str, default=None)
    parser.add_argument("--target_resolution", type=str_to_list, default=None)
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--reasoning_method", type=str, default='cot')
    parser.add_argument("--image_pos", type=str, default="after")
    parser.add_argument("--video_jsonl_path", type=str, default="")
    parser.add_argument("--duration_path", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=1, 
                        help="Maximum number of worker threads")
    parser.add_argument("--rate_limit", type=int, default=50, 
                        help="Maximum API calls per minute")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config_path)
    model_name = config["model"]["name"]
    model_path = config["model"]["path"]
    video_dir = config["paths"]["video_dir"]
    image_dir = config["paths"]["image_dir"]
    
    # 处理has_image参数
    if args.has_image:
        has_image = args.has_image.lower() in ["true", "1", "yes"]
    else:
        try:
            has_image = config["params"]["has_image"]
        except:
            has_image = True
    args.has_image = str(has_image)
    
    # 处理输出文件路径
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = config["paths"]["output_file"]
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理问题文件路径
    if args.question_file != "None" and args.question_file is not None:
        question_file = args.question_file
    else:
        question_file = config["paths"]["question_file"]
    
    # 构建输出文件名
    file_name_without_extension = os.path.splitext(os.path.basename(question_file))[0]
    output_file_name_without_extension = os.path.splitext(os.path.basename(output_file))[0]
    new_file_name = f"{file_name_without_extension}_{output_file_name_without_extension}.jsonl"
    output_file = os.path.join(output_dir, new_file_name)
    
    # 创建API速率限制器
    rate_limiter = APIRateLimiter(max_calls_per_minute=args.rate_limit)
    
    # 初始化模型
    model = Gemini(rate_limiter=rate_limiter)
    
    # 设置帧数
    if args.nframes:
        nframes = args.nframes
    else:
        nframes = config["params"].get("nframes", "64")  # 设置默认值为64
    
    model.set_frame_num(int(nframes))
    logger.info(f"使用 {nframes} 帧")
    logger.info(f"输出文件: {output_file}")
    
    # 加载问题数据
    question_data = load_questions_from_jsonl(question_file)
    
    # 加载已有结果
    processed_ids, total_answers, valid_answers, correct_answers, question_type_stats = load_existing_results(output_file)
    
    # 加载视频时长信息
    duration_dict = {}
    with open(args.duration_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                video = data["video"]
                duration = data["duration"]
                duration_dict[video] = duration
            except json.JSONDecodeError:
                logger.warning(f"无法解析视频时长数据行: {line}")
    
    # 所有选项
    all_options = set(chr(ord('a') + i) for i in range(10))  # a到j
    
    # 创建一个字典来存储结果，按视频分组
    results = {}
    
    # 为每个视频创建一个任务列表，这样可以一次性处理完一个视频的所有任务
    for video_key in question_data.keys():
        results[video_key] = []
    
    # 并发处理每个视频
    for video_key, questions in tqdm(question_data.items(), total=len(question_data), desc="Processing Videos"):
        video_path = os.path.join(video_dir, f"{video_key}.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"视频文件不存在: {video_path}")
            continue
        
        # 使用线程池处理每个视频的所有问题
        futures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 为该视频的每个问题创建一个任务
            for question in questions:
                future = executor.submit(
                    process_question, 
                    args, model, video_dir, image_dir, video_key, 
                    question, duration_dict, processed_ids, all_options
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理视频 {video_key} 的问题"):
                try:
                    result = future.result()
                    if result:
                        results[video_key].append(result)
                        
                        # 更新统计信息
                        question_type = result["question_type"]
                        total_answers += 1
                        question_type_stats["total"]["total"] += 1
                        question_type_stats[question_type]["total"] += 1
                        
                        if result["model_response"]:
                            valid_answers += 1
                            question_type_stats["total"]["valid"] += 1
                            question_type_stats[question_type]["valid"] += 1
                        
                        if result["is_true"]:
                            correct_answers += 1
                            question_type_stats["total"]["correct"] += 1
                            question_type_stats[question_type]["correct"] += 1
                except Exception as e:
                    logger.error(f"处理问题结果时出错: {e}")
        
        # 处理完一个视频的所有问题后，将该视频的所有结果写入文件
        if results[video_key]:
            logger.info(f"将视频 {video_key} 的 {len(results[video_key])} 个结果写入文件")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({video_key: results[video_key]}, ensure_ascii=False) + "\n")
            # 清空已写入的结果，释放内存
            results[video_key] = []
    
    # 计算正确率
    for q_type, stats in question_type_stats.items():
        valid = stats["valid"]
        correct = stats["correct"]
        stats["correct_rate"] = correct / valid if valid > 0 else 0.0
    
    # 写入统计信息
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(question_type_stats, ensure_ascii=False) + "\n")
    
    # 打印统计结果
    logger.info("统计结果：")
    logger.info(f"总回答数量: {total_answers}")
    logger.info(f"有效回答数量: {valid_answers}")
    logger.info(f"回答正确数量: {correct_answers}")
    logger.info(f"总正确率: {correct_answers / valid_answers if valid_answers > 0 else 0:.2%}")
    
    logger.info

if __name__ == "__main__":
    import sys
    # 设置默认参数
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    has_image = sys.argv[2] if len(sys.argv) > 2 else "1"
    nframes = sys.argv[3] if len(sys.argv) > 3 else "32"
    question_file = sys.argv[4] if len(sys.argv) > 4 else "/Users/dehua/code/image-video-bench/res/all_data_v3.jsonl"
    output_dir = sys.argv[5] if len(sys.argv) > 5 else "/Users/dehua/code/image-video-bench/outputs"
    # output_file = sys.argv[6]
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_file = "/Users/dehua/code/image-video-bench/outputs/all_data_v3_gemini_32.jsonl"
    # 构建命令行参数
    sys.argv = [
        "concurrent_api_inference.py",
        "--config_path", "gpt.yaml",
        "--has_image", "1",
        "--nframes", "32",
        "--question_file", "/Users/dehua/code/image-video-bench/res/all_data_v3.jsonl",
        "--output_file", "/Users/dehua/code/image-video-bench/outputs/all_data_v3_gemini_thinking_32.jsonl",
        "--image_pos", "after",
        "--video_jsonl_path", "./video_frames.jsonl",
        "--duration_path", "./video_duration.jsonl",
        "--max_workers", "1",
        "--rate_limit", "60"
    ]
    
    # 输出运行参数
    print("运行参数:")
    print(f"GPU ID: {gpu_id}")
    print(f"Has Image: {has_image}")
    print(f"Number of Frames: {nframes}")
    print(f"Question File: {question_file}")
    print(f"Output Directory: {output_dir}")
    
    # 调用main函数
    main()
    
    print(f"处理完成! 结果保存在: {output_file}")