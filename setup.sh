#!/usr/bin/env bash
set -e

echo "--- 1. 激活 Conda 环境 ---"
# 确保 conda 可用，并激活你的环境
source /llm_reco/dehua/anaconda3/etc/profile.d/conda.sh
conda activate visual-rft

echo "--- 2. 升级 pip ---"
pip install --upgrade pip

echo "--- 3. 安装 PyTorch + CUDA 12.4 （pip wheel 包含所需 CUDA 库） ---"
pip install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

echo "--- 4. 安装本地项目（开发模式）---"
# 假设项目根目录里有 src/virft 并且其中定义了 [dev] extras
cd /llm_reco/dehua/code/Visual-RFT/src/virft
pip install --force-reinstall --no-cache-dir -e ".[dev]"
cd -

echo "--- 5. 安装其他依赖 ---"
pip install --no-cache-dir \
    tensorboardx \
    qwen_vl_utils \
    vllm==0.7.2 \
    flash-attn --no-build-isolation

echo "--- 6. Pin Transformers 到指定 commit ---"
pip install --no-build-isolation \
    git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

echo "--- 安装完成！请验证 CUDA 是否可用：---"
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
EOF