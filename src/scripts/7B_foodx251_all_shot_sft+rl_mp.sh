# 如果你希望保留环境变量，可以先在 shell 里 export：
export DEBUG_MODE="true"
export LOG_PATH="/llm_reco/dehua/data/debug_log_7b_GRPO_foodx251_all_shot.txt"
export DATA_PATH=/llm_reco/dehua/code/Visual-RFT/share_data/foodx251_all_dataset_nocot
export CKPT_PATH=/llm_reco/dehua/model/food_model/Qwen2.5-VL-foodx251_raw
export SAVE_PATH=/llm_reco/dehua/code/Visual-RFT/outputs/Qwen2.5-VL-7B-Instruct_GRPO_foodx251_all_shot_nocot
export http_proxy="http://10.156.157.159:11080"
export https_proxy="http://10.156.157.159:11080"

deepspeed \
  --hostfile /llm_reco/dehua/hostfile \
  --num_nodes 3 \
  --num_gpus 8 \
  --master_addr 10.82.122.33 \
  --master_port 12331 \
  src/virft/src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed /llm_reco/dehua/code/Visual-RFT/src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing False \
    --max_pixels 100352 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-_GRPO_foodx251_all_shot_nocot \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 8
