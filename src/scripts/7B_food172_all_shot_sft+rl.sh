export DEBUG_MODE="true"
export LOG_PATH="/llm_reco/dehua/data/debug_log_7b_GRPO_food172_all_shot_nocot.txt"

# export DATA_PATH=/map-vepfs/datasets/food101/food172-8-shot-train.parquet
export DATA_PATH=/llm_reco/dehua/code/Visual-RFT/share_data/food172_all_dataset_nocot
# export CKPT_PATH=/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct
# export CKPT_PATH=/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-7B-Instruct
export CKPT_PATH=/llm_reco/dehua/model/food_model/Qwen2.5-VL-food172_raw
export SAVE_PATH=/llm_reco/dehua/code/Visual-RFT/outputs/Qwen2.5-VL-7B-Instruct_GRPO_food172_all_shot_nocot

export http_proxy="http://10.156.157.159:11080"
export https_proxy="http://10.156.157.159:11080"
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

cd /llm_reco/dehua/code/Visual-RFT/src/virft
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12311" \
    src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH}  \
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
    --run_name Qwen2.5-VL-7B-_GRPO_food172_all_shot_nocot \
    --save_strategy steps \
    --save_step 1000 \
    --save_only_model true \
    --num_generations 8
