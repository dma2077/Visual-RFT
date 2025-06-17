export DEBUG_MODE="true"
export LOG_PATH="./debug_log_7b_GRPO_food101_all_shot.txt"

# export DATA_PATH=/map-vepfs/datasets/food101/food172-8-shot-train.parquet
export DATA_PATH=/llm_reco/dehua/code/Visual-RFT/share_data/food101_all_dataset
# export CKPT_PATH=/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct
export CKPT_PATH=/llm_reco/dehua/model/food_model/Qwen2.5-VL-food101_raw
export SAVE_PATH=/llm_reco/dehua/code/Visual-RFT/outputs/Qwen2.5-VL-7B-Instruct_GRPO_food101_all_shot


wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73

cd /llm_reco/dehua/code/Visual-RFT/src/virft
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12383" \
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
    --run_name Qwen2.5-VL-7B-_GRPO_food101_all_shot \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 8
