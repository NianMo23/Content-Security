#!/bin/bash

# 指令微调训练示例脚本
# 请根据实际情况修改路径和参数

echo "开始指令微调训练..."

# 设置参数
MODEL_NAME="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B"  # 基础模型路径
TRAIN_DATA="../data/balanced_train_instruction.jsonl"  # 训练数据
VAL_DATA="../data/balanced_val_instruction.jsonl"      # 验证数据
OUTPUT_DIR="./instruct_lora_output"                    # 输出目录
GPU_IDS="0,1,2,3"                                      # 使用的GPU

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行训练
python train_instruct_lora.py \
    --model_name $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --output_dir $OUTPUT_DIR \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --gpu_ids $GPU_IDS \
    --fp16 \
    --seed 42

echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"