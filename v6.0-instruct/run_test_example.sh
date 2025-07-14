#!/bin/bash

# 指令微调测试示例脚本
# 请根据实际情况修改路径和参数

echo "开始指令微调模型测试..."

# 设置参数
BASE_MODEL="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B"  # 基础模型路径
LORA_MODEL="./instruct_lora_output/final_model"                             # LoRA模型路径
TEST_DATA="../data/test_instruction.jsonl"                                  # 测试数据
GPU_IDS="0,1,2,3"                                                          # 使用的GPU

# 检查模型文件是否存在
if [ ! -d "$LORA_MODEL" ]; then
    echo "错误: LoRA模型目录不存在: $LORA_MODEL"
    echo "请先运行训练脚本或检查模型路径"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "错误: 测试数据文件不存在: $TEST_DATA"
    echo "请检查测试数据路径"
    exit 1
fi

# 运行测试
python test_instruct_lora.py \
    --base_model $BASE_MODEL \
    --lora_model $LORA_MODEL \
    --test_data $TEST_DATA \
    --batch_size 4 \
    --max_length 512 \
    --max_new_tokens 50 \
    --gpu_ids $GPU_IDS \
    --fp16 \
    --seed 42

echo "测试完成！"
echo "结果文件保存在: $LORA_MODEL 目录下"