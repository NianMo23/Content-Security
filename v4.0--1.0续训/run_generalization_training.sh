#!/bin/bash

# 泛化能力增强续训脚本
# 专门解决违法违规和正常类别泛化差的问题

# ===================  配置参数  ===================

# 模型路径配置
BASE_CHECKPOINT="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B"
LORA_MODEL_PATH="../lora-771746/checkpoint-200"  # 修改为您现有的LoRA模型路径

# 数据路径配置
TRAIN_DATA="../data/mixed_train_dataset.xlsx"
VAL_DATA="../data/balanced_val.xlsx"

# 输出路径
OUTPUT_DIR="../generalization-lora-$(date +%Y%m%d_%H%M%S)"

# GPU配置
GPU_IDS="0,1,2,3,4,5"  # 根据您的GPU情况修改

# 泛化增强参数
AUGMENTATION_RATIO=3.0      # 数据增强倍率（增加更多样化的样本）
FOCUS_VIOLATION_NORMAL=true # 专注于违法违规和正常类别

# 泛化训练参数（更保守的设置）
NUM_EPOCHS=8               # 更多轮数让模型充分学习
BATCH_SIZE=12              # 较小batch size增强泛化能力
LEARNING_RATE=8e-6         # 更小学习率防止过拟合
MAX_LENGTH=256
GRADIENT_ACCUMULATION_STEPS=6
WEIGHT_DECAY=0.02          # 增加正则化
WARMUP_STEPS=100           # 更多预热步数

# ===================  运行泛化增强训练  ===================

echo "开始泛化能力增强续训..."
echo "专门解决违法违规和正常类别泛化差的问题"
echo ""
echo "配置信息:"
echo "  基础模型: $BASE_CHECKPOINT"
echo "  现有LoRA: $LORA_MODEL_PATH"
echo "  数据增强倍率: $AUGMENTATION_RATIO"
echo "  学习率: $LEARNING_RATE"
echo "  权重衰减: $WEIGHT_DECAY"
echo "  输出目录: $OUTPUT_DIR"
echo ""

python generalization_enhanced_training.py \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --lora_model_path "$LORA_MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --augmentation_ratio $AUGMENTATION_RATIO \
    --focus_violation_normal \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --gpu_ids "$GPU_IDS" \
    --fp16 \
    --clear_cache \
    --early_stopping_patience 4 \
    --logging_steps 20 \
    --eval_steps 80

echo ""
echo "泛化能力增强训练完成！"
echo "最佳泛化模型保存在：$OUTPUT_DIR/best_generalization_model"
echo ""
echo "关键改进："
echo "1. 数据增强：提供了 ${AUGMENTATION_RATIO}x 的增强样本"
echo "2. 一致性正则化：确保预测的一致性"
echo "3. 多样性损失：增强特征表示的多样性"
echo "4. 置信度校准：防止过度自信"
echo "5. 平滑性正则化：相似输入产生相似输出"