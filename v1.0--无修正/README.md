# V1.0 基础版本 - 无修正

## 📋 概述

V1.0是项目的基础版本，实现了基于Qwen3-1.7B模型的文本分类任务。这个版本采用了最直接的方法，使用LoRA微调技术对预训练模型进行适配，支持6分类任务（正常、歧视、违法违规、政治安全、暴恐、色情低俗）。

## 🎯 主要特点

- **基础架构**: 使用标准的Qwen3-1.7B模型作为基座
- **LoRA微调**: 采用参数高效的LoRA技术进行模型微调
- **多GPU支持**: 支持分布式训练和推理
- **6分类任务**: 支持内容安全相关的6个类别分类
- **简单直接**: 没有复杂的损失函数或特殊技术，专注于基础功能实现

## 🗂️ 文件结构

```
v1.0--无修正/
├── classification_model.py              # 分类模型定义
├── train_lora.py                       # 单GPU训练脚本
├── train_lora_multi_gpu.py             # 多GPU训练脚本
├── test_lora.py                        # 单GPU测试脚本
├── test_saved_lora.py                  # 保存模型测试脚本
├── test_saved_lora_multi_gpu.py        # 多GPU保存模型测试
├── binary_classification_test.py       # 二分类测试
├── qwen3_classification_direct.py      # 直接分类推理
├── load_model.py                       # 模型加载工具
├── get_data_by_labels.py               # 数据标签处理
└── test_cert.py                        # 认证测试
```

## 🔧 核心技术

### 1. 模型架构
- **基座模型**: Qwen3-1.7B
- **微调方法**: LoRA (Low-Rank Adaptation)
- **分类头**: 线性分类层，输出6个类别的概率

### 2. 训练策略
- **优化器**: AdamW
- **学习率**: 标准学习率调度
- **批次大小**: 根据GPU内存动态调整
- **损失函数**: 标准交叉熵损失

### 3. LoRA参数
- **秩 (r)**: 8
- **缩放因子 (alpha)**: 16
- **dropout**: 0.1
- **目标模块**: 注意力层的查询、键、值投影

## 🚀 使用方法

### 训练

```bash
# 单GPU训练
python train_lora.py \
    --checkpoint /path/to/qwen3-1.7b \
    --train_data ../data/balanced_train.xlsx \
    --val_data ../data/balanced_val.xlsx \
    --output_dir ./output

# 多GPU训练
python train_lora_multi_gpu.py \
    --checkpoint /path/to/qwen3-1.7b \
    --train_data ../data/balanced_train.xlsx \
    --val_data ../data/balanced_val.xlsx \
    --output_dir ./output \
    --gpu_ids 0,1,2,3
```

### 测试

```bash
# 测试保存的模型
python test_saved_lora_multi_gpu.py \
    --checkpoint /path/to/qwen3-1.7b \
    --lora_model ./output/best_model \
    --test_data ../data/test.xlsx \
    --gpu_ids 0,1,2,3
```

## 📊 数据格式

输入数据需要Excel格式，包含以下字段：
- `text_cn`: 待分类的中文文本
- `extracted_label`: 标签（0-5的数字或对应的中文标签）

## 🎪 版本特色

### 优势
1. **简单可靠**: 使用成熟的技术栈，稳定性好
2. **易于理解**: 代码结构清晰，便于学习和修改
3. **资源友好**: LoRA技术减少了训练所需的计算资源
4. **快速部署**: 可以快速搭建和运行

### 局限性
1. **性能有限**: 没有针对特定问题的优化
2. **泛化能力**: 可能存在过拟合问题
3. **损失函数**: 使用标准损失，未考虑类别不平衡等问题
4. **特征学习**: 特征表示能力相对基础

## 🔍 适用场景

- 文本分类任务的快速原型验证
- 基础模型性能评估
- 作为后续版本的对比基线
- 学习和理解LoRA微调技术

## 📈 性能表现

V1.0版本提供了项目的基础性能指标，为后续版本的改进提供了参考基线。虽然性能相对基础，但为理解问题和制定改进策略奠定了重要基础。

## 🔄 后续发展

V1.0版本的经验和发现为后续版本的改进指明了方向：
- V2.0引入了Focal Loss解决类别不平衡问题
- V2.5加入了对比学习提升特征表示
- V3.0专注于二分类任务优化
- V4.0解决了泛化能力问题

