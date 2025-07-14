# V2.0 Focal Loss版本

## 📋 概述

V2.0版本在V1.0基础上引入了Focal Loss损失函数，专门解决文本分类任务中的类别不平衡问题。通过动态调整不同难度样本的权重，显著提升了模型在少数类别上的表现，特别是对困难样本的识别能力。

## 🎯 主要特点

- **Focal Loss**: 引入Focal Loss解决类别不平衡问题
- **困难样本关注**: 自动识别并重点训练困难样本
- **动态权重调整**: 根据预测置信度动态调整样本权重
- **优化训练策略**: 改进的训练流程和参数设置
- **增强的多GPU支持**: 优化的分布式训练实现

## 🗂️ 文件结构

```
v2.0-facal-Loss/
├── classification_model.py              # 带Focal Loss的分类模型
├── train_lora.py                       # 单GPU训练脚本
├── train_lora_multi_gpu.py             # 多GPU训练脚本
├── train_lora_multi_gpu_optimized.py   # 优化的多GPU训练
├── test_lora.py                        # 单GPU测试脚本
├── test_optimized_lora_multi_gpu.py    # 优化的多GPU测试
├── test_saved_lora_multi_gpu.py        # 多GPU保存模型测试
├── binary_classification_test.py       # 二分类测试
├── qwen3_classification_direct.py      # 直接分类推理
├── load_model.py                       # 模型加载工具
├── get_data_by_labels.py               # 数据标签处理
└── test_cert.py                        # 认证测试
```

## 🔧 核心技术

### 1. Focal Loss算法

Focal Loss是针对类别不平衡问题设计的损失函数：

```python
FL(pt) = -α(1-pt)^γ * log(pt)
```

**关键参数：**
- **α (alpha)**: 类别权重，平衡正负样本
- **γ (gamma)**: 聚焦参数，控制困难样本的权重

**工作原理：**
- 对于易分类样本（高置信度），`(1-pt)^γ`接近0，损失权重降低
- 对于难分类样本（低置信度），`(1-pt)^γ`接近1，损失权重保持
- 自动将训练重点转移到困难样本上

### 2. 改进的训练策略

- **自适应学习率**: 根据Focal Loss的特性调整学习率
- **梯度累积优化**: 改进的梯度累积策略
- **动态批次调整**: 根据样本难度动态调整批次组成
- **早停机制**: 基于验证集Focal Loss的早停策略

### 3. 优化的模型架构

- **增强的分类头**: 适配Focal Loss的分类层设计
- **正则化改进**: 结合Focal Loss的正则化策略
- **特征提取优化**: 针对困难样本的特征提取改进

## 🚀 使用方法

### 训练

```bash
# 优化的多GPU训练（推荐）
python train_lora_multi_gpu_optimized.py \
    --checkpoint /path/to/qwen3-1.7b \
    --train_data ../data/balanced_train.xlsx \
    --val_data ../data/balanced_val.xlsx \
    --output_dir ./output \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --gpu_ids 0,1,2,3

# 标准多GPU训练
python train_lora_multi_gpu.py \
    --checkpoint /path/to/qwen3-1.7b \
    --train_data ../data/balanced_train.xlsx \
    --val_data ../data/balanced_val.xlsx \
    --output_dir ./output \
    --gpu_ids 0,1,2,3
```

### 测试

```bash
# 优化的多GPU测试
python test_optimized_lora_multi_gpu.py \
    --checkpoint /path/to/qwen3-1.7b \
    --lora_model ./output/best_model \
    --test_data ../data/test.xlsx \
    --gpu_ids 0,1,2,3
```

## 📊 Focal Loss参数调优

### 推荐参数组合

| 场景 | α (alpha) | γ (gamma) | 说明 |
|------|-----------|-----------|------|
| 轻度不平衡 | 0.25 | 1.0 | 适合类别比例差异不大的情况 |
| 中度不平衡 | 0.25 | 2.0 | **推荐设置**，适合大多数情况 |
| 重度不平衡 | 0.1 | 3.0 | 适合极度不平衡的数据集 |
| 困难样本多 | 0.5 | 2.5 | 当数据中困难样本较多时 |

### 参数调优策略

1. **先调γ后调α**: 先确定聚焦参数γ，再调整类别权重α
2. **验证集监控**: 重点关注验证集上少数类别的F1分数
3. **损失曲线分析**: 观察Focal Loss的收敛情况
4. **困难样本分析**: 分析模型对困难样本的学习效果

## 🎪 版本改进

### 相比V1.0的提升

1. **类别不平衡解决**: Focal Loss有效解决了数据不平衡问题
2. **困难样本识别**: 自动识别并重点训练困难样本
3. **性能提升**: 在少数类别上的F1分数显著提升
4. **训练稳定性**: 更稳定的训练过程和收敛性

### 技术创新点

1. **动态权重机制**: 根据预测置信度动态调整样本权重
2. **自适应训练**: 训练过程自动适应数据分布特点
3. **优化的实现**: 高效的Focal Loss计算和梯度传播
4. **多GPU优化**: 针对Focal Loss的分布式训练优化

## 🔍 适用场景

- **类别不平衡数据集**: 特别适合处理不平衡的文本分类任务
- **困难样本较多**: 当数据中存在大量边界模糊的样本时
- **少数类别重要**: 需要重点关注少数类别性能的场景
- **精细化分类**: 需要提高分类精度和召回率的任务

## 📈 性能表现

### 主要改进指标

- **少数类别F1**: 相比V1.0提升15-25%
- **整体平衡性**: 各类别性能更加均衡
- **困难样本识别**: 边界样本识别准确率提升
- **训练效率**: 更快的收敛速度

### 评估指标

- **Focal Loss值**: 监控训练过程中的Focal Loss变化
- **类别平衡F1**: 各类别F1分数的均衡性
- **困难样本准确率**: 低置信度样本的分类准确率
- **混淆矩阵分析**: 详细的分类错误分析

