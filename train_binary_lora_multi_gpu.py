import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import time
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from qwen3_classification_direct import Qwen3ForSequenceClassification
import torch.multiprocessing as mp

# =========================== 二分类优化模型 ===========================
class BinaryQwen3ForSequenceClassification(nn.Module):
    """
    二分类版本的Qwen3分类模型
    专门用于正常(0) vs 违规(1)的二分类任务
    """
    def __init__(self, model_path):
        super().__init__()
        
        # 加载原始模型，设置为2分类
        self.base_model_wrapper = Qwen3ForSequenceClassification(model_path, num_labels=2)
        
        # 复制原始模型的组件
        self.base_model = self.base_model_wrapper.base_model
        self.dropout = self.base_model_wrapper.dropout
        self.config = self.base_model_wrapper.config
        self.num_labels = 2  # 二分类
        self.hidden_size = self.base_model_wrapper.hidden_size
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 从kwargs中提取参数，避免重复传递
        position_ids = kwargs.pop('position_ids', None)
        past_key_values = kwargs.pop('past_key_values', None)
        inputs_embeds = kwargs.pop('inputs_embeds', None)
        use_cache = kwargs.pop('use_cache', None)
        output_attentions = kwargs.pop('output_attentions', None)
        output_hidden_states = kwargs.pop('output_hidden_states', None)
        return_dict = kwargs.pop('return_dict', None)
        
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.last_hidden_state
        
        # 使用序列的最后一个有效token
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            sequence_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            sequence_hidden_states = hidden_states[:, -1, :]
        
        # 应用dropout
        pooled_output = self.dropout(sequence_hidden_states)
        
        # 通过分类头获取logits (2分类)
        logits = self.base_model.lm_head(pooled_output)  # [batch_size, 2]
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        # 返回与原始模型兼容的输出格式
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if kwargs.get('output_hidden_states', False) else None,
            attentions=outputs.attentions if kwargs.get('output_attentions', False) else None
        )

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 二分类数据集类
class BinaryClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 二分类标签映射：正常=0，所有违规类=1
        self.original_label_map = {
            '正常': 0,
            '歧视': 1,
            '违法违规': 1,
            '政治安全': 1,
            '暴恐': 1,
            '色情低俗': 1
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text_cn'])
        label_text = str(row['extracted_label']).strip()
        
        # 将原始标签转换为二分类标签
        if label_text in self.original_label_map:
            binary_label = self.original_label_map[label_text]
        else:
            # 如果标签不在映射中，尝试将其作为数字处理
            try:
                original_label = int(label_text)
                # 0=正常，其他=违规
                binary_label = 0 if original_label == 0 else 1
            except ValueError:
                raise ValueError(f"未知的标签: '{label_text}'。支持的标签: {list(self.original_label_map.keys())}")
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(binary_label, dtype=torch.long)
        }

# 统计二分类数据分布
def analyze_binary_data_distribution(train_data_path):
    """
    分析二分类数据分布
    
    Args:
        train_data_path: 训练数据路径
    
    Returns:
        dict: 包含数据分布信息的字典
    """
    df = pd.read_excel(train_data_path)
    
    # 统计二分类样本数量
    normal_count = 0
    violation_count = 0
    
    # 统计原始类别分布
    original_distribution = {}
    
    for label_text in df['extracted_label']:
        label_str = str(label_text).strip()
        original_distribution[label_str] = original_distribution.get(label_str, 0) + 1
        
        if label_str == '正常':
            normal_count += 1
        else:
            violation_count += 1
    
    total_samples = normal_count + violation_count
    
    data_info = {
        'normal_count': normal_count,
        'violation_count': violation_count,
        'total_samples': total_samples,
        'normal_ratio': normal_count / total_samples,
        'violation_ratio': violation_count / total_samples,
        'class_balance_ratio': min(normal_count, violation_count) / max(normal_count, violation_count),
        'original_distribution': original_distribution
    }
    
    return data_info

# 二分类评估函数
def evaluate_binary(model, dataloader, device, rank, use_fp16=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 保存预测概率
    total_loss = 0
    
    # 获取模型数据类型
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度推理
            with torch.amp.autocast('cuda', enabled=use_fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            # 获取预测和概率
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().float().numpy())  # 转换为float32再转numpy
    
    # 计算基础指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1  # 违规类为正类
    )
    
    # 计算详细的混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 计算关键业务指标
    # 正常类误报率：正常被误判为违规的比例
    normal_false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 违规类漏报率：违规被误判为正常的比例
    violation_miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 违规类召回率（检出率）
    violation_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 正常类特异性（正确识别正常的能力）
    normal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 计算不同阈值下的指标（用于调优）
    all_probs = np.array(all_probs)
    violation_probs = all_probs[:, 1]  # 违规类的概率
    
    # 尝试不同阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics = {}
    
    for threshold in thresholds:
        threshold_preds = (violation_probs >= threshold).astype(int)
        threshold_cm = confusion_matrix(all_labels, threshold_preds, labels=[0, 1])
        
        if threshold_cm.shape == (2, 2):
            tn_t, fp_t, fn_t, tp_t = threshold_cm.ravel()
            threshold_metrics[threshold] = {
                'accuracy': (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t),
                'normal_false_positive_rate': fp_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0,
                'violation_miss_rate': fn_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0,
                'violation_recall': tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0,
                'precision': tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
            }
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        # 业务关键指标
        'normal_false_positive_rate': normal_false_positive_rate,  # 正常误报率
        'violation_miss_rate': violation_miss_rate,               # 违规漏报率
        'violation_recall': violation_recall,                     # 违规召回率
        'normal_specificity': normal_specificity,                 # 正常特异性
        # 混淆矩阵详情
        'confusion_matrix': {
            'true_negative': int(tn),   # 正常->正常
            'false_positive': int(fp),  # 正常->违规
            'false_negative': int(fn),  # 违规->正常
            'true_positive': int(tp)    # 违规->违规
        },
        'support': {
            'normal': int(tn + fp),
            'violation': int(tp + fn)
        },
        # 不同阈值下的指标
        'threshold_metrics': threshold_metrics
    }

def setup_logging(rank, output_dir):
    """设置日志记录配置"""
    if rank == 0:  # 只在主进程设置日志
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # 创建logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除现有的handlers
        logger.handlers.clear()
        
        # 创建文件handler
        log_file = os.path.join(output_dir, 'binary_training.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        # 添加handlers到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    return None

def save_hyperparameters(args, output_dir, rank, data_info=None):
    """保存超参数配置到文件"""
    if rank == 0:  # 只在主进程保存
        import json
        import time
        
        # 创建超参数字典
        hyperparams = {
            # 基础配置
            "experiment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_type": "binary_classification",
                "checkpoint": args.checkpoint,
                "train_data": args.train_data,
                "val_data": args.val_data,
                "output_dir": args.output_dir,
                "optimization": "binary_classification_standard_ce_loss"
            },
            
            # 训练超参数
            "training_hyperparameters": {
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_steps": args.warmup_steps,
                "logging_steps": args.logging_steps,
                "eval_steps": args.eval_steps,
                "save_steps": args.save_steps
            },
            
            # LoRA参数
            "lora_config": {
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules
            },
            
            # 优化器参数
            "optimizer_config": {
                "weight_decay": args.weight_decay,
                "adam_epsilon": args.adam_epsilon,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2
            },
            
            # 学习率调度
            "scheduler_config": {
                "scheduler_type": args.scheduler_type,
                "min_lr": getattr(args, 'min_lr', None),
                "poly_power": getattr(args, 'poly_power', None)
            },
            
            # 早停机制
            "early_stopping": {
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_metric": args.early_stopping_metric
            },
            
            # 系统配置
            "system_config": {
                "fp16": args.fp16,
                "seed": args.seed,
                "gpu_ids": args.gpu_ids,
                "clear_cache": args.clear_cache
            }
        }
        
        # 添加数据分析信息
        if data_info is not None:
            hyperparams["data_analysis"] = data_info
        
        # 保存为JSON文件
        config_file = os.path.join(output_dir, 'binary_hyperparameters.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(hyperparams, f, indent=2, ensure_ascii=False)
        
        # 同时保存为易读的文本文件
        txt_file = os.path.join(output_dir, 'binary_hyperparameters.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("二分类训练超参数配置\n")
            f.write("=" * 60 + "\n")
            f.write(f"训练时间: {hyperparams['experiment_info']['timestamp']}\n\n")
            
            for section_name, section_data in hyperparams.items():
                if section_name == "experiment_info":
                    continue
                f.write(f"[{section_name.upper()}]\n")
                for key, value in section_data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        return config_file, txt_file
    return None, None

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'  # 使用不同端口避免冲突
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """每个GPU上的训练工作进程"""
    # 获取实际的GPU ID
    actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
    
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 设置日志记录（只在主进程）
    logger = setup_logging(rank, args.output_dir)
    
    # 只在主进程打印信息
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"开始二分类多GPU训练，使用 {world_size} 个GPU")
        logger.info(f"主进程运行在 rank {rank}，使用 GPU {actual_gpu_id}")
        logger.info(f"损失函数: 标准交叉熵损失")
    
    # 设置当前GPU
    torch.cuda.set_device(actual_gpu_id)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f'cuda:{actual_gpu_id}')
    
    # 清理GPU缓存
    if args.clear_cache:
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"已清理GPU {actual_gpu_id} 缓存")
    
    if is_main_process:
        logger.info(f"使用设备: {device}")
        logger.info(f"世界大小: {world_size}")
        logger.info(f"当前排名: {rank}")
        
        # 记录GPU内存信息
        if torch.cuda.is_available():
            for i, gpu_id in enumerate(args.gpu_ids if args.gpu_ids else range(world_size)):
                if gpu_id < torch.cuda.device_count():
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    logger.info(f"GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")
    
    # 分析二分类数据分布
    if is_main_process:
        logger.info("分析二分类数据分布...")
        data_info = analyze_binary_data_distribution(args.train_data)
        logger.info(f"正常样本数: {data_info['normal_count']}")
        logger.info(f"违规样本数: {data_info['violation_count']}")
        logger.info(f"正常样本比例: {data_info['normal_ratio']:.4f}")
        logger.info(f"违规样本比例: {data_info['violation_ratio']:.4f}")
        logger.info(f"类别平衡比例: {data_info['class_balance_ratio']:.4f}")
        logger.info(f"原始类别分布: {data_info['original_distribution']}")
    else:
        data_info = None
    
    # 保存超参数配置
    config_file, txt_file = save_hyperparameters(args, args.output_dir, rank, data_info)
    
    if is_main_process and config_file:
        logger.info(f"超参数已保存到: {config_file} 和 {txt_file}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载二分类模型
    if is_main_process:
        logger.info("正在加载二分类模型...")
    
    model = BinaryQwen3ForSequenceClassification(args.checkpoint)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        modules_to_save=["classifier", "lm_head"]
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    if is_main_process:
        logger.info("LoRA配置:")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 将模型移到设备并包装为DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    
    # 准备二分类数据集
    if is_main_process:
        logger.info("加载二分类数据集...")
    
    train_dataset = BinaryClassificationDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = BinaryClassificationDataset(args.val_data, tokenizer, args.max_length)
    
    if is_main_process:
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    # 设置优化器和调度器
    effective_lr = args.learning_rate * world_size
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
        betas=(args.adam_beta1, args.adam_beta2)
    )
    
    if is_main_process:
        logger.info(f"原始学习率: {args.learning_rate}, 调整后学习率: {effective_lr} (缩放因子: {world_size})")
    
    # 计算总训练步数
    total_samples = len(train_dataset)
    samples_per_step = args.batch_size * world_size * args.gradient_accumulation_steps
    steps_per_epoch = total_samples // samples_per_step
    total_steps = steps_per_epoch * args.num_epochs
    
    if is_main_process:
        logger.info(f"训练配置:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  批次大小: {args.batch_size}")
        logger.info(f"  梯度累积步数: {args.gradient_accumulation_steps}")
        logger.info(f"  每步处理样本数: {samples_per_step}")
        logger.info(f"  每轮步数: {steps_per_epoch}")
        logger.info(f"  总训练步数: {total_steps}")
        logger.info(f"  训练轮数: {args.num_epochs}")
    
    # 学习率调度器
    if args.scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    elif args.scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    elif args.scheduler_type == "polynomial":
        from transformers import get_polynomial_decay_schedule_with_warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            power=args.poly_power
        )
    
    # 混合精度训练
    model_dtype = next(model.parameters()).dtype
    use_amp_scaler = args.fp16 and model_dtype != torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if use_amp_scaler else None
    
    # 创建输出目录
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    best_val_f1 = 0
    best_val_loss = float('inf')
    best_violation_recall = 0  # 最佳违规召回率
    global_step = 0
    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"===== 开始第 {epoch + 1}/{args.num_epochs} 轮二分类训练 =====")
        
        # 设置分布式采样器的epoch
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training", disable=(rank != 0))
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=args.fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.gradient_accumulation_steps
            
            if use_amp_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # 梯度累积
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_amp_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 日志记录
                if global_step % args.logging_steps == 0 and is_main_process:
                    avg_loss = train_loss / (step + 1)
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    train_pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})
                
                # 评估
                if global_step % args.eval_steps == 0:
                    val_results = evaluate_binary(model, val_loader, device, rank)
                    
                    if is_main_process:
                        logger.info(f"验证集结果 - Step {global_step}:")
                        logger.info(f"  Loss: {val_results['loss']:.4f}")
                        logger.info(f"  准确率: {val_results['accuracy']:.4f}")
                        logger.info(f"  F1: {val_results['f1']:.4f}")
                        logger.info(f"  精确率: {val_results['precision']:.4f}")
                        logger.info(f"  召回率: {val_results['recall']:.4f}")
                        
                        # 关键业务指标
                        logger.info(f"\n  【关键业务指标】")
                        logger.info(f"    正常误报率: {val_results['normal_false_positive_rate']:.4f}")
                        logger.info(f"    违规漏报率: {val_results['violation_miss_rate']:.4f}")
                        logger.info(f"    违规召回率: {val_results['violation_recall']:.4f}")
                        logger.info(f"    正常特异性: {val_results['normal_specificity']:.4f}")
                        
                        # 混淆矩阵
                        cm = val_results['confusion_matrix']
                        logger.info(f"\n  【混淆矩阵】")
                        logger.info(f"    正常->正常: {cm['true_negative']}")
                        logger.info(f"    正常->违规: {cm['false_positive']}")
                        logger.info(f"    违规->正常: {cm['false_negative']}")
                        logger.info(f"    违规->违规: {cm['true_positive']}")
                        
                        # 不同阈值下的指标
                        logger.info(f"\n  【不同阈值下的指标】")
                        for threshold, metrics in val_results['threshold_metrics'].items():
                            logger.info(f"    阈值{threshold}: 准确率={metrics['accuracy']:.4f}, "
                                      f"误报率={metrics['normal_false_positive_rate']:.4f}, "
                                      f"漏报率={metrics['violation_miss_rate']:.4f}")
                        
                        # 保存最佳模型
                        improved = False
                        if args.early_stopping_metric == "f1" and val_results['f1'] > best_val_f1:
                            best_val_f1 = val_results['f1']
                            improved = True
                        elif args.early_stopping_metric == "loss" and val_results['loss'] < best_val_loss:
                            best_val_loss = val_results['loss']
                            improved = True
                        elif args.early_stopping_metric == "recall" and val_results['violation_recall'] > best_violation_recall:
                            best_violation_recall = val_results['violation_recall']
                            improved = True
                        
                        if improved:
                            patience_counter = 0
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(os.path.join(args.output_dir, 'best_binary_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_binary_model'))
                            logger.info(f"保存最佳二分类模型，F1: {val_results['f1']:.4f}, "
                                      f"违规召回率: {val_results['violation_recall']:.4f}")
                        else:
                            patience_counter += 1
                            logger.info(f"验证指标未改善，耐心计数: {patience_counter}/{early_stopping_patience}")
                            
                        # 早停检查
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"触发早停，在步骤 {global_step} 停止训练")
                            if torch.distributed.is_initialized():
                                torch.distributed.barrier()
                            cleanup_distributed()
                            return
                    
                    model.train()
                
                # 定期保存检查点
                if global_step % args.save_steps == 0 and is_main_process:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-binary-{global_step}')
                    model_to_save.save_pretrained(checkpoint_dir)
                    logger.info(f"保存二分类检查点到: {checkpoint_dir}")
        
        # Epoch结束后的评估
        val_results = evaluate_binary(model, val_loader, device, rank)
        
        if is_main_process:
            logger.info(f"Epoch {epoch + 1} 验证集最终结果:")
            logger.info(f"  Loss: {val_results['loss']:.4f}")
            logger.info(f"  准确率: {val_results['accuracy']:.4f}")
            logger.info(f"  F1: {val_results['f1']:.4f}")
            logger.info(f"  关键指标:")
            logger.info(f"    正常误报率: {val_results['normal_false_positive_rate']:.4f}")
            logger.info(f"    违规漏报率: {val_results['violation_miss_rate']:.4f}")
            
            # 每个epoch结束后保存模型
            epoch_output_dir = os.path.join(args.output_dir, f'epoch-binary-{epoch + 1}')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            logger.info(f"已保存第 {epoch + 1} 个epoch的二分类模型到: {epoch_output_dir}")
    
    # 确保所有进程都完成训练后再清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="二分类LoRA多GPU训练脚本")
    # 基础配置
    parser.add_argument('--checkpoint', type=str,
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B")
    parser.add_argument('--train_data', type=str, default="./data/r789-b-50000_train.xlsx")
    parser.add_argument('--val_data', type=str, default="./data/r789-b-50000_val.xlsx")
    parser.add_argument('--test_data', type=str, default="./data/r789-b-50000_test.xlsx")
    parser.add_argument('--output_dir', type=str, default="./lora-binary-73-1754")
    
    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)  # 二分类可以使用稍低的学习率
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA缩放因子')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout率')
    parser.add_argument('--target_modules', type=str, nargs='+',
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help='LoRA目标模块')
    
    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                       help='Adam优化器的epsilon参数')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                       help='Adam优化器的beta1参数')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                       help='Adam优化器的beta2参数')
    
    # 学习率调度
    parser.add_argument('--scheduler_type', type=str, default="linear",
                       choices=["linear", "cosine", "polynomial"],
                       help='学习率调度器类型')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='最小学习率，用于cosine调度器')
    parser.add_argument('--poly_power', type=float, default=1.0,
                       help='多项式调度器的幂次')
    
    # 早停机制
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值')
    parser.add_argument('--early_stopping_metric', type=str, default="f1",
                       choices=["f1", "loss", "recall"],
                       help='早停监控指标')
    
    # 日志和保存
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=200)
    
    # 系统配置
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5",
                       help='指定使用的GPU ID，用逗号分隔')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='指定使用的GPU数量，从GPU 0开始使用')
    parser.add_argument('--clear_cache', action='store_true', default=True,
                       help='训练前清理GPU缓存')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行多GPU训练")
    
    # 处理GPU配置
    total_gpus = torch.cuda.device_count()
    print(f"系统检测到 {total_gpus} 个GPU")
    
    # 解析GPU配置
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        for gpu_id in gpu_ids:
            if gpu_id >= total_gpus:
                raise ValueError(f"指定的GPU ID {gpu_id} 超出可用范围 (0-{total_gpus-1})")
        args.gpu_ids = gpu_ids
        world_size = len(gpu_ids)
        print(f"使用指定的GPU: {gpu_ids}")
    elif args.num_gpus is not None:
        if args.num_gpus > total_gpus:
            raise ValueError(f"指定的GPU数量 {args.num_gpus} 超出可用GPU数量 {total_gpus}")
        args.gpu_ids = list(range(args.num_gpus))
        world_size = args.num_gpus
        print(f"使用前 {args.num_gpus} 个GPU: {args.gpu_ids}")
    else:
        # 使用所有可用GPU
        args.gpu_ids = list(range(total_gpus))
        world_size = total_gpus
        print(f"使用所有可用GPU: {args.gpu_ids}")
    
    if world_size < 2:
        print("警告: 只使用1个GPU，建议使用单GPU训练脚本")
    
    # 记录训练开始信息
    print(f"开始启动二分类多进程训练，世界大小: {world_size}")
    print(f"输出目录: {args.output_dir}")
    print(f"损失函数: 标准交叉熵损失")
    print(f"任务类型: 二分类 (正常 vs 违规)")
    print(f"日志将保存到: {os.path.join(args.output_dir, 'binary_training.log')}")
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()