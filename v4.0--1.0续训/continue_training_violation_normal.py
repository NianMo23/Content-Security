import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch.multiprocessing as mp
import json
from datetime import datetime
from collections import Counter

# 导入我们的专门损失函数
from violation_normal_loss import (
    ViolationNormalContrastiveLoss, 
    ViolationNormalFocalLoss,
    ViolationNormalBoundaryLoss,
    ViolationNormalTripletLoss,
    CombinedViolationNormalLoss,
    ConfidenceCalibrationLoss
)

# 导入原始模型
from qwen3_classification_direct import Qwen3ForSequenceClassification

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ViolationNormalDataset(Dataset):
    """
    专门针对违法违规和正常类别混淆问题的数据集
    可以对这两类样本进行重采样和增强
    """
    def __init__(self, data_path, tokenizer, max_length=512, 
                 balance_violation_normal=True, violation_oversample_rate=2.0):
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 定义标签映射
        self.label_map = {
            '正常': 0,
            '歧视': 1,
            '违法违规': 2,
            '政治安全': 3,
            '暴恐': 4,
            '色情低俗': 5
        }
        
        # 如果需要平衡违法违规和正常类别
        if balance_violation_normal:
            self._balance_violation_normal_samples(violation_oversample_rate)
        
        print(f"数据集大小: {len(self.data)}")
        self._print_class_distribution()
    
    def _balance_violation_normal_samples(self, oversample_rate):
        """平衡违法违规和正常类别的样本数量"""
        violation_samples = self.data[self.data['extracted_label'] == '违法违规'].copy()
        normal_samples = self.data[self.data['extracted_label'] == '正常'].copy()
        other_samples = self.data[~self.data['extracted_label'].isin(['违法违规', '正常'])].copy()
        
        print(f"原始数据分布 - 违法违规: {len(violation_samples)}, 正常: {len(normal_samples)}, 其他: {len(other_samples)}")
        
        # 如果违法违规样本较少，进行过采样
        if len(violation_samples) > 0 and len(normal_samples) > 0:
            target_violation_count = int(len(violation_samples) * oversample_rate)
            
            if target_violation_count > len(violation_samples):
                # 需要增加违法违规样本
                additional_needed = target_violation_count - len(violation_samples)
                additional_samples = violation_samples.sample(n=additional_needed, replace=True, random_state=42)
                violation_samples = pd.concat([violation_samples, additional_samples], ignore_index=True)
                print(f"违法违规样本过采样后: {len(violation_samples)}")
        
        # 重新组合数据
        self.data = pd.concat([violation_samples, normal_samples, other_samples], ignore_index=True)
        # 打乱数据
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def _print_class_distribution(self):
        """打印类别分布"""
        label_counts = self.data['extracted_label'].value_counts()
        print("类别分布:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text_cn'])
        label_text = str(row['extracted_label']).strip()
        
        # 将文本标签转换为数字
        if label_text in self.label_map:
            label = self.label_map[label_text]
        else:
            try:
                label = int(label_text)
            except ValueError:
                raise ValueError(f"未知的标签: '{label_text}'。支持的标签: {list(self.label_map.keys())}")
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def extract_features_from_model(model, input_ids, attention_mask):
    """
    从模型中提取特征向量（用于对比学习）
    """
    # 获取基础模型输出
    outputs = model.module.base_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    ) if hasattr(model, 'module') else model.base_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    # 获取最后一层隐藏状态
    hidden_states = outputs.last_hidden_state
    
    # 使用与原模型相同的池化策略
    if attention_mask is not None:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        pooled_output = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]
    else:
        pooled_output = hidden_states[:, -1, :]
    
    return pooled_output

def enhanced_evaluate(model, dataloader, device, rank, custom_criterion=None, use_fp16=True):
    """
    增强的评估函数，专门关注违法违规和正常类别的表现
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    loss_details = {}
    
    # 获取模型数据类型
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度推理
            with torch.amp.autocast('cuda', enabled=use_fp16, 
                                   dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if custom_criterion is not None:
                    # 提取特征用于自定义损失
                    features = extract_features_from_model(model, input_ids, attention_mask)
                    loss, batch_loss_details = custom_criterion(outputs.logits, features, labels)
                    
                    # 累积损失详情
                    for key, value in batch_loss_details.items():
                        if key not in loss_details:
                            loss_details[key] = []
                        loss_details[key].append(value)
                else:
                    loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失详情
    for key in loss_details:
        loss_details[key] = np.mean(loss_details[key])
    
    # 专门分析违法违规和正常类别
    violation_normal_metrics = analyze_violation_normal_confusion(all_labels, all_preds)
    
    # 计算总体指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    avg_loss = total_loss / len(dataloader)
    
    result = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
        'per_class_support': support_per_class,
        'violation_normal_metrics': violation_normal_metrics,
        'loss_details': loss_details
    }
    
    return result

def analyze_violation_normal_confusion(labels, predictions):
    """
    专门分析违法违规(2)和正常(0)类别之间的混淆情况
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # 计算违法违规类别的指标
    violation_true = (labels == 2)
    violation_pred = (predictions == 2)
    
    violation_tp = np.sum(violation_true & violation_pred)
    violation_fp = np.sum(~violation_true & violation_pred)
    violation_fn = np.sum(violation_true & ~violation_pred)
    violation_tn = np.sum(~violation_true & ~violation_pred)
    
    # 计算正常类别的指标
    normal_true = (labels == 0)
    normal_pred = (predictions == 0)
    
    normal_tp = np.sum(normal_true & normal_pred)
    normal_fp = np.sum(~normal_true & normal_pred)
    normal_fn = np.sum(normal_true & ~normal_pred)
    normal_tn = np.sum(~normal_true & ~normal_pred)
    
    # 重点关注的混淆情况
    violation_to_normal = np.sum(violation_true & normal_pred)  # 违法违规被误判为正常（漏报）
    normal_to_violation = np.sum(normal_true & violation_pred)  # 正常被误判为违法违规（误报）
    
    # 计算各项指标
    violation_precision = violation_tp / (violation_tp + violation_fp) if (violation_tp + violation_fp) > 0 else 0
    violation_recall = violation_tp / (violation_tp + violation_fn) if (violation_tp + violation_fn) > 0 else 0
    violation_f1 = 2 * violation_precision * violation_recall / (violation_precision + violation_recall) if (violation_precision + violation_recall) > 0 else 0
    
    normal_precision = normal_tp / (normal_tp + normal_fp) if (normal_tp + normal_fp) > 0 else 0
    normal_recall = normal_tp / (normal_tp + normal_fn) if (normal_tp + normal_fn) > 0 else 0
    normal_f1 = 2 * normal_precision * normal_recall / (normal_precision + normal_recall) if (normal_precision + normal_recall) > 0 else 0
    
    # 混淆率
    violation_miss_rate = violation_to_normal / np.sum(violation_true) if np.sum(violation_true) > 0 else 0
    normal_false_positive_rate = normal_to_violation / np.sum(normal_true) if np.sum(normal_true) > 0 else 0
    
    return {
        'violation_precision': violation_precision,
        'violation_recall': violation_recall,
        'violation_f1': violation_f1,
        'normal_precision': normal_precision,
        'normal_recall': normal_recall,
        'normal_f1': normal_f1,
        'violation_to_normal_count': int(violation_to_normal),
        'normal_to_violation_count': int(normal_to_violation),
        'violation_miss_rate': violation_miss_rate,
        'normal_false_positive_rate': normal_false_positive_rate,
        'violation_total_true': int(np.sum(violation_true)),
        'normal_total_true': int(np.sum(normal_true)),
        # 改进指标：两类之间的分离度
        'separation_score': (violation_precision + violation_recall + normal_precision + normal_recall) / 4,
        'confusion_penalty': violation_miss_rate + normal_false_positive_rate
    }

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # 使用不同端口
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def setup_logging(rank, output_dir):
    """设置日志记录配置"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # 文件日志
        log_file = os.path.join(output_dir, 'continue_training.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    return None

def continue_train_worker(rank, world_size, args):
    """续训工作进程"""
    actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
    
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 设置日志
    logger = setup_logging(rank, args.output_dir)
    
    if rank == 0:
        logger.info("="*60)
        logger.info("开始违法违规-正常类别混淆问题的续训")
        logger.info("="*60)
        logger.info(f"基础模型路径: {args.base_checkpoint}")
        logger.info(f"LoRA模型路径: {args.lora_model_path}")
        logger.info(f"损失函数类型: {args.loss_type}")
        logger.info(f"使用GPU: {args.gpu_ids}")
    
    # 设置设备
    torch.cuda.set_device(actual_gpu_id)
    set_seed(args.seed)
    device = torch.device(f'cuda:{actual_gpu_id}')
    
    if args.clear_cache:
        torch.cuda.empty_cache()
    
    is_main_process = rank == 0
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    if is_main_process:
        logger.info("创建针对违法违规-正常混淆的数据集...")
    
    train_dataset = ViolationNormalDataset(
        args.train_data, tokenizer, args.max_length,
        balance_violation_normal=args.balance_data,
        violation_oversample_rate=args.violation_oversample_rate
    )
    val_dataset = ViolationNormalDataset(
        args.val_data, tokenizer, args.max_length,
        balance_violation_normal=False  # 验证集不平衡
    )
    
    if is_main_process:
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
    
    # 加载预训练的LoRA模型
    if is_main_process:
        logger.info("加载现有LoRA模型进行续训...")
    
    # 先加载基础模型
    base_model = Qwen3ForSequenceClassification(args.base_checkpoint, num_labels=6)
    
    # 加载已训练的LoRA权重
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    
    # 移动到设备并包装DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    
    if is_main_process:
        logger.info("LoRA模型加载完成，开始续训")
    
    # 创建数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True, drop_last=False
    )
    
    # 设置优化器（续训使用更小的学习率）
    continue_lr = args.learning_rate * world_size
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=continue_lr,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
        betas=(args.adam_beta1, args.adam_beta2)
    )
    
    # 设置学习率调度器
    total_samples = len(train_dataset)
    samples_per_step = args.batch_size * world_size * args.gradient_accumulation_steps
    steps_per_epoch = total_samples // samples_per_step
    total_steps = steps_per_epoch * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 设置自定义损失函数
    if args.loss_type == 'contrastive':
        criterion = ViolationNormalContrastiveLoss(lambda_contrast=0.5)
    elif args.loss_type == 'focal':
        criterion = ViolationNormalFocalLoss(alpha_violation=3.0, gamma_violation=3.0)
    elif args.loss_type == 'boundary':
        criterion = ViolationNormalBoundaryLoss(margin=2.0, lambda_boundary=0.3)
    elif args.loss_type == 'triplet':
        criterion = ViolationNormalTripletLoss(margin=1.0, lambda_triplet=0.2)
    elif args.loss_type == 'combined':
        criterion = CombinedViolationNormalLoss(
            use_focal=True,
            use_contrastive=True,
            use_boundary=True,
            use_triplet=True,
            contrastive_weight=0.4,
            boundary_weight=0.2,
            triplet_weight=0.1
        )
    elif args.loss_type == 'calibration':
        criterion = ConfidenceCalibrationLoss(lambda_calibration=0.1, target_confidence=0.9)
    else:
        criterion = None  # 使用原始损失
    
    if is_main_process and criterion is not None:
        logger.info(f"使用自定义损失函数: {args.loss_type}")
    
    # 混合精度训练设置
    model_dtype = next(model.parameters()).dtype
    use_amp_scaler = args.fp16 and model_dtype != torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if use_amp_scaler else None
    
    # 续训循环
    best_separation_score = 0  # 关注分离度得分
    best_violation_f1 = 0
    best_normal_f1 = 0
    global_step = 0
    patience_counter = 0
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"===== 续训第 {epoch + 1}/{args.num_epochs} 轮 =====")
        
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        train_loss_details = {}
        train_pbar = tqdm(train_loader, desc="Continue Training", disable=(rank != 0))
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            with torch.amp.autocast('cuda', enabled=args.fp16, 
                                   dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if criterion is not None:
                    # 使用自定义损失函数
                    features = extract_features_from_model(model, input_ids, attention_mask)
                    
                    if isinstance(criterion, (ViolationNormalContrastiveLoss, ViolationNormalBoundaryLoss, 
                                            ViolationNormalTripletLoss, ConfidenceCalibrationLoss)):
                        loss, batch_loss_details = criterion(outputs.logits, features, outputs.loss)
                    elif isinstance(criterion, CombinedViolationNormalLoss):
                        loss, batch_loss_details = criterion(outputs.logits, features, labels)
                    else:
                        loss = criterion(outputs.logits, labels)
                        batch_loss_details = {}
                    
                    # 累积损失详情
                    for key, value in batch_loss_details.items():
                        if key not in train_loss_details:
                            train_loss_details[key] = []
                        train_loss_details[key].append(value)
                else:
                    loss = outputs.loss
                
                loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            if use_amp_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # 梯度累积和更新
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
                    
                    # 计算平均损失详情
                    avg_loss_details = {}
                    for key in train_loss_details:
                        avg_loss_details[key] = np.mean(train_loss_details[key])
                    
                    log_msg = f"Step {global_step} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                    if avg_loss_details:
                        detail_str = ', '.join([f"{k}: {v:.4f}" for k, v in avg_loss_details.items()])
                        log_msg += f", Details: {detail_str}"
                    
                    logger.info(log_msg)
                
                # 评估
                if global_step % args.eval_steps == 0:
                    val_results = enhanced_evaluate(model, val_loader, device, rank, criterion)
                    
                    if is_main_process:
                        vn_metrics = val_results['violation_normal_metrics']
                        
                        logger.info(f"验证结果 - Step {global_step}:")
                        logger.info(f"  总体指标 - Loss: {val_results['loss']:.4f}, F1: {val_results['f1']:.4f}")
                        logger.info(f"  违法违规类别 - P: {vn_metrics['violation_precision']:.4f}, "
                                   f"R: {vn_metrics['violation_recall']:.4f}, F1: {vn_metrics['violation_f1']:.4f}")
                        logger.info(f"  正常类别 - P: {vn_metrics['normal_precision']:.4f}, "
                                   f"R: {vn_metrics['normal_recall']:.4f}, F1: {vn_metrics['normal_f1']:.4f}")
                        logger.info(f"  混淆分析:")
                        logger.info(f"    违法违规 -> 正常 (漏报): {vn_metrics['violation_to_normal_count']}/{vn_metrics['violation_total_true']} "
                                   f"({vn_metrics['violation_miss_rate']:.4f})")
                        logger.info(f"    正常 -> 违法违规 (误报): {vn_metrics['normal_to_violation_count']}/{vn_metrics['normal_total_true']} "
                                   f"({vn_metrics['normal_false_positive_rate']:.4f})")
                        logger.info(f"  分离度得分: {vn_metrics['separation_score']:.4f}")
                        logger.info(f"  混淆惩罚: {vn_metrics['confusion_penalty']:.4f}")
                        
                        if val_results['loss_details']:
                            detail_str = ', '.join([f"{k}: {v:.4f}" for k, v in val_results['loss_details'].items()])
                            logger.info(f"  损失详情: {detail_str}")
                        
                        # 保存最佳模型（优先考虑分离度和违法违规F1）
                        improved = False
                        current_separation = vn_metrics['separation_score']
                        current_violation_f1 = vn_metrics['violation_f1']
                        current_normal_f1 = vn_metrics['normal_f1']
                        
                        if (current_separation > best_separation_score or 
                            (current_separation == best_separation_score and current_violation_f1 > best_violation_f1)):
                            best_separation_score = current_separation
                            best_violation_f1 = current_violation_f1
                            best_normal_f1 = current_normal_f1
                            improved = True
                        
                        if improved:
                            patience_counter = 0
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(os.path.join(args.output_dir, 'best_continue_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_continue_model'))
                            
                            logger.info(f"保存最佳续训模型 - 分离度: {best_separation_score:.4f}, "
                                       f"违法违规F1: {best_violation_f1:.4f}, 正常F1: {best_normal_f1:.4f}")
                            
                            # 保存详细评估结果
                            eval_results = {
                                'step': global_step,
                                'epoch': epoch + 1,
                                'separation_score': best_separation_score,
                                'violation_f1': best_violation_f1,
                                'normal_f1': best_normal_f1,
                                'detailed_metrics': val_results
                            }
                            with open(os.path.join(args.output_dir, 'best_continue_eval_results.json'), 'w', encoding='utf-8') as f:
                                json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
                        else:
                            patience_counter += 1
                            logger.info(f"指标未改善，耐心计数: {patience_counter}/{args.early_stopping_patience}")
                        
                        # 早停检查
                        if patience_counter >= args.early_stopping_patience:
                            logger.info(f"触发早停，在步骤 {global_step} 停止续训")
                            break
                    
                    model.train()
        
        # Epoch结束评估
        val_results = enhanced_evaluate(model, val_loader, device, rank, criterion)
        
        if is_main_process:
            vn_metrics = val_results['violation_normal_metrics']
            logger.info(f"Epoch {epoch + 1} 续训结果:")
            logger.info(f"  分离度得分: {vn_metrics['separation_score']:.4f}")
            logger.info(f"  违法违规F1: {vn_metrics['violation_f1']:.4f}")
            logger.info(f"  正常F1: {vn_metrics['normal_f1']:.4f}")
            logger.info(f"  混淆惩罚: {vn_metrics['confusion_penalty']:.4f}")
            
            # 保存每个epoch的模型
            epoch_output_dir = os.path.join(args.output_dir, f'continue-epoch-{epoch + 1}')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            logger.info(f"已保存续训第 {epoch + 1} 个epoch的模型")
    
    # 清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="违法违规-正常类别混淆问题续训")
    
    # 基础配置
    parser.add_argument('--base_checkpoint', type=str, 
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                       help="基础Qwen3模型路径")
    parser.add_argument('--lora_model_path', type=str, required=True,
                       help="现有LoRA模型路径，例如: ./lora-72-1815/epoch-6")
    parser.add_argument('--train_data', type=str, default="../data/balanced_train.xlsx")
    parser.add_argument('--val_data', type=str, default="../data/balanced_val.xlsx")
    parser.add_argument('--output_dir', type=str, default="../continue-lora-violation-normal")
    
    # 数据配置
    parser.add_argument('--balance_data', action='store_true', default=True,
                       help='是否平衡违法违规和正常类别的数据')
    parser.add_argument('--violation_oversample_rate', type=float, default=2.0,
                       help='违法违规样本过采样倍率')
    
    # 损失函数选择
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['contrastive', 'focal', 'boundary', 'triplet', 'combined', 'calibration', 'builtin'],
                       help='专门的损失函数类型')
    
    # 续训超参数（通常比初始训练更保守）
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='续训轮数，通常较少')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='续训学习率，通常比初始训练小')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    # 优化器配置
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    
    # 学习率调度
    parser.add_argument('--warmup_steps', type=int, default=50,
                       help='续训预热步数，通常较少')
    
    # 早停机制
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    
    # 日志和保存
    parser.add_argument('--logging_steps', type=int, default=25)
    parser.add_argument('--eval_steps', type=int, default=100)
    
    # 系统配置
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3")
    parser.add_argument('--clear_cache', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # 验证LoRA模型路径
    if not os.path.exists(args.lora_model_path):
        raise ValueError(f"LoRA模型路径不存在: {args.lora_model_path}")
    
    # GPU配置
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    total_gpus = torch.cuda.device_count()
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        for gpu_id in gpu_ids:
            if gpu_id >= total_gpus:
                raise ValueError(f"GPU ID {gpu_id} 超出范围")
        args.gpu_ids = gpu_ids
        world_size = len(gpu_ids)
    else:
        args.gpu_ids = list(range(total_gpus))
        world_size = total_gpus
    
    print("="*60)
    print("违法违规-正常类别混淆问题续训")
    print("="*60)
    print(f"基础模型: {args.base_checkpoint}")
    print(f"现有LoRA: {args.lora_model_path}")
    print(f"损失函数: {args.loss_type}")
    print(f"使用GPU: {args.gpu_ids}")
    print(f"输出目录: {args.output_dir}")
    print(f"续训轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    
    # 启动续训
    mp.spawn(continue_train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()