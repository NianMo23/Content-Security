import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
import random
import re
from collections import Counter

# 导入基础损失函数
from violation_normal_loss import CombinedViolationNormalLoss
from qwen3_classification_direct import Qwen3ForSequenceClassification

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

class GeneralizationLoss(nn.Module):
    """
    专门提升泛化能力的损失函数
    结合了多种泛化增强技术
    """
    def __init__(self, 
                 lambda_consistency=0.3,      # 一致性正则化权重
                 lambda_diversity=0.2,        # 多样性损失权重  
                 lambda_confidence=0.1,       # 置信度校准权重
                 lambda_smoothness=0.15,      # 平滑性正则化权重
                 temperature=3.0,             # 知识蒸馏温度
                 dropout_rate=0.1):           # 增强Dropout比率
        super().__init__()
        
        self.lambda_consistency = lambda_consistency
        self.lambda_diversity = lambda_diversity
        self.lambda_confidence = lambda_confidence
        self.lambda_smoothness = lambda_smoothness
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        
        # 基础损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 标签平滑
        self.label_smoothing = 0.1
        
    def forward(self, logits, features, labels, logits_clean=None):
        """
        Args:
            logits: 模型输出的logits
            features: 模型特征表示
            labels: 真实标签
            logits_clean: 无dropout的clean logits（用于一致性正则化）
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        device = logits.device
        
        # 1. 基础交叉熵损失（带标签平滑）
        smoothed_labels = self._label_smoothing(labels, num_classes, device)
        ce_loss = F.cross_entropy(logits, smoothed_labels)
        
        total_loss = ce_loss
        loss_details = {'ce_loss': ce_loss.item()}
        
        # 2. 一致性正则化损失（如果提供了clean logits）
        if logits_clean is not None and self.lambda_consistency > 0:
            consistency_loss = F.kl_div(
                F.log_softmax(logits / self.temperature, dim=1),
                F.softmax(logits_clean / self.temperature, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            total_loss += self.lambda_consistency * consistency_loss
            loss_details['consistency_loss'] = consistency_loss.item()
        
        # 3. 特征多样性损失（增强特征表示的多样性）
        if self.lambda_diversity > 0:
            diversity_loss = self._compute_diversity_loss(features)
            total_loss += self.lambda_diversity * diversity_loss
            loss_details['diversity_loss'] = diversity_loss.item()
        
        # 4. 置信度校准损失（防止过度自信）
        if self.lambda_confidence > 0:
            confidence_loss = self._compute_confidence_calibration_loss(logits, labels)
            total_loss += self.lambda_confidence * confidence_loss
            loss_details['confidence_loss'] = confidence_loss.item()
        
        # 5. 平滑性正则化（相似输入应该有相似输出）
        if self.lambda_smoothness > 0:
            smoothness_loss = self._compute_smoothness_loss(features, logits)
            total_loss += self.lambda_smoothness * smoothness_loss
            loss_details['smoothness_loss'] = smoothness_loss.item()
        
        return total_loss, loss_details
    
    def _label_smoothing(self, labels, num_classes, device):
        """标签平滑"""
        smoothed = torch.full((labels.size(0), num_classes), 
                             self.label_smoothing / (num_classes - 1), 
                             device=device)
        smoothed.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
        return smoothed
    
    def _compute_diversity_loss(self, features):
        """计算特征多样性损失，鼓励不同样本有不同的特征表示"""
        # 计算特征间的相似性
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 除了对角线外，其他位置的相似性应该尽可能小
        batch_size = features.size(0)
        mask = torch.eye(batch_size, device=features.device).bool()
        off_diagonal = similarity_matrix[~mask]
        
        # 多样性损失：非对角线元素的平方和
        diversity_loss = torch.mean(off_diagonal ** 2)
        return diversity_loss
    
    def _compute_confidence_calibration_loss(self, logits, labels):
        """计算置信度校准损失，防止模型过度自信"""
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        # 目标：正确预测的样本置信度适中，错误预测的置信度应该低
        predictions = torch.argmax(logits, dim=1)
        correct_mask = (predictions == labels).float()
        
        # 正确预测的样本：置信度不要太高（避免过拟合）
        correct_confidence_loss = torch.mean(correct_mask * (max_probs - 0.8) ** 2)
        
        # 错误预测的样本：置信度应该低
        wrong_confidence_loss = torch.mean((1 - correct_mask) * max_probs ** 2)
        
        return correct_confidence_loss + wrong_confidence_loss
    
    def _compute_smoothness_loss(self, features, logits):
        """计算平滑性损失，相似特征应该有相似的预测"""
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算特征相似性
        features_norm = F.normalize(features, p=2, dim=1)
        feature_sim = torch.mm(features_norm, features_norm.t())
        
        # 计算预测相似性
        probs = F.softmax(logits, dim=1)
        pred_sim = torch.mm(probs, probs.t())
        
        # 平滑性损失：特征相似的样本应该有相似的预测
        smoothness_loss = F.mse_loss(feature_sim, pred_sim)
        return smoothness_loss

class DataAugmentation:
    """
    专门针对违法违规和正常类别的数据增强
    提升泛化能力
    """
    def __init__(self, aug_prob=0.5):
        self.aug_prob = aug_prob
        
        # 同义词替换词典（可以根据实际情况扩展）
        self.synonyms = {
            '违法': ['非法', '不合法', '触法', '犯法'],
            '违规': ['违反规定', '不符合规定', '违反规则', '不合规'],
            '正常': ['合法', '合规', '正当', '合理', '正确'],
            '内容': ['信息', '文本', '资料', '材料'],
            '发布': ['发送', '传播', '分享', '公布'],
            '用户': ['使用者', '网友', '人员', '个人']
        }
        
    def augment_text(self, text):
        """对文本进行数据增强"""
        if random.random() > self.aug_prob:
            return text
        
        augmented_texts = []
        
        # 1. 同义词替换
        aug_text1 = self._synonym_replacement(text)
        augmented_texts.append(aug_text1)
        
        # 2. 随机删除（删除不重要的词）
        aug_text2 = self._random_deletion(text, delete_prob=0.1)
        augmented_texts.append(aug_text2)
        
        # 3. 随机插入同义词
        aug_text3 = self._random_insertion(text)
        augmented_texts.append(aug_text3)
        
        # 4. 句子重排（对于较长文本）
        aug_text4 = self._sentence_shuffle(text)
        augmented_texts.append(aug_text4)
        
        # 随机选择一种增强方式
        return random.choice(augmented_texts)
    
    def _synonym_replacement(self, text, replace_prob=0.3):
        """同义词替换"""
        words = list(text)
        new_words = []
        
        i = 0
        while i < len(words):
            replaced = False
            if random.random() < replace_prob:
                # 尝试匹配多字词
                for length in [3, 2, 1]:
                    if i + length <= len(words):
                        word = ''.join(words[i:i+length])
                        if word in self.synonyms:
                            synonym = random.choice(self.synonyms[word])
                            new_words.extend(list(synonym))
                            i += length
                            replaced = True
                            break
            
            if not replaced:
                new_words.append(words[i])
                i += 1
        
        return ''.join(new_words)
    
    def _random_deletion(self, text, delete_prob=0.1):
        """随机删除字符"""
        if len(text) <= 10:  # 太短的文本不删除
            return text
        
        chars = list(text)
        new_chars = []
        
        for char in chars:
            if random.random() > delete_prob:
                new_chars.append(char)
        
        if len(new_chars) == 0:
            return text
        
        return ''.join(new_chars)
    
    def _random_insertion(self, text):
        """随机插入同义词"""
        words = text.split()
        if len(words) < 2:
            return text
        
        # 选择一个位置插入同义词
        insert_pos = random.randint(0, len(words))
        synonym_word = random.choice(list(self.synonyms.keys()))
        synonym = random.choice(self.synonyms[synonym_word])
        
        words.insert(insert_pos, synonym)
        return ''.join(words)
    
    def _sentence_shuffle(self, text):
        """句子重排"""
        # 简单的句子分割（根据标点符号）
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '。'.join(sentences) + '。'
        
        return text

class GeneralizationDataset(Dataset):
    """
    增强泛化能力的数据集
    包含数据增强和多种采样策略
    """
    def __init__(self, data_path, tokenizer, max_length=512, 
                 use_augmentation=True, augmentation_ratio=1.0,
                 focus_violation_normal=True):
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        self.augmentation_ratio = augmentation_ratio
        self.focus_violation_normal = focus_violation_normal
        
        # 数据增强器
        self.augmentor = DataAugmentation(aug_prob=0.5)
        
        # 标签映射
        self.label_map = {
            '正常': 0,
            '歧视': 1,
            '违法违规': 2,
            '政治安全': 3,
            '暴恐': 4,
            '色情低俗': 5
        }
        
        # 处理数据
        self._prepare_data()
        
        print(f"泛化增强数据集大小: {len(self.processed_data)}")
        self._print_class_distribution()
    
    def _prepare_data(self):
        """准备增强泛化的数据"""
        self.processed_data = []
        
        # 原始数据
        for _, row in self.data.iterrows():
            text = str(row['text_cn'])
            label_text = str(row['extracted_label']).strip()
            
            if label_text in self.label_map:
                label = self.label_map[label_text]
                self.processed_data.append({
                    'text': text,
                    'label': label,
                    'is_augmented': False
                })
        
        # 数据增强（特别针对违法违规和正常类别）
        if self.use_augmentation:
            augmented_count = 0
            target_labels = [0, 2] if self.focus_violation_normal else list(range(6))
            
            for _, row in self.data.iterrows():
                text = str(row['text_cn'])
                label_text = str(row['extracted_label']).strip()
                
                if label_text in self.label_map:
                    label = self.label_map[label_text]
                    
                    # 对目标类别进行数据增强
                    if label in target_labels:
                        num_augmentations = int(self.augmentation_ratio)
                        for _ in range(num_augmentations):
                            aug_text = self.augmentor.augment_text(text)
                            if aug_text != text:  # 确保增强后的文本不同
                                self.processed_data.append({
                                    'text': aug_text,
                                    'label': label,
                                    'is_augmented': True
                                })
                                augmented_count += 1
            
            print(f"生成了 {augmented_count} 个增强样本")
        
        # 打乱数据
        random.shuffle(self.processed_data)
    
    def _print_class_distribution(self):
        """打印类别分布"""
        label_counts = {}
        aug_counts = {}
        
        for item in self.processed_data:
            label = item['label']
            is_aug = item['is_augmented']
            
            if label not in label_counts:
                label_counts[label] = 0
                aug_counts[label] = 0
            
            label_counts[label] += 1
            if is_aug:
                aug_counts[label] += 1
        
        label_names = {v: k for k, v in self.label_map.items()}
        
        print("类别分布（原始+增强）:")
        for label_id, count in label_counts.items():
            label_name = label_names.get(label_id, f"未知{label_id}")
            aug_count = aug_counts[label_id]
            orig_count = count - aug_count
            print(f"  {label_name}: {count} (原始: {orig_count}, 增强: {aug_count})")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        text = item['text']
        label = item['label']
        
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
            'labels': torch.tensor(label, dtype=torch.long),
            'is_augmented': torch.tensor(item['is_augmented'], dtype=torch.bool)
        }

def generalization_train_worker(rank, world_size, args):
    """泛化增强训练工作进程"""
    actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
    
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 设置日志
    logger = setup_logging(rank, args.output_dir)
    
    if rank == 0:
        logger.info("="*60)
        logger.info("开始泛化能力增强续训")
        logger.info("="*60)
        logger.info(f"基础模型路径: {args.base_checkpoint}")
        logger.info(f"LoRA模型路径: {args.lora_model_path}")
        logger.info(f"数据增强倍率: {args.augmentation_ratio}")
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
    
    # 创建泛化增强数据集
    if is_main_process:
        logger.info("创建泛化增强数据集...")
    
    train_dataset = GeneralizationDataset(
        args.train_data, tokenizer, args.max_length,
        use_augmentation=True,
        augmentation_ratio=args.augmentation_ratio,
        focus_violation_normal=args.focus_violation_normal
    )
    val_dataset = GeneralizationDataset(
        args.val_data, tokenizer, args.max_length,
        use_augmentation=False,  # 验证集不增强
        focus_violation_normal=args.focus_violation_normal
    )
    
    # 加载预训练的LoRA模型
    if is_main_process:
        logger.info("加载现有LoRA模型...")
    
    base_model = Qwen3ForSequenceClassification(args.base_checkpoint, num_labels=6)
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    
    # 为一致性正则化创建无dropout版本
    model_clean = PeftModel.from_pretrained(
        Qwen3ForSequenceClassification(args.base_checkpoint, num_labels=6), 
        args.lora_model_path
    )
    model_clean.eval()
    
    # 移动到设备
    model.to(device)
    model_clean.to(device)
    
    # DDP包装
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    model_clean = DDP(model_clean, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    
    # 创建数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True, drop_last=False
    )
    
    # 设置优化器（泛化训练使用更小的学习率）
    generalization_lr = args.learning_rate * 0.5 * world_size  # 进一步降低学习率
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=generalization_lr,
        weight_decay=args.weight_decay * 2,  # 增加权重衰减
        eps=args.adam_epsilon
    )
    
    # 学习率调度器
    total_samples = len(train_dataset)
    samples_per_step = args.batch_size * world_size * args.gradient_accumulation_steps
    steps_per_epoch = total_samples // samples_per_step
    total_steps = steps_per_epoch * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps * 2,  # 增加预热步数
        num_training_steps=total_steps
    )
    
    # 泛化损失函数
    criterion = GeneralizationLoss(
        lambda_consistency=0.3,
        lambda_diversity=0.2,
        lambda_confidence=0.1,
        lambda_smoothness=0.15,
        temperature=3.0
    )
    
    if is_main_process:
        logger.info("使用泛化增强损失函数")
    
    # 混合精度设置
    model_dtype = next(model.parameters()).dtype
    use_amp_scaler = args.fp16 and model_dtype != torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if use_amp_scaler else None
    
    # 训练循环
    best_generalization_score = 0
    global_step = 0
    patience_counter = 0
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"===== 泛化增强第 {epoch + 1}/{args.num_epochs} 轮 =====")
        
        train_sampler.set_epoch(epoch)
        model.train()
        model_clean.eval()
        
        train_loss = 0
        train_loss_details = {}
        train_pbar = tqdm(train_loader, desc="Generalization Training", disable=(rank != 0))
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            with torch.amp.autocast('cuda', enabled=args.fp16, 
                                   dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                # 训练模式的输出（有dropout）
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 提取特征
                features = extract_features_from_model(model, input_ids, attention_mask)
                
                # Clean模式的输出（无dropout，用于一致性正则化）
                with torch.no_grad():
                    outputs_clean = model_clean(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # 计算泛化损失
                loss, batch_loss_details = criterion(
                    outputs.logits, features, labels, outputs_clean.logits
                )
                
                loss = loss / args.gradient_accumulation_steps
                
                # 累积损失详情
                for key, value in batch_loss_details.items():
                    if key not in train_loss_details:
                        train_loss_details[key] = []
                    train_loss_details[key].append(value)
            
            # 反向传播
            if use_amp_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # 梯度更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_amp_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更严格的梯度裁剪
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 定期日志
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
                    val_results = enhanced_evaluate(model, val_loader, device, rank)
                    
                    if is_main_process:
                        vn_metrics = val_results['violation_normal_metrics']
                        
                        # 计算泛化得分（综合多个指标）
                        separation_score = vn_metrics['separation_score']
                        f1_balance = min(vn_metrics['violation_f1'], vn_metrics['normal_f1'])  # 取较小的F1
                        confusion_penalty = vn_metrics['confusion_penalty']
                        generalization_score = separation_score + f1_balance - confusion_penalty
                        
                        logger.info(f"验证结果 - Step {global_step}:")
                        logger.info(f"  泛化得分: {generalization_score:.4f}")
                        logger.info(f"  分离度: {separation_score:.4f}")
                        logger.info(f"  F1平衡: {f1_balance:.4f}")
                        logger.info(f"  混淆惩罚: {confusion_penalty:.4f}")
                        logger.info(f"  违法违规F1: {vn_metrics['violation_f1']:.4f}")
                        logger.info(f"  正常F1: {vn_metrics['normal_f1']:.4f}")
                        
                        # 保存最佳泛化模型
                        if generalization_score > best_generalization_score:
                            best_generalization_score = generalization_score
                            patience_counter = 0
                            
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(os.path.join(args.output_dir, 'best_generalization_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_generalization_model'))
                            
                            logger.info(f"保存最佳泛化模型 - 得分: {best_generalization_score:.4f}")
                            
                            # 保存详细结果
                            eval_results = {
                                'step': global_step,
                                'epoch': epoch + 1,
                                'generalization_score': best_generalization_score,
                                'detailed_metrics': val_results
                            }
                            with open(os.path.join(args.output_dir, 'best_generalization_results.json'), 'w', encoding='utf-8') as f:
                                json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
                        else:
                            patience_counter += 1
                            logger.info(f"泛化得分未改善，耐心计数: {patience_counter}/{args.early_stopping_patience}")
                        
                        # 早停
                        if patience_counter >= args.early_stopping_patience:
                            logger.info(f"触发早停，在步骤 {global_step} 停止训练")
                            break
                    
                    model.train()
    
    # 清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    cleanup_distributed()

# 使用之前定义的辅助函数
def setup_distributed(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'  # 不同端口
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def setup_logging(rank, output_dir):
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_file = os.path.join(output_dir, 'generalization_training.log')
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    return None

def extract_features_from_model(model, input_ids, attention_mask):
    """提取模型特征"""
    # 获取基础模型 - 需要直接访问transformer部分
    if hasattr(model, 'module'):
        # DDP包装的模型
        if hasattr(model.module.base_model, 'model'):
            # PeftModel -> base_model -> model (transformer)
            base_model = model.module.base_model.model
        else:
            # 直接是transformer
            base_model = model.module.base_model
    else:
        # 非DDP模型
        if hasattr(model.base_model, 'model'):
            # PeftModel -> base_model -> model (transformer)
            base_model = model.base_model.model
        else:
            # 直接是transformer
            base_model = model.base_model
    
    # 调用transformer模型获取隐藏状态
    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    # 获取最后一层的隐藏状态
    if hasattr(outputs, 'last_hidden_state'):
        hidden_states = outputs.last_hidden_state
    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        # 如果没有last_hidden_state，使用最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]
    else:
        raise AttributeError("无法从模型输出中获取隐藏状态")
    
    # 池化操作：取最后一个有效token的隐藏状态
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

def enhanced_evaluate(model, dataloader, device, rank):
    """评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算违法违规-正常混淆指标
    violation_normal_metrics = analyze_violation_normal_confusion(all_labels, all_preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'violation_normal_metrics': violation_normal_metrics
    }

def analyze_violation_normal_confusion(labels, predictions):
    """分析违法违规和正常类别混淆"""
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    violation_true = (labels == 2)
    violation_pred = (predictions == 2)
    normal_true = (labels == 0)
    normal_pred = (predictions == 0)
    
    violation_tp = np.sum(violation_true & violation_pred)
    violation_fp = np.sum(~violation_true & violation_pred)
    violation_fn = np.sum(violation_true & ~violation_pred)
    
    normal_tp = np.sum(normal_true & normal_pred)
    normal_fp = np.sum(~normal_true & normal_pred)
    normal_fn = np.sum(normal_true & ~normal_pred)
    
    violation_precision = violation_tp / (violation_tp + violation_fp) if (violation_tp + violation_fp) > 0 else 0
    violation_recall = violation_tp / (violation_tp + violation_fn) if (violation_tp + violation_fn) > 0 else 0
    violation_f1 = 2 * violation_precision * violation_recall / (violation_precision + violation_recall) if (violation_precision + violation_recall) > 0 else 0
    
    normal_precision = normal_tp / (normal_tp + normal_fp) if (normal_tp + normal_fp) > 0 else 0
    normal_recall = normal_tp / (normal_tp + normal_fn) if (normal_tp + normal_fn) > 0 else 0
    normal_f1 = 2 * normal_precision * normal_recall / (normal_precision + normal_recall) if (normal_precision + normal_recall) > 0 else 0
    
    # 混淆情况
    violation_to_normal = np.sum(violation_true & normal_pred)
    normal_to_violation = np.sum(normal_true & violation_pred)
    
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
        'separation_score': (violation_precision + violation_recall + normal_precision + normal_recall) / 4,
        'confusion_penalty': violation_miss_rate + normal_false_positive_rate
    }

def main():
    parser = argparse.ArgumentParser(description="泛化能力增强续训")
    
    # 基础配置
    parser.add_argument('--base_checkpoint', type=str, 
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B")
    parser.add_argument('--lora_model_path', type=str, required=True)
    parser.add_argument('--train_data', type=str, default="../data/balanced_train.xlsx")
    parser.add_argument('--val_data', type=str, default="../data/balanced_val.xlsx")
    parser.add_argument('--output_dir', type=str, default="../generalization-lora")
    
    # 泛化增强配置
    parser.add_argument('--augmentation_ratio', type=float, default=2.0,
                       help='数据增强倍率')
    parser.add_argument('--focus_violation_normal', action='store_true', default=True,
                       help='专注于违法违规和正常类别')
    
    # 训练配置
    parser.add_argument('--num_epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=12)  # 较小batch size增强泛化
    parser.add_argument('--learning_rate', type=float, default=8e-6)  # 更小学习率
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0.02)  # 增加正则化
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=80)
    
    # 系统配置
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3")
    parser.add_argument('--clear_cache', action='store_true', default=True)
    
    args = parser.parse_args()
    
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
    print("泛化能力增强续训")
    print("="*60)
    print(f"基础模型: {args.base_checkpoint}")
    print(f"现有LoRA: {args.lora_model_path}")
    print(f"数据增强倍率: {args.augmentation_ratio}")
    print(f"学习率: {args.learning_rate}")
    print(f"权重衰减: {args.weight_decay}")
    
    # 启动训练
    mp.spawn(generalization_train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()