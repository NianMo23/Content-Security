import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from dataclasses import dataclass
from typing import Optional, Tuple
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import time
import logging
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from qwen3_classification_direct import Qwen3ForSequenceClassification
import torch.multiprocessing as mp
from contrastive_loss_utils import SimpleContrastiveLoss, FocalContrastiveLoss, MarginContrastiveLoss

@dataclass
class SequenceClassifierOutputWithFeatures(SequenceClassifierOutputWithPast):
    """
    扩展的分类输出，包含特征表示
    """
    features: Optional[torch.FloatTensor] = None

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Qwen3ForSequenceClassificationWithContrastive(nn.Module):
    """
    修改版的Qwen3分类模型，支持输出特征表示用于对比损失
    """
    def __init__(self, checkpoint, num_labels=6):
        super().__init__()
        
        # 加载基础模型
        print("正在加载Qwen3模型...")
        from transformers import AutoModelForCausalLM
        self.base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=None
        )
        
        # 保存配置信息
        self.config = self.base_model.config
        self.num_labels = num_labels
        self.hidden_size = self.config.hidden_size  # 2048
        
        # 直接替换lm_head层
        print(f"替换lm_head层: {self.hidden_size} -> {num_labels}")
        self.base_model.lm_head = nn.Linear(
            self.hidden_size,
            num_labels,
            bias=False
        )
        
        # 初始化新的分类头权重
        nn.init.xavier_uniform_(self.base_model.lm_head.weight)
        
        # 添加dropout层用于正则化
        self.dropout = nn.Dropout(0.1)
        
        # 添加特征投影层用于对比学习
        self.feature_projection = nn.Linear(self.hidden_size, 256)
        
        # 将新层移到正确的设备和数据类型
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype
        self.base_model.lm_head = self.base_model.lm_head.to(device=device, dtype=dtype)
        self.feature_projection = self.feature_projection.to(device=device, dtype=dtype)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_features=False,  # 新增参数，控制是否返回特征
        **kwargs
    ):
        # 获取Qwen3模型的基础输出（不包括lm_head）
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
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 使用序列的最后一个有效token（考虑padding）
        if attention_mask is not None:
            # 找到每个序列的最后一个有效token位置
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            sequence_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            # 如果没有attention_mask，使用最后一个token
            sequence_hidden_states = hidden_states[:, -1, :]
            
        # 应用dropout
        pooled_output = self.dropout(sequence_hidden_states)
        
        # 通过新的分类头获取logits
        logits = self.base_model.lm_head(pooled_output)  # [batch_size, num_labels]
        
        # 获取用于对比学习的特征表示
        features = self.feature_projection(pooled_output)  # [batch_size, 256]
        features = F.normalize(features, p=2, dim=1)  # L2归一化
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # 构建返回结果
        if return_features:
            result = SequenceClassifierOutputWithFeatures(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None,
                features=features
            )
        else:
            result = SequenceClassifierOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions if output_attentions else None
            )
            
        return result

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数实现
    更稳定的对比损失实现
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        计算InfoNCE损失
        Args:
            features: [batch_size, feature_dim] 归一化后的特征
            labels: [batch_size] 标签
        """
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 计算特征间的余弦相似度矩阵
        similarity_matrix = torch.mm(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        negative_mask = 1 - positive_mask
        
        # 移除对角线
        positive_mask.fill_diagonal_(0)
        
        # 对每个样本，计算InfoNCE损失
        loss = 0
        num_valid = 0
        
        for i in range(batch_size):
            # 获取正样本
            positive_indices = positive_mask[i].nonzero().squeeze(-1)
            if len(positive_indices) == 0:
                continue
            
            # 获取负样本
            negative_indices = negative_mask[i].nonzero().squeeze(-1)
            if len(negative_indices) == 0:
                continue
            
            # 正样本相似度
            positive_sim = similarity_matrix[i, positive_indices]
            
            # 负样本相似度
            negative_sim = similarity_matrix[i, negative_indices]
            
            # 计算InfoNCE损失
            # loss = -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
            pos_loss = 0
            for pos_sim in positive_sim:
                numerator = torch.exp(pos_sim)
                denominator = numerator + torch.sum(torch.exp(negative_sim))
                pos_loss += -torch.log(numerator / denominator + 1e-8)
            
            loss += pos_loss / len(positive_indices)
            num_valid += 1
        
        if num_valid > 0:
            loss = loss / num_valid
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss

class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失函数 (Supervised Contrastive Loss)
    基于论文 "Supervised Contrastive Learning" (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        计算监督对比损失
        Args:
            features: [batch_size, feature_dim] 归一化后的特征
            labels: [batch_size] 标签
        """
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 计算特征间的余弦相似度矩阵
        similarity_matrix = torch.mm(features, features.T) / self.temperature
        
        # 创建标签相等的掩码
        labels_reshaped = labels.unsqueeze(1)
        mask = torch.eq(labels_reshaped, labels_reshaped.T).float().to(device)
        
        # 去除对角线（自己与自己的相似度）
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(batch_size).view(-1, 1).to(device),
                                                     0)
        mask = mask * logits_mask
        
        # exp(相似度/温度) - 避免数值溢出
        max_sim = similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        similarity_matrix = similarity_matrix - max_sim
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        
        # 对于每个样本计算损失
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 只计算正样本对的损失
        # 计算每个样本的正样本数量
        pos_per_sample = mask.sum(1)
        
        # 如果某个样本没有正样本对，使用mask避免计算
        has_pos_sample = pos_per_sample > 0
        
        # 计算平均log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (pos_per_sample + 1e-8)
        
        # 损失是负的log-likelihood
        loss = -(mean_log_prob_pos * has_pos_sample.float()).sum() / (has_pos_sample.sum() + 1e-8)
        
        return loss

# 自定义数据集类
class ClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
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
            # 如果标签不在映射中，尝试将其作为数字处理
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

# 评估函数
def evaluate(model, dataloader, device, rank, use_fp16=True):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    # 获取模型数据类型
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度推理（评估时不需要特征）
            with torch.amp.autocast('cuda', enabled=use_fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_features=False
                )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 计算每个类别的准确率
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # 计算每个类别的准确率
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = []
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            per_class_accuracy.append(cm[i][i] / cm[i].sum())
        else:
            per_class_accuracy.append(0.0)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
        'per_class_support': support_per_class
    }

def setup_logging(rank, output_dir, gpu_id):
    """设置日志记录"""
    if rank == 0:  # 只有主进程记录日志
        # 创建日志目录
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名，包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_contrastive_log_{timestamp}.log")
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [GPU{gpu_id}] - %(levelname)s - %(message)s'.format(gpu_id=gpu_id),
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        # 创建日志记录器
        logger = logging.getLogger(__name__)
        logger.info(f"日志文件已创建: {log_file}")
        logger.info("="*80)
        logger.info("开始多GPU LoRA训练 - 对比损失版本")
        logger.info("="*80)
        
        return logger
    else:
        # 非主进程返回空记录器
        return logging.getLogger(__name__)

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
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
    
    # 设置当前GPU
    torch.cuda.set_device(actual_gpu_id)
    
    # 设置随机种子
    set_seed(args.seed)  # 所有进程使用相同的种子，保证一致性
    
    # 设置设备
    device = torch.device(f'cuda:{actual_gpu_id}')
    
    # 只在主进程打印信息
    is_main_process = rank == 0
    
    # 设置日志记录（只有主进程记录日志）
    logger = setup_logging(rank, args.output_dir, actual_gpu_id)
    
    if is_main_process:
        logger.info(f"Running DDP on rank {rank}, using GPU {actual_gpu_id}")
        logger.info(f"使用设备: {device}")
        logger.info(f"世界大小: {world_size}")
        logger.info(f"当前排名: {rank}")
        logger.info(f"对比损失权重: {args.contrastive_weight}")
        logger.info(f"对比损失温度: {args.temperature}")
    
    # 清理GPU缓存
    if args.clear_cache:
        torch.cuda.empty_cache()
        if is_main_process:
            logger.info(f"已清理GPU {actual_gpu_id} 缓存")
    
    if is_main_process:
        # 打印GPU内存信息
        if torch.cuda.is_available():
            logger.info("GPU内存信息:")
            for i, gpu_id in enumerate(args.gpu_ids if args.gpu_ids else range(world_size)):
                if gpu_id < torch.cuda.device_count():
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    logger.info(f"GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if is_main_process:
        logger.info("正在加载模型...")
    
    model = Qwen3ForSequenceClassificationWithContrastive(args.checkpoint, num_labels=6)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["lm_head", "feature_projection"]  # 保存分类头和特征投影层
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    if is_main_process:
        logger.info("LoRA配置:")
        model.print_trainable_parameters()
    
    # 将模型移到设备并包装为DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    
    # 初始化对比损失函数 - 使用更稳定的实现
    if args.contrastive_loss_type == 'simple':
        contrastive_loss_fn = SimpleContrastiveLoss(temperature=args.temperature)
    elif args.contrastive_loss_type == 'focal':
        contrastive_loss_fn = FocalContrastiveLoss(temperature=args.temperature, gamma=2.0)
    elif args.contrastive_loss_type == 'margin':
        contrastive_loss_fn = MarginContrastiveLoss(margin=0.2, temperature=args.temperature)
    else:
        # 默认使用修正后的SupervisedContrastiveLoss
        contrastive_loss_fn = SupervisedContrastiveLoss(temperature=args.temperature)
    
    if is_main_process:
        logger.info(f"使用对比损失类型: {args.contrastive_loss_type}")
    
    # 准备数据集
    if is_main_process:
        logger.info("加载数据集...")
    
    train_dataset = ClassificationDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ClassificationDataset(args.val_data, tokenizer, args.max_length)
    
    # 创建分布式采样器 - 确保不丢失数据
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False  # 确保不丢失训练数据
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    # 设置优化器和调度器 - 根据GPU数量调整学习率
    effective_lr = args.learning_rate * world_size  # 线性缩放学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    
    if is_main_process:
        logger.info(f"原始学习率: {args.learning_rate}, 调整后学习率: {effective_lr} (缩放因子: {world_size})")
    
    # 计算总训练步数 - 基于完整数据集而不是分割后的数据
    total_samples = len(train_dataset)
    samples_per_step = args.batch_size * world_size * args.gradient_accumulation_steps
    steps_per_epoch = total_samples // samples_per_step
    total_steps = steps_per_epoch * args.num_epochs
    
    if is_main_process:
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"每步处理样本数: {samples_per_step}")
        logger.info(f"每轮步数: {steps_per_epoch}")
        logger.info(f"总训练步数: {total_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
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
    global_step = 0
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"===== Epoch {epoch + 1}/{args.num_epochs} =====")
        
        # 设置分布式采样器的epoch
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_ce_loss = 0
        train_contrastive_loss = 0
        train_pbar = tqdm(train_loader, desc="Training", disable=(rank != 0))
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=args.fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                # 前向传播，同时获取特征
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_features=True
                )
                
                # 交叉熵损失
                ce_loss = outputs.loss
                
                # 对比损失
                features = outputs.features
                contrastive_loss = contrastive_loss_fn(features, labels)
                
                # 总损失：交叉熵损失 + 对比损失
                total_loss = ce_loss + args.contrastive_weight * contrastive_loss
                total_loss = total_loss / args.gradient_accumulation_steps
            
            if use_amp_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            train_loss += total_loss.item() * args.gradient_accumulation_steps
            train_ce_loss += ce_loss.item()
            train_contrastive_loss += contrastive_loss.item()
            
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
                    avg_total_loss = train_loss / (step + 1)
                    avg_ce_loss = train_ce_loss / (step + 1)
                    avg_contrastive_loss = train_contrastive_loss / (step + 1)
                    train_pbar.set_postfix({
                        'total_loss': f'{avg_total_loss:.4f}',
                        'ce_loss': f'{avg_ce_loss:.4f}',
                        'cont_loss': f'{avg_contrastive_loss:.4f}'
                    })
                
                # 评估
                if global_step % args.eval_steps == 0:
                    val_results = evaluate(model, val_loader, device, rank)
                    
                    if is_main_process:
                        logger.info(f"验证集结果 - Step {global_step}:")
                        logger.info(f"Loss: {val_results['loss']:.4f}")
                        logger.info(f"Accuracy: {val_results['accuracy']:.4f}")
                        logger.info(f"F1: {val_results['f1']:.4f}")
                        logger.info(f"Precision: {val_results['precision']:.4f}")
                        logger.info(f"Recall: {val_results['recall']:.4f}")
                        
                        # 打印各类别准确率
                        logger.info("各类别准确率:")
                        label_names = ['正常', '歧视', '违法违规', '政治安全', '暴恐', '色情低俗']
                        for i, (name, acc) in enumerate(zip(label_names, val_results['per_class_accuracy'])):
                            logger.info(f"  {name}: {acc:.4f}")
                        
                        # 保存最佳模型
                        if val_results['f1'] > best_val_f1:
                            best_val_f1 = val_results['f1']
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                            logger.info(f"保存最佳模型，F1: {best_val_f1:.4f}")
                    
                    model.train()
                
                # 定期保存检查点
                if global_step % args.save_steps == 0 and is_main_process:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(os.path.join(args.output_dir, f'checkpoint-{global_step}'))
        
        # Epoch结束后的评估
        val_results = evaluate(model, val_loader, device, rank)
        
        if is_main_process:
            avg_total_loss = train_loss / len(train_loader)
            avg_ce_loss = train_ce_loss / len(train_loader)
            avg_contrastive_loss = train_contrastive_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch + 1} 训练损失:")
            logger.info(f"Total Loss: {avg_total_loss:.4f}")
            logger.info(f"CrossEntropy Loss: {avg_ce_loss:.4f}")
            logger.info(f"Contrastive Loss: {avg_contrastive_loss:.4f}")
            
            logger.info(f"Epoch {epoch + 1} 验证集结果:")
            logger.info(f"Loss: {val_results['loss']:.4f}")
            logger.info(f"Accuracy: {val_results['accuracy']:.4f}")
            logger.info(f"F1: {val_results['f1']:.4f}")
            
            # 每个epoch结束后保存LoRA模型
            epoch_output_dir = os.path.join(args.output_dir, f'epoch-{epoch + 1}')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            logger.info(f"已保存第 {epoch + 1} 个epoch的LoRA模型到: {epoch_output_dir}")
    
    # 训练完成
    if is_main_process:
        logger.info("="*80)
        logger.info("训练完成!")
        logger.info(f"最佳验证F1得分: {best_val_f1:.4f}")
        logger.info("="*80)
    
    # 确保所有进程都完成训练后再清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default="/home/users/sx_yanghx2/models/Qwen3-4B")
    parser.add_argument('--train_data', type=str, default="./balanced_train.xlsx")
    parser.add_argument('--val_data', type=str, default="./balanced_val.xlsx")
    parser.add_argument('--test_data', type=str, default="./test.xlsx")
    parser.add_argument('--output_dir', type=str, default="./lora_model_multi_gpu_contrastive/Qwen3-4B-202507031830")
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                       help='指定使用的GPU ID，用逗号分隔，例如: 0,1,2 或 0,2,4')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='指定使用的GPU数量，从GPU 0开始使用')
    parser.add_argument('--clear_cache', action='store_true', default=True,
                       help='训练前清理GPU缓存')
    
    # 对比损失相关参数
    parser.add_argument('--contrastive_weight', type=float, default=1,
                       help='对比损失的权重')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='对比损失的温度参数')
    parser.add_argument('--contrastive_loss_type', type=str, default='focal',
                       choices=['simple', 'focal', 'margin', 'supervised'],
                       help='对比损失类型')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行多GPU训练")
    
    # 处理GPU配置
    total_gpus = torch.cuda.device_count()
    print(f"系统检测到 {total_gpus} 个GPU")
    
    # 创建输出目录（提前创建，确保日志目录可以建立）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析GPU配置
    if args.gpu_ids is not None:
        # 用户指定了具体的GPU ID
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        # 验证GPU ID的有效性
        for gpu_id in gpu_ids:
            if gpu_id >= total_gpus:
                raise ValueError(f"指定的GPU ID {gpu_id} 超出可用范围 (0-{total_gpus-1})")
        args.gpu_ids = gpu_ids
        world_size = len(gpu_ids)
        print(f"使用指定的GPU: {gpu_ids}")
    elif args.num_gpus is not None:
        # 用户指定了GPU数量
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
    
    print(f"对比损失配置:")
    print(f"  权重: {args.contrastive_weight}")
    print(f"  温度: {args.temperature}")
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()