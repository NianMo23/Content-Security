import torch
import torch.nn as nn
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
from qwen3_simple_model import Qwen3ForClassification
import torch.multiprocessing as mp
import json

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text  # 保留原始文本
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
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 计算每个类别的指标
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
    
    # 计算详细的漏报和误报分析
    class_names = ['正常', '歧视', '违法违规', '政治安全', '暴恐', '色情低俗']
    normal_class = 0
    abnormal_classes = [1, 2, 3, 4, 5]
    
    # 计算每一类的漏报率（每个异常类别被误判为正常的比例）
    per_class_miss_rate = {}
    
    for class_idx in abnormal_classes:
        class_name = class_names[class_idx]
        # 获取该类的真实样本
        true_class_mask = np.array(all_labels) == class_idx
        true_class_count = np.sum(true_class_mask)
        
        if true_class_count > 0:
            # 该类被误判为正常的数量（漏报）
            pred_class_labels = np.array(all_preds)[true_class_mask]
            miss_to_normal_count = np.sum(pred_class_labels == normal_class)
            miss_rate = miss_to_normal_count / true_class_count
            
            per_class_miss_rate[class_name] = {
                'rate': miss_rate,
                'count': int(miss_to_normal_count),
                'total': int(true_class_count)
            }
        else:
            per_class_miss_rate[class_name] = {
                'rate': 0.0,
                'count': 0,
                'total': 0
            }
    
    # 计算总漏报率（所有异常类别被误判为正常的总体情况）
    all_abnormal_mask = np.isin(all_labels, abnormal_classes)
    total_abnormal_count = np.sum(all_abnormal_mask)
    
    if total_abnormal_count > 0:
        # 异常样本被预测为正常的数量
        abnormal_pred_labels = np.array(all_preds)[all_abnormal_mask]
        total_miss_to_normal_count = np.sum(abnormal_pred_labels == normal_class)
        total_miss_rate = total_miss_to_normal_count / total_abnormal_count
    else:
        total_miss_rate = 0.0
        total_miss_to_normal_count = 0
    
    # 计算正常类的误报率（正常被误判为异常的比例）
    normal_mask = np.array(all_labels) == normal_class
    normal_count = np.sum(normal_mask)
    
    if normal_count > 0:
        # 正常样本被预测为异常的数量
        normal_pred_labels = np.array(all_preds)[normal_mask]
        normal_false_positive_count = np.sum(normal_pred_labels != normal_class)
        normal_false_positive_rate = normal_false_positive_count / normal_count
        
        # 统计正常被误判为各个异常类别的情况
        normal_misclassified_to = {}
        for class_idx in abnormal_classes:
            class_name = class_names[class_idx]
            misclassified_count = np.sum(normal_pred_labels == class_idx)
            misclassified_rate = misclassified_count / normal_count
            normal_misclassified_to[class_name] = {
                'count': int(misclassified_count),
                'rate': misclassified_rate
            }
    else:
        normal_false_positive_rate = 0.0
        normal_false_positive_count = 0
        normal_misclassified_to = {}
    
    # 计算二分类指标（正常vs异常）
    binary_true = np.array([0 if label == normal_class else 1 for label in all_labels])
    binary_pred = np.array([0 if pred == normal_class else 1 for pred in all_preds])
    
    # 计算二分类混淆矩阵
    binary_cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    
    if binary_cm.shape == (2, 2):
        tn, fp, fn, tp = binary_cm.ravel()
        binary_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        binary_accuracy = 0.0
    
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
        'per_class_support': support_per_class,
        # 新增的详细漏报和误报分析
        'per_class_miss_rate': per_class_miss_rate,
        'total_miss_rate': total_miss_rate,
        'total_miss_count': int(total_miss_to_normal_count),
        'total_abnormal_count': int(total_abnormal_count),
        'normal_false_positive_rate': normal_false_positive_rate,
        'normal_false_positive_count': int(normal_false_positive_count),
        'normal_total_count': int(normal_count),
        'normal_misclassified_to': normal_misclassified_to,
        'binary_accuracy': binary_accuracy,
        'binary_confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
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
        log_file = os.path.join(output_dir, 'training.log')
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

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
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
    
    if rank == 0:
        logger.info(f"开始多GPU训练，使用 {world_size} 个GPU")
        logger.info(f"主进程运行在 rank {rank}，使用 GPU {actual_gpu_id}")
    
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
    
    # 只在主进程打印信息
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"使用设备: {device}")
        logger.info(f"世界大小: {world_size}")
        logger.info(f"当前排名: {rank}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if is_main_process:
        logger.info("正在加载Qwen3模型...")
    
    model = Qwen3ForClassification(args.checkpoint)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    
    # 应用LoRA
    model.model = get_peft_model(model.model, lora_config)
    
    if is_main_process:
        logger.info("LoRA配置:")
        # 获取可训练参数信息并记录到日志
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 将模型移到设备并包装为DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    
    # 设置静态图，告诉DDP模型图在训练过程中不会改变
    model._set_static_graph()
    
    # 准备数据集
    if is_main_process:
        logger.info("加载数据集...")
    
    train_dataset = ClassificationDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ClassificationDataset(args.val_data, tokenizer, args.max_length)
    
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
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"===== 开始第 {epoch + 1}/{args.num_epochs} 轮训练 =====")
        
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
                    val_results = evaluate(model, val_loader, device, rank, args.fp16)
                    
                    if is_main_process:
                        logger.info(f"验证集结果 - Step {global_step}:")
                        logger.info(f"  Loss: {val_results['loss']:.4f}")
                        logger.info(f"  6分类准确率: {val_results['accuracy']:.4f}")
                        logger.info(f"  F1: {val_results['f1']:.4f}")
                        logger.info(f"  二分类准确率: {val_results['binary_accuracy']:.4f}")
                        
                        # 保存最佳模型
                        if val_results['f1'] > best_val_f1:
                            best_val_f1 = val_results['f1']
                            patience_counter = 0
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                            logger.info(f"保存最佳模型，F1: {val_results['f1']:.4f}")
                        else:
                            patience_counter += 1
                            
                        # 早停检查
                        if patience_counter >= args.early_stopping_patience:
                            logger.info(f"触发早停，在步骤 {global_step} 停止训练")
                            if torch.distributed.is_initialized():
                                torch.distributed.barrier()
                            cleanup_distributed()
                            return
                    
                    model.train()
        
        # Epoch结束后的评估
        val_results = evaluate(model, val_loader, device, rank, args.fp16)
        
        if is_main_process:
            logger.info(f"Epoch {epoch + 1} 验证集最终结果:")
            logger.info(f"  Loss: {val_results['loss']:.4f}")
            logger.info(f"  6分类准确率: {val_results['accuracy']:.4f}")
            logger.info(f"  F1: {val_results['f1']:.4f}")
            logger.info(f"  二分类准确率: {val_results['binary_accuracy']:.4f}")
            
            # 每个epoch结束后保存模型
            epoch_output_dir = os.path.join(args.output_dir, f'epoch-{epoch + 1}')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.model.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            logger.info(f"已保存第 {epoch + 1} 个epoch的模型到: {epoch_output_dir}")
    
    # 确保所有进程都完成训练后再清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument('--checkpoint', type=str,
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B")
    parser.add_argument('--train_data', type=str, default="../data/balanced_train.xlsx")
    parser.add_argument('--val_data', type=str, default="../data/balanced_val.xlsx")
    parser.add_argument('--output_dir', type=str, default="../simple-model-output")
    
    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--target_modules', type=str, nargs='+',
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    
    # 学习率调度
    parser.add_argument('--warmup_steps', type=int, default=100)
    
    # 早停机制
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    
    # 日志和保存
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=100)
    
    # 系统配置
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3",
                       help='指定使用的GPU ID，用逗号分隔')
    parser.add_argument('--clear_cache', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行多GPU训练")
    
    # 处理GPU配置
    total_gpus = torch.cuda.device_count()
    print(f"系统检测到 {total_gpus} 个GPU")
    
    # 解析GPU配置
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    for gpu_id in gpu_ids:
        if gpu_id >= total_gpus:
            raise ValueError(f"指定的GPU ID {gpu_id} 超出可用范围 (0-{total_gpus-1})")
    args.gpu_ids = gpu_ids
    world_size = len(gpu_ids)
    print(f"使用指定的GPU: {gpu_ids}")
    
    # 记录训练开始信息
    print(f"开始启动多进程训练，世界大小: {world_size}")
    print(f"输出目录: {args.output_dir}")
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()