#!/usr/bin/env python3
"""
指令微调训练脚本 - 基于LoRA的文本分类指令微调
支持多GPU分布式训练
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class InstructionDataset(Dataset):
    """指令微调数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 读取JSONL文件
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效的JSON行: {line[:100]}... 错误: {e}")
        
        logger.info(f"成功加载 {len(self.data)} 条指令数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取指令和输出
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # 构建完整的文本：指令 + 输出
        # 使用特殊token分隔指令和回答
        full_text = f"{instruction}/no_think\n\n{output}"
        
        # 对文本进行编码
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 创建标签（用于语言模型训练，标签就是input_ids）
        labels = encoding['input_ids'].clone()
        
        # 找到"/no_think"的位置，只对输出部分计算loss
        try:
            # 编码"/no_think"来找到分割点
            separator_tokens = self.tokenizer.encode("/no_think", add_special_tokens=False)
            input_ids = encoding['input_ids'].squeeze()
            
            # 找到分隔符的位置
            separator_pos = -1
            for i in range(len(input_ids) - len(separator_tokens) + 1):
                if torch.equal(input_ids[i:i+len(separator_tokens)], torch.tensor(separator_tokens)):
                    separator_pos = i + len(separator_tokens)
                    break
            
            # 如果找到分隔符，则只对输出部分计算loss
            if separator_pos > 0:
                labels[:separator_pos] = -100  # 忽略指令部分的loss
        except Exception as e:
            logger.warning(f"处理样本 {idx} 时出错: {e}")
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    # 对于生成任务，这里可以根据需要自定义指标
    # 暂时返回简单的loss指标
    return {}

class InstructionTrainer(Trainer):
    """自定义训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失函数"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # 获取logits
        logits = outputs.get('logits')
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 将logits和labels展平
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算损失
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train_worker(rank, world_size, args):
    """每个GPU上的训练工作进程"""
    try:
        # 获取实际的GPU ID
        actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
        
        # 设置当前GPU
        torch.cuda.set_device(actual_gpu_id)
        
        # 设置分布式环境
        setup_distributed(rank, world_size)
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 设置设备
        device = torch.device(f'cuda:{actual_gpu_id}')
        
        # 只在主进程打印信息
        is_main_process = rank == 0
        
        if is_main_process:
            logger.info(f"使用设备: {device}")
            logger.info(f"世界大小: {world_size}")
            logger.info(f"当前排名: {rank}")
            logger.info(f"开始多GPU训练，使用 {world_size} 个GPU: {args.gpu_ids}")
        
        # 在分布式同步点确保所有进程都到达这里
        dist.barrier()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if is_main_process:
            logger.info("加载基础模型...")
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            device_map=None,  # 手动管理设备
            trust_remote_code=True
        )
        model.to(device)
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        
        if is_main_process:
            logger.info("LoRA配置完成")
            model.print_trainable_parameters()
        
        # 准备数据集
        train_dataset = InstructionDataset(args.train_data, tokenizer, args.max_length)
        val_dataset = InstructionDataset(args.val_data, tokenizer, args.max_length) if args.val_data else None
        
        if is_main_process:
            logger.info(f"训练集样本数: {len(train_dataset)}")
            if val_dataset:
                logger.info(f"验证集样本数: {len(val_dataset)}")
        
        # 创建输出目录
        output_dir = args.output_dir
        if is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            fp16=args.fp16,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to=None,  # 禁用wandb等
            local_rank=rank,
            ddp_find_unused_parameters=False,
        )
        
        # 创建训练器
        trainer = InstructionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # 开始训练
        if is_main_process:
            logger.info("开始训练...")
        
        trainer.train()
        
        # 保存最终模型
        if is_main_process:
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            logger.info(f"最终模型已保存到: {final_model_path}")
        
        # 清理分布式环境
        cleanup_distributed()
        
    except Exception as e:
        logger.error(f"进程 {rank} 发生错误: {e}")
        if rank == 0:
            logger.error(f"主进程错误详情: {str(e)}")
        try:
            if 'dist' in globals() and dist.is_initialized():
                cleanup_distributed()
        except:
            pass
        raise e

def main():
    parser = argparse.ArgumentParser(description='指令微调训练脚本')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, required=True,
                        help='预训练模型名称或路径')
    parser.add_argument('--train_data', type=str, required=True,
                        help='训练数据JSONL文件路径')
    parser.add_argument('--val_data', type=str, default=None,
                        help='验证数据JSONL文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='预热步数')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    
    # 训练控制参数
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='日志记录步数')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='模型保存步数')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='评估步数')
    
    # 多GPU配置
    parser.add_argument('--gpu_ids', type=str, default="0",
                        help='指定使用的GPU ID，用逗号分隔，例如: 0,1,2')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='使用混合精度训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行GPU训练")
    
    # 处理GPU配置
    total_gpus = torch.cuda.device_count()
    logger.info(f"系统检测到 {total_gpus} 个GPU")
    
    # 解析GPU配置
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    # 验证GPU ID的有效性
    for gpu_id in gpu_ids:
        if gpu_id >= total_gpus:
            raise ValueError(f"指定的GPU ID {gpu_id} 超出可用范围 (0-{total_gpus-1})")
    
    args.gpu_ids = gpu_ids
    world_size = len(gpu_ids)
    
    logger.info(f"使用GPU: {gpu_ids}")
    logger.info(f"训练数据: {args.train_data}")
    logger.info(f"验证数据: {args.val_data}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练轮数: {args.num_epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    
    # 启动多进程训练
    if world_size > 1:
        logger.info(f"启动多GPU训练，世界大小: {world_size}")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        logger.info("启动单GPU训练")
        train_worker(0, 1, args)

if __name__ == "__main__":
    main()