import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import logging
from tqdm import tqdm
import argparse

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 自定义数据集类
class ClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
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
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            outputs = model(**batch)
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument('--checkpoint', type=str, default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B")
    parser.add_argument('--train_data', type=str, default="zzz/data/balanced_train.xlsx")
    parser.add_argument('--val_data', type=str, default="zzz/data/balanced_val.xlsx")
    parser.add_argument('--output_dir', type=str, default="zzz/efficient-output")
    
    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    logger.info("正在加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=6,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 获取可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 将模型移到设备
    model.to(device)
    
    # 准备数据集
    logger.info("加载数据集...")
    train_dataset = ClassificationDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ClassificationDataset(args.val_data, tokenizer, args.max_length)
    
    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 计算总训练步数
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    
    # 设置学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # 训练循环
    best_val_f1 = 0
    global_step = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"===== 开始第 {epoch + 1}/{args.num_epochs} 轮训练 =====")
        
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            # 将数据移到设备
            batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            
            # 混合精度训练
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # 定期记录训练损失
            if (step + 1) % (args.gradient_accumulation_steps * 50) == 0:
                avg_loss = train_loss / (step + 1)
                logger.info(f"Epoch {epoch + 1} Step {global_step} - Loss: {avg_loss:.4f}")
        
        # 每个epoch结束后评估
        logger.info("评估验证集...")
        val_results = evaluate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch + 1} 验证集结果:")
        logger.info(f"  Loss: {val_results['loss']:.4f}")
        logger.info(f"  准确率: {val_results['accuracy']:.4f}")
        logger.info(f"  F1: {val_results['f1']:.4f}")
        
        # 保存最佳模型
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
            logger.info(f"保存最佳模型，F1: {val_results['f1']:.4f}")
        
        # 每个epoch结束后保存模型
        model.save_pretrained(os.path.join(args.output_dir, f'epoch-{epoch + 1}'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, f'epoch-{epoch + 1}'))
        logger.info(f"已保存第 {epoch + 1} 个epoch的模型")
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main()