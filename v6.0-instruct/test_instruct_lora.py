#!/usr/bin/env python3
"""
指令微调测试脚本 - 测试LoRA微调后的指令模型
支持多GPU分布式推理和准确率评估
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import logging
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class InstructionTestDataset(Dataset):
    """指令微调测试数据集类"""
    
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
        
        logger.info(f"成功加载 {len(self.data)} 条测试数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取指令和真实输出
        instruction = item.get('instruction', '')
        true_output = item.get('output', '')
        
        # 构建输入文本（只包含指令部分）
        input_text = f"{instruction}/no_think"
        
        # 对输入文本进行编码
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'instruction': instruction,
            'true_output': true_output,
            'input_text': input_text
        }

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def extract_answer_from_output(output_text):
    """从模型输出中提取答案"""
    # 查找</think>后的内容
    think_pattern = r'</think>\s*\n*\s*([A-F])'
    match = re.search(think_pattern, output_text)
    if match:
        return match.group(1)
    
    # 如果没找到</think>，直接查找最后的字母答案
    letter_pattern = r'\b([A-F])\s*$'
    match = re.search(letter_pattern, output_text.strip())
    if match:
        return match.group(1)
    
    # 查找任何A-F字母
    letters = re.findall(r'\b([A-F])\b', output_text)
    if letters:
        return letters[-1]  # 返回最后一个找到的字母
    
    return None

def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens=50, device=None):
    """生成模型回复"""
    model.eval()
    
    with torch.no_grad():
        # 生成回复
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪心解码
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

def gather_results(all_preds, all_labels, all_texts, world_size, device):
    """在所有GPU之间收集预测结果、标签和文本"""
    # 将列表转换为张量
    preds_tensor = torch.tensor(all_preds, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    # 收集每个进程的数据长度
    local_size = torch.tensor([len(all_preds)], dtype=torch.long, device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # 计算最大大小，用于padding
    max_size = max([s.item() for s in size_list])
    
    # Padding到相同长度
    if len(all_preds) < max_size:
        padding_size = max_size - len(all_preds)
        preds_padding = torch.full((padding_size,), -1, dtype=torch.long, device=device)
        labels_padding = torch.full((padding_size,), -1, dtype=torch.long, device=device)
        preds_tensor = torch.cat([preds_tensor, preds_padding])
        labels_tensor = torch.cat([labels_tensor, labels_padding])
        # 对文本进行padding
        all_texts.extend([''] * padding_size)
    
    # 收集所有GPU的结果
    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)
    
    # 收集文本数据（使用all_gather_object）
    gathered_texts = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_texts, all_texts)
    
    # 合并结果并去除padding
    final_preds = []
    final_labels = []
    final_texts = []
    
    for i, (preds, labels, texts, size) in enumerate(zip(gathered_preds, gathered_labels, gathered_texts, size_list)):
        valid_preds = preds[:size.item()].cpu().numpy()
        valid_labels = labels[:size.item()].cpu().numpy()
        valid_texts = texts[:size.item()]
        final_preds.extend(valid_preds)
        final_labels.extend(valid_labels)
        final_texts.extend(valid_texts)
    
    return final_preds, final_labels, final_texts

def evaluate_model(model, dataloader, tokenizer, device, rank, world_size, max_new_tokens=50):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_generated_texts = []
    all_instructions = []
    
    # 标签映射
    label_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_outputs = batch['true_output']
            instructions = batch['instruction']
            
            batch_size = input_ids.size(0)
            
            for i in range(batch_size):
                # 生成回复
                generated_text = generate_response(
                    model, tokenizer, 
                    input_ids[i:i+1], 
                    attention_mask[i:i+1], 
                    max_new_tokens, 
                    device
                )
                
                # 提取真实标签
                true_output = true_outputs[i]
                true_answer = extract_answer_from_output(true_output)
                
                # 提取预测标签
                pred_answer = extract_answer_from_output(generated_text)
                
                # 转换为数字标签
                true_label = label_to_num.get(true_answer, -1) if true_answer else -1
                pred_label = label_to_num.get(pred_answer, -1) if pred_answer else -1
                
                # 只保留有效的预测
                if true_label != -1:
                    all_true_labels.append(true_label)
                    all_predictions.append(pred_label if pred_label != -1 else 0)  # 无效预测默认为A
                    all_generated_texts.append(generated_text)
                    all_instructions.append(instructions[i])
    
    # 在多GPU之间收集所有结果
    if world_size > 1:
        all_predictions, all_true_labels, all_generated_texts = gather_results(
            all_predictions, all_true_labels, all_generated_texts, world_size, device
        )
    
    return all_predictions, all_true_labels, all_generated_texts, all_instructions

def test_worker(rank, world_size, args):
    """每个GPU上的测试工作进程"""
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
            logger.info(f"开始多GPU测试，使用 {world_size} 个GPU: {args.gpu_ids}")
        
        # 在分布式同步点确保所有进程都到达这里
        dist.barrier()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if is_main_process:
            logger.info("加载基础模型...")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        base_model.to(device)
        
        # 同步所有进程
        dist.barrier()
        
        if is_main_process:
            logger.info(f"加载LoRA模型: {args.lora_model}")
        
        # 加载LoRA模型
        model = PeftModel.from_pretrained(base_model, args.lora_model)
        model.to(device)
        
        # 同步所有进程
        dist.barrier()
        
        # 包装为DDP模型
        model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
        model.eval()
        
        if is_main_process:
            logger.info("LoRA模型加载完成！")
        
        # 准备测试数据集
        if is_main_process:
            logger.info("加载测试数据集...")
        
        test_dataset = InstructionTestDataset(args.test_data, tokenizer, args.max_length)
        
        if is_main_process:
            logger.info(f"测试集样本数: {len(test_dataset)}")
        
        # 创建分布式采样器
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=1,
            pin_memory=True,
            drop_last=False
        )
        
        # 确保所有进程都完成初始化
        dist.barrier()
        
        # 评估模型
        if is_main_process:
            logger.info("\n开始多GPU评估...")
        
        all_predictions, all_true_labels, all_generated_texts, all_instructions = evaluate_model(
            model, test_loader, tokenizer, device, rank, world_size, args.max_new_tokens
        )
        
        # 只在主进程计算和打印结果
        if is_main_process and len(all_predictions) > 0:
            # 计算基础指标
            accuracy = accuracy_score(all_true_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_labels, all_predictions, average='weighted'
            )
            
            # 计算每个类别的指标
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                all_true_labels, all_predictions, average=None
            )
            
            # 计算混淆矩阵
            cm = confusion_matrix(all_true_labels, all_predictions)
            
            # 标签名称
            label_names = ['正常(A)', '政治安全(B)', '歧视(C)', '违法违规(D)', '色情低俗(E)', '暴恐(F)']
            
            # 打印结果
            print("\n=== 指令微调模型测试结果 ===")
            print(f"总样本数: {len(all_predictions)}")
            print(f"准确率: {accuracy:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            
            # 打印每个类别的详细指标
            print("\n=== 每个类别的详细指标 ===")
            print(f"{'类别':<15} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<8}")
            print("-" * 60)
            for i, label_name in enumerate(label_names):
                if i < len(precision_per_class):
                    prec = precision_per_class[i]
                    rec = recall_per_class[i]
                    f1_score = f1_per_class[i]
                    sup = support_per_class[i]
                    print(f"{label_name:<15} {prec:<8.4f} {rec:<8.4f} {f1_score:<8.4f} {sup:<8}")
            
            # 打印混淆矩阵
            print("\n=== 混淆矩阵 ===")
            print("行：真实标签，列：预测标签")
            print("标签顺序：A(正常), B(政治安全), C(歧视), D(违法违规), E(色情低俗), F(暴恐)")
            print(cm)
            
            # 保存详细结果
            output_dir = os.path.dirname(args.lora_model)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存JSON结果
            results_file = os.path.join(output_dir, f'instruct_test_results_{timestamp}.json')
            results_data = {
                'test_config': {
                    'num_gpus': world_size,
                    'gpu_ids': args.gpu_ids,
                    'base_model': args.base_model,
                    'lora_model': args.lora_model,
                    'test_data': args.test_data,
                    'batch_size': args.batch_size,
                    'max_length': args.max_length,
                    'max_new_tokens': args.max_new_tokens,
                    'fp16': args.fp16,
                    'timestamp': timestamp
                },
                'results': {
                    'total_samples': len(all_predictions),
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'per_class_metrics': {
                        label_names[i]: {
                            'precision': precision_per_class[i],
                            'recall': recall_per_class[i],
                            'f1': f1_per_class[i],
                            'support': int(support_per_class[i])
                        } for i in range(len(label_names)) if i < len(precision_per_class)
                    },
                    'confusion_matrix': cm.tolist()
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"\n测试结果已保存到: {results_file}")
            
            # 保存详细预测结果到CSV
            if len(all_generated_texts) > 0:
                # 标签映射的反向字典
                num_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
                
                # 创建详细结果的DataFrame
                detailed_results = pd.DataFrame({
                    '指令': all_instructions[:len(all_predictions)],
                    '真实标签': [num_to_label.get(label, str(label)) for label in all_true_labels],
                    '预测标签': [num_to_label.get(pred, str(pred)) for pred in all_predictions],
                    '预测正确': [1 if pred == label else 0 for pred, label in zip(all_predictions, all_true_labels)],
                    '生成文本': all_generated_texts[:len(all_predictions)]
                })
                
                # 保存到CSV文件
                csv_file = os.path.join(output_dir, f'detailed_instruct_predictions_{timestamp}.csv')
                detailed_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
                logger.info(f"详细预测结果已保存到: {csv_file}")
                logger.info(f"CSV文件包含 {len(detailed_results)} 条记录")
            
            # 关键指标总结
            print("\n=== 关键指标总结 ===")
            print(f"✓ 指令微调模型准确率: {accuracy:.4f}")
            print(f"✓ 加权F1分数: {f1:.4f}")
            print(f"✓ 使用了 {world_size} 个GPU: {args.gpu_ids}")
        
        # 确保所有进程都完成评估后再清理
        dist.barrier()
        
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
    parser = argparse.ArgumentParser(description='指令微调模型测试脚本')
    
    # 模型参数
    parser.add_argument('--base_model', type=str, required=True,
                        help='基础模型路径')
    parser.add_argument('--lora_model', type=str, required=True,
                        help='LoRA模型路径')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据JSONL文件路径')
    
    # 推理参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                        help='输入最大序列长度')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='生成的最大新token数')
    
    # 多GPU配置
    parser.add_argument('--gpu_ids', type=str, default="0",
                        help='指定使用的GPU ID，用逗号分隔，例如: 0,1,2')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='使用混合精度推理')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行GPU推理")
    
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
    logger.info(f"基础模型: {args.base_model}")
    logger.info(f"LoRA模型: {args.lora_model}")
    logger.info(f"测试数据: {args.test_data}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"最大生成长度: {args.max_new_tokens}")
    
    # 启动多进程测试
    if world_size > 1:
        logger.info(f"启动多GPU测试，世界大小: {world_size}")
        mp.spawn(test_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        logger.info("启动单GPU测试")
        test_worker(0, 1, args)

if __name__ == "__main__":
    main()