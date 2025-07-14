import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import PeftModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import argparse
from tqdm import tqdm
from qwen3_simple_model import Qwen3ForClassification
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import json
import gc  # 导入垃圾回收模块

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

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'  # 使用不同的端口避免冲突
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def gather_results(all_preds, all_labels, world_size, device):
    """在所有GPU之间收集预测结果和标签 - 内存优化版本"""
    try:
        # 将列表转换为numpy数组先，减少GPU内存使用
        preds_np = np.array(all_preds, dtype=np.int64)
        labels_np = np.array(all_labels, dtype=np.int64)
        
        # 创建小的张量来收集大小信息
        local_size = torch.tensor([len(all_preds)], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        
        # 转换为CPU并计算最大大小
        sizes = [s.cpu().item() for s in size_list]
        max_size = max(sizes)
        
        # 在CPU上准备数据，减少GPU内存压力
        if len(preds_np) < max_size:
            padding_size = max_size - len(preds_np)
            preds_padded = np.concatenate([preds_np, np.full(padding_size, -1, dtype=np.int64)])
            labels_padded = np.concatenate([labels_np, np.full(padding_size, -1, dtype=np.int64)])
        else:
            preds_padded = preds_np
            labels_padded = labels_np
        
        # 转移到GPU进行收集（一次性操作）
        preds_tensor = torch.from_numpy(preds_padded).to(device, non_blocking=True)
        labels_tensor = torch.from_numpy(labels_padded).to(device, non_blocking=True)
        
        # 收集所有GPU的结果
        gathered_preds = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(world_size)]
        gathered_labels = [torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(world_size)]
        
        dist.all_gather(gathered_preds, preds_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
        
        # 立即清理本地张量
        del preds_tensor, labels_tensor
        torch.cuda.empty_cache()
        
        # 合并结果并去除padding（在CPU上操作）
        final_preds = []
        final_labels = []
        
        for i, (preds, labels, size) in enumerate(zip(gathered_preds, gathered_labels, sizes)):
            valid_preds = preds[:size].cpu().numpy()
            valid_labels = labels[:size].cpu().numpy()
            final_preds.extend(valid_preds)
            final_labels.extend(valid_labels)
            
            # 立即清理
            del preds, labels
        
        # 清理收集的张量
        del gathered_preds, gathered_labels
        torch.cuda.empty_cache()
        
        return final_preds, final_labels
        
    except Exception as e:
        print(f"Error in gather_results: {e}")
        # 如果收集失败，至少返回本地结果
        return all_preds, all_labels

# 评估函数 - 多GPU版本，包含漏报误报分析 - 内存优化版本
def evaluate(model, dataloader, device, rank, world_size, use_fp16=True):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    batch_count = 0
    
    # 获取模型数据类型
    model_dtype = next(model.parameters()).dtype
    
    # 减少显存使用的设置
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", disable=(rank != 0))):
            try:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
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
                batch_count += 1
                
                # 立即转移到CPU并转换为numpy，减少GPU内存占用
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                
                # 立即清理GPU上的张量
                del input_ids, attention_mask, labels, outputs, loss, logits, preds
                
                # 每10个batch清理一次缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU {rank}: 内存不足在batch {batch_idx}, 跳过此batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
    
    # 最终清理
    torch.cuda.empty_cache()
    gc.collect()
    
    # 在多GPU之间收集所有结果（如果有的话）
    if world_size > 1 and len(all_preds) > 0:
        all_preds, all_labels = gather_results(all_preds, all_labels, world_size, device)
    
    # 计算平均loss（需要考虑不同GPU上的batch数量可能不同）
    if world_size > 1:
        loss_tensor = torch.tensor([total_loss, batch_count], device=device, dtype=torch.float32)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0)  # 避免除零
        del loss_tensor
    else:
        # 单GPU情况，直接计算
        avg_loss = total_loss / max(batch_count, 1.0)
    
    torch.cuda.empty_cache()
    
    # 只在主进程计算指标（因为需要完整的数据集结果）
    if rank == 0:
        # 计算基础指标
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
        
        # ===== 漏报和误报分析 =====
        # 标签定义：0=正常，1=歧视，2=违法违规，3=政治安全，4=暴恐，5=色情低俗
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
            # 漏报误报分析结果
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
    else:
        # 非主进程返回简单结果
        return {'loss': avg_loss}

def test_worker(rank, world_size, args):
    """每个GPU上的测试工作进程"""
    # 获取实际的GPU ID
    actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
    
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(actual_gpu_id)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f'cuda:{actual_gpu_id}')
    
    # 清理GPU缓存
    if args.clear_cache:
        torch.cuda.empty_cache()
    
    # 只在主进程打印信息
    is_main_process = rank == 0
    
    if is_main_process:
        print(f"使用设备: {device}")
        print(f"世界大小: {world_size}")
        print(f"当前排名: {rank}")
        print(f"开始多GPU测试，使用 {world_size} 个GPU: {args.gpu_ids}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    if is_main_process:
        print("加载简化模型...")
    
    base_model = Qwen3ForClassification(args.checkpoint)
    
    # 直接从保存的LoRA模型加载
    if is_main_process:
        print(f"加载LoRA模型: {args.lora_model}")
    
    model = PeftModel.from_pretrained(base_model.model, args.lora_model)
    base_model.model = model
    
    # 将模型移到设备并包装为DDP
    base_model.to(device)
    model = DDP(base_model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    model.eval()
    
    if is_main_process:
        print("简化LoRA模型加载完成！")
    
    # 准备测试数据集
    if is_main_process:
        print("加载测试数据集...")
    
    test_dataset = ClassificationDataset(args.test_data, tokenizer, args.max_length)
    
    if is_main_process:
        print(f"测试集样本数: {len(test_dataset)}")
    
    # 创建分布式采样器
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=2,  # 减少worker数量
        pin_memory=False,  # 关闭pin_memory减少内存使用
        drop_last=False,
        persistent_workers=False  # 不保持worker进程
    )
    
    # 确保所有进程都完成初始化
    dist.barrier()
    
    # 评估模型
    if is_main_process:
        print("\n开始多GPU评估...")
    
    test_results = evaluate(model, test_loader, device, rank, world_size, use_fp16=args.fp16)
    
    # 只在主进程打印和保存结果
    if is_main_process and test_results and 'accuracy' in test_results:
        # 打印基础结果
        print("\n=== 多GPU测试集结果 ===")
        print(f"Loss: {test_results['loss']:.4f}")
        print(f"6分类准确率: {test_results['accuracy']:.4f}")
        print(f"F1: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"二分类准确率: {test_results['binary_accuracy']:.4f}")
        
        # 打印漏报和误报分析
        print("\n=== 【正常类误报分析】 ===")
        print(f"正常内容误报率: {test_results['normal_false_positive_rate']:.4f} "
              f"({test_results['normal_false_positive_count']}/{test_results['normal_total_count']})")
        
        print("\n=== 【异常类漏报分析】 ===")
        print(f"总漏报率(异常->正常): {test_results['total_miss_rate']:.4f} "
              f"({test_results['total_miss_count']}/{test_results['total_abnormal_count']})")
        
        # 定义标签名称
        label_names = ['正常', '歧视', '违法违规', '政治安全', '暴恐', '色情低俗']
        
        # 打印每个类别的详细指标
        print("\n=== 每个类别的详细指标 ===")
        print(f"{'类别':<10} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<8}")
        print("-" * 60)
        for i, label_name in enumerate(label_names):
            if i < len(test_results['per_class_accuracy']):
                acc = test_results['per_class_accuracy'][i]
                prec = test_results['per_class_precision'][i]
                rec = test_results['per_class_recall'][i]
                f1 = test_results['per_class_f1'][i]
                sup = test_results['per_class_support'][i]
                print(f"{label_name:<10} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {sup:<8}")
        
        # 保存结果
        output_dir = os.path.dirname(args.lora_model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'simple_multi_gpu_test_results_{timestamp}.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': {
                    'num_gpus': world_size,
                    'gpu_ids': args.gpu_ids,
                    'model_path': args.lora_model,
                    'test_data': args.test_data,
                    'batch_size': args.batch_size,
                    'max_length': args.max_length,
                    'fp16': args.fp16,
                    'timestamp': timestamp,
                    'model_type': 'simple_qwen3_with_lora'
                },
                'results': {
                    'loss': test_results['loss'],
                    'accuracy': test_results['accuracy'],
                    'f1': test_results['f1'],
                    'precision': test_results['precision'],
                    'recall': test_results['recall'],
                    'binary_accuracy': test_results['binary_accuracy'],
                    'per_class_metrics': {
                        label_names[i]: {
                            'accuracy': test_results['per_class_accuracy'][i],
                            'precision': test_results['per_class_precision'][i],
                            'recall': test_results['per_class_recall'][i],
                            'f1': test_results['per_class_f1'][i],
                            'support': int(test_results['per_class_support'][i])
                        } for i in range(len(label_names))
                    },
                    'miss_report_analysis': {
                        'total_miss_rate': test_results['total_miss_rate'],
                        'total_miss_count': test_results['total_miss_count'],
                        'total_abnormal_count': test_results['total_abnormal_count'],
                        'per_class_miss_rate': test_results['per_class_miss_rate']
                    },
                    'false_positive_analysis': {
                        'normal_false_positive_rate': test_results['normal_false_positive_rate'],
                        'normal_false_positive_count': test_results['normal_false_positive_count'],
                        'normal_total_count': test_results['normal_total_count'],
                        'normal_misclassified_to': test_results['normal_misclassified_to']
                    },
                    'binary_confusion_matrix': test_results['binary_confusion_matrix']
                }
            }, f, indent=4, ensure_ascii=False)
        
        print(f"\n多GPU测试结果已保存到: {results_file}")
        
        # 关键业务指标总结
        print("\n=== 关键业务指标总结 ===")
        print(f"✓ 正常内容误报率: {test_results['normal_false_positive_rate']:.4f}")
        print(f"✓ 异常内容总漏报率: {test_results['total_miss_rate']:.4f}")
        print(f"✓ 二分类准确率(正常vs异常): {test_results['binary_accuracy']:.4f}")
        print(f"✓ 使用了 {world_size} 个GPU: {args.gpu_ids}")
        print(f"✓ 模型类型: 简化Qwen3 + LoRA")
    
    # 确保所有进程都完成评估后再清理
    dist.barrier()
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                        help="原始模型的路径")
    parser.add_argument('--lora_model', type=str, required=False, default="../simple-model-output/best_model",
                        help="保存的LoRA模型路径")
    parser.add_argument('--test_data', type=str, required=False, default="../data/r789-b-50000.xlsx",
                        help="测试数据集路径")
    parser.add_argument('--batch_size', type=int, default=2)  # 降低默认批次大小
    parser.add_argument('--max_length', type=int, default=256)
    
    # 多GPU配置
    parser.add_argument('--gpu_ids', type=str, default="1,2,3,4",
                        help='指定使用的GPU ID，用逗号分隔，例如: 0,1,2 或 0,2,4')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='指定使用的GPU数量，从GPU 0开始使用')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='使用混合精度推理')
    parser.add_argument('--clear_cache', action='store_true', default=True,
                        help='测试前清理GPU缓存')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行多GPU测试")
    
    # 处理GPU配置
    total_gpus = torch.cuda.device_count()
    print(f"系统检测到 {total_gpus} 个GPU")
    
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
        print("注意: 使用1个GPU运行多GPU代码，这是完全正常的")
    
    # 记录测试开始信息
    print(f"开始启动多进程测试，世界大小: {world_size}")
    print(f"LoRA模型路径: {args.lora_model}")
    print(f"测试数据路径: {args.test_data}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.max_length}")
    print(f"使用混合精度: {args.fp16}")
    
    # 启动多进程测试
    mp.spawn(test_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()