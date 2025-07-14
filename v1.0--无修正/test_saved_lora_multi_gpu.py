import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import argparse
from tqdm import tqdm
from qwen3_classification_direct import Qwen3ForSequenceClassification
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
        # 根据文件扩展名选择读取方式
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            self.data = pd.read_excel(data_path)
        else:
            # 尝试两种格式
            try:
                self.data = pd.read_excel(data_path)
            except Exception as e1:
                try:
                    self.data = pd.read_csv(data_path, encoding='utf-8-sig')
                except Exception as e2:
                    raise ValueError(f"无法读取文件 {data_path}。Excel错误: {e1}，CSV错误: {e2}")
        
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
            'texts': text  # 添加原始文本
        }

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12366'  # 使用不同的端口避免冲突
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

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

# 自定义collate函数以处理文本数据
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    texts = [item['texts'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'texts': texts
    }

# 评估函数 - 多GPU版本，包含漏报误报分析
def evaluate(model, dataloader, device, rank, world_size, use_fp16=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_texts = []
    total_loss = 0
    
    # 获取模型数据类型
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['texts']  # 获取原始文本
            
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
            all_texts.extend(texts)
    
    # 在多GPU之间收集所有结果
    if world_size > 1:
        all_preds, all_labels, all_texts = gather_results(all_preds, all_labels, all_texts, world_size, device)
    
    # 计算平均loss（需要考虑不同GPU上的batch数量可能不同）
    loss_tensor = torch.tensor([total_loss, len(dataloader)], device=device)
    if world_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor[0].item() / loss_tensor[1].item()
    
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
            'texts': all_texts,
            'per_class_accuracy': per_class_accuracy,
            'per_class_precision': precision_per_class,
            'per_class_recall': recall_per_class,
            'per_class_f1': f1_per_class,
            'per_class_support': support_per_class,
            # 漏报误报分析结果
            'per_class_miss_rate': per_class_miss_rate,           # 每类的漏报率（被误判为正常的比例）
            'total_miss_rate': total_miss_rate,                   # 总漏报率（所有异常->正常）
            'total_miss_count': int(total_miss_to_normal_count),  # 总漏报数量
            'total_abnormal_count': int(total_abnormal_count),    # 总异常样本数
            'normal_false_positive_rate': normal_false_positive_rate,       # 正常的误报率
            'normal_false_positive_count': int(normal_false_positive_count), # 正常误报数量
            'normal_total_count': int(normal_count),              # 正常样本总数
            'normal_misclassified_to': normal_misclassified_to,   # 正常被误判为各异常类别的详情
            'binary_accuracy': binary_accuracy,                  # 二分类准确率
            'binary_confusion_matrix': {
                'true_negative': int(tn),   # 正常->正常
                'false_positive': int(fp),  # 正常->异常
                'false_negative': int(fn),  # 异常->正常
                'true_positive': int(tp)    # 异常->异常
            }
        }
    else:
        # 非主进程返回简单结果
        return {'loss': avg_loss}

def test_worker(rank, world_size, args):
    """每个GPU上的测试工作进程"""
    try:
        # 获取实际的GPU ID
        actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
        
        # 设置当前GPU - 这需要在分布式初始化之前
        torch.cuda.set_device(actual_gpu_id)
        
        # 设置分布式环境
        setup_distributed(rank, world_size)
        
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
            
            # 记录GPU内存信息
            if torch.cuda.is_available():
                for i, gpu_id in enumerate(args.gpu_ids):
                    if gpu_id < torch.cuda.device_count():
                        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                        print(f"GPU {gpu_id}: {memory_total:.2f}GB total memory")
        
        # 在分布式同步点确保所有进程都到达这里
        dist.barrier()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 只在主进程打印加载信息
        if is_main_process:
            print("加载基础模型...")
        
        # 确保所有进程同步加载
        dist.barrier()
        
        # 加载基础模型到当前设备
        base_model = Qwen3ForSequenceClassification(args.checkpoint, num_labels=6)
        base_model.to(device)
        
        # 同步所有进程
        dist.barrier()
        
        if is_main_process:
            print(f"加载LoRA模型: {args.lora_model}")
        
        # 加载LoRA模型
        model = PeftModel.from_pretrained(base_model, args.lora_model)
        model.to(device)
        
        # 同步所有进程
        dist.barrier()
        
        # 包装为DDP模型
        model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
        model.eval()
        
        if is_main_process:
            print("LoRA模型加载完成！")
            print(f"模型已加载到多个GPU: {args.gpu_ids}")
        
        # 准备测试数据集
        if is_main_process:
            print("加载测试数据集...")
        
        test_dataset = ClassificationDataset(args.test_data, tokenizer, args.max_length)
        
        if is_main_process:
            print(f"测试集样本数: {len(test_dataset)}")
        
        # 创建分布式采样器
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        
        # 减少num_workers以避免多进程冲突
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=1,  # 减少worker数量避免冲突
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
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
            if test_results['normal_misclassified_to']:
                print("正常被误判为各异常类的详情:")
                for class_name, info in test_results['normal_misclassified_to'].items():
                    if info['count'] > 0:
                        print(f"  -> {class_name}: {info['rate']:.4f} ({info['count']}条)")
            
            print("\n=== 【异常类漏报分析】 ===")
            print(f"总漏报率(异常->正常): {test_results['total_miss_rate']:.4f} "
                  f"({test_results['total_miss_count']}/{test_results['total_abnormal_count']})")
            print("各异常类漏报率(被误判为正常):")
            for class_name, info in test_results['per_class_miss_rate'].items():
                print(f"  {class_name}: {info['rate']:.4f} ({info['count']}/{info['total']})")
            
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
            
            # 打印混淆矩阵
            cm = confusion_matrix(test_results['labels'], test_results['predictions'])
            print("\n=== 混淆矩阵 ===")
            print(cm)
            
            # 保存结果
            output_dir = os.path.dirname(args.lora_model)
            # 添加时间戳到文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f'multi_gpu_test_results_with_analysis_{timestamp}.json')
            
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
                        'timestamp': timestamp
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
                        # 漏报误报分析结果
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
            
            # 保存详细预测结果到CSV文件
            if 'texts' in test_results:
                # 创建标签映射的反向字典
                label_map_reverse = {
                    0: '正常',
                    1: '歧视',
                    2: '违法违规',
                    3: '政治安全',
                    4: '暴恐',
                    5: '色情低俗'
                }
                
                # 创建详细结果的DataFrame
                detailed_results = pd.DataFrame({
                    '原始文本': test_results['texts'],
                    '原始标签': [label_map_reverse.get(label, str(label)) for label in test_results['labels']],
                    '预测标签': [label_map_reverse.get(pred, str(pred)) for pred in test_results['predictions']],
                    '原始标签_数字': test_results['labels'],
                    '预测标签_数字': test_results['predictions'],
                    '预测正确': [1 if pred == label else 0 for pred, label in zip(test_results['predictions'], test_results['labels'])]
                })
                
                # 保存到CSV文件
                csv_file = os.path.join(output_dir, f'detailed_predictions_{timestamp}.csv')
                detailed_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"详细预测结果已保存到: {csv_file}")
                print(f"CSV文件包含 {len(detailed_results)} 条记录")
            
            # 关键业务指标总结
            print("\n=== 关键业务指标总结 ===")
            print(f"✓ 正常内容误报率: {test_results['normal_false_positive_rate']:.4f}")
            print(f"✓ 异常内容总漏报率: {test_results['total_miss_rate']:.4f}")
            print(f"✓ 二分类准确率(正常vs异常): {test_results['binary_accuracy']:.4f}")
            print(f"✓ 使用了 {world_size} 个GPU: {args.gpu_ids}")
        
        # 确保所有进程都完成评估后再清理
        dist.barrier()
        
        # 清理分布式环境
        cleanup_distributed()
        
    except Exception as e:
        # 处理异常情况
        print(f"进程 {rank} 发生错误: {e}")
        if rank == 0:
            print(f"主进程错误详情: {str(e)}")
        # 尝试清理资源
        try:
            if 'dist' in globals() and dist.is_initialized():
                cleanup_distributed()
        except:
            pass
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                        help="原始模型的路径")
    parser.add_argument('--lora_model', type=str, required=False, default="../lora-202507141343/best_model",
                        help="保存的LoRA模型路径")
    parser.add_argument('--test_data', type=str, required=False, default="../data/test3.csv",
                        help="测试数据集路径")
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--max_length', type=int, default=256)
    
    # 多GPU配置
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5,6,7",
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
        print("警告: 只使用1个GPU，建议使用单GPU测试脚本")
    
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