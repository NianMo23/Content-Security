import torch
import torch.nn as nn
import torch.nn.functional as F
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

# =========================== 二分类模型定义 ===========================
class BinaryQwen3ForSequenceClassification(nn.Module):
    """
    二分类版本的Qwen3分类模型（用于测试加载）
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
        logits = self.base_model.lm_head(pooled_output)
        
        # 计算损失（测试时通常不需要）
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

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'  # 使用不同的端口避免冲突
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def gather_results(all_preds, all_labels, world_size, device):
    """在所有GPU之间收集预测结果和标签"""
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
    
    # 收集所有GPU的结果
    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)
    
    # 合并结果并去除padding
    final_preds = []
    final_labels = []
    
    for i, (preds, labels, size) in enumerate(zip(gathered_preds, gathered_labels, size_list)):
        valid_preds = preds[:size.item()].cpu().numpy()
        valid_labels = labels[:size.item()].cpu().numpy()
        final_preds.extend(valid_preds)
        final_labels.extend(valid_labels)
    
    return final_preds, final_labels

# 二分类评估函数 - 多GPU版本
def evaluate_binary(model, dataloader, device, rank, world_size, use_fp16=True):
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
    
    # 在多GPU之间收集所有结果
    if world_size > 1:
        all_preds, all_labels = gather_results(all_preds, all_labels, world_size, device)
    
    # 计算平均loss
    loss_tensor = torch.tensor([total_loss, len(dataloader)], device=device)
    if world_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor[0].item() / loss_tensor[1].item()
    
    # 只在主进程计算指标
    if rank == 0:
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
        
        # 计算不同阈值下的指标
        if world_size == 1:  # 单GPU情况下才计算阈值指标
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
        else:
            threshold_metrics = {}
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs if world_size == 1 else None,
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
        print(f"开始二分类多GPU测试，使用 {world_size} 个GPU: {args.gpu_ids}")
        
        # 记录GPU内存信息
        if torch.cuda.is_available():
            for i, gpu_id in enumerate(args.gpu_ids):
                if gpu_id < torch.cuda.device_count():
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    print(f"GPU {gpu_id}: {memory_total:.2f}GB total memory")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载二分类基础模型
    if is_main_process:
        print("加载二分类基础模型...")
    
    base_model = BinaryQwen3ForSequenceClassification(args.checkpoint)
    
    # 直接从保存的LoRA模型加载
    if is_main_process:
        print(f"加载二分类LoRA模型: {args.lora_model}")
    
    try:
        model = PeftModel.from_pretrained(base_model, args.lora_model)
        if is_main_process:
            print("二分类LoRA模型加载成功！")
    except Exception as e:
        if is_main_process:
            print(f"二分类LoRA模型加载失败: {e}")
            print("错误详情:")
            import traceback
            traceback.print_exc()
        
        # 清理并退出
        cleanup_distributed()
        return
    
    # 将模型移到设备并包装为DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    model.eval()
    
    if is_main_process:
        print("二分类模型加载完成！")
        print(f"模型已加载到多个GPU: {args.gpu_ids}")
    
    # 准备测试数据集
    if is_main_process:
        print("加载测试数据集...")
    
    test_dataset = BinaryClassificationDataset(args.test_data, tokenizer, args.max_length)
    
    if is_main_process:
        print(f"测试集样本数: {len(test_dataset)}")
    
    # 创建分布式采样器
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # 确保所有进程都完成初始化
    dist.barrier()
    
    # 评估模型
    if is_main_process:
        print("\n开始二分类多GPU评估...")
    
    test_results = evaluate_binary(model, test_loader, device, rank, world_size, use_fp16=args.fp16)
    
    # 只在主进程打印和保存结果
    if is_main_process and test_results and 'accuracy' in test_results:
        # 打印基础结果
        print("\n=== 二分类多GPU测试集结果 ===")
        print(f"Loss: {test_results['loss']:.4f}")
        print(f"准确率: {test_results['accuracy']:.4f}")
        print(f"F1: {test_results['f1']:.4f}")
        print(f"精确率: {test_results['precision']:.4f}")
        print(f"召回率: {test_results['recall']:.4f}")
        
        # 关键业务指标
        print("\n=== 【关键业务指标】 ===")
        print(f"正常误报率: {test_results['normal_false_positive_rate']:.4f}")
        print(f"违规漏报率: {test_results['violation_miss_rate']:.4f}")
        print(f"违规召回率: {test_results['violation_recall']:.4f}")
        print(f"正常特异性: {test_results['normal_specificity']:.4f}")
        
        # 混淆矩阵
        cm = test_results['confusion_matrix']
        print("\n=== 【混淆矩阵】 ===")
        print(f"正常->正常: {cm['true_negative']}")
        print(f"正常->违规: {cm['false_positive']}")
        print(f"违规->正常: {cm['false_negative']}")
        print(f"违规->违规: {cm['true_positive']}")
        
        # 样本分布
        support = test_results['support']
        print(f"\n=== 【样本分布】 ===")
        print(f"正常样本数: {support['normal']}")
        print(f"违规样本数: {support['violation']}")
        print(f"总样本数: {support['normal'] + support['violation']}")
        
        # 不同阈值下的指标
        if test_results['threshold_metrics']:
            print("\n=== 【不同阈值下的指标】 ===")
            for threshold, metrics in test_results['threshold_metrics'].items():
                print(f"阈值{threshold}: 准确率={metrics['accuracy']:.4f}, "
                      f"误报率={metrics['normal_false_positive_rate']:.4f}, "
                      f"漏报率={metrics['violation_miss_rate']:.4f}")
        
        # 打印详细的分类报告
        print("\n=== 【分类报告】 ===")
        from sklearn.metrics import classification_report
        target_names = ['正常', '违规']
        print(classification_report(test_results['labels'], test_results['predictions'], 
                                   target_names=target_names, digits=4))
        
        # 保存结果
        output_dir = os.path.dirname(args.lora_model)
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'binary_multi_gpu_test_results_{timestamp}.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': {
                    'model_type': 'binary_classification',
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
                    # 业务关键指标
                    'normal_false_positive_rate': test_results['normal_false_positive_rate'],
                    'violation_miss_rate': test_results['violation_miss_rate'],
                    'violation_recall': test_results['violation_recall'],
                    'normal_specificity': test_results['normal_specificity'],
                    # 混淆矩阵和支持度
                    'confusion_matrix': test_results['confusion_matrix'],
                    'support': test_results['support'],
                    # 不同阈值下的指标
                    'threshold_metrics': test_results['threshold_metrics']
                }
            }, f, indent=4, ensure_ascii=False)
        
        print(f"\n二分类多GPU测试结果已保存到: {results_file}")
        
        # 关键业务指标总结
        print("\n=== 关键业务指标总结 ===")
        print(f"✓ 正常内容误报率: {test_results['normal_false_positive_rate']:.4f}")
        print(f"✓ 违规内容漏报率: {test_results['violation_miss_rate']:.4f}")
        print(f"✓ 违规内容召回率: {test_results['violation_recall']:.4f}")
        print(f"✓ 整体准确率: {test_results['accuracy']:.4f}")
        print(f"✓ 使用了 {world_size} 个GPU: {args.gpu_ids}")
        print(f"✓ 模型类型: 二分类 (正常 vs 违规)")
    
    # 确保所有进程都完成评估后再清理
    dist.barrier()
    
    # 清理分布式环境
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="二分类LoRA模型多GPU测试脚本")
    parser.add_argument('--checkpoint', type=str, required=False, default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                        help="原始模型的路径")
    parser.add_argument('--lora_model', type=str, required=False, default="./lora-binary-classification/checkpoint-binary-1000",
                        help="保存的二分类LoRA模型路径")
    parser.add_argument('--test_data', type=str, required=False, default="./data/r789-b-50000_test.xlsx",
                        help="测试数据集路径")
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--max_length', type=int, default=256)
    
    # 多GPU配置
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5",
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
    print(f"开始启动二分类多进程测试，世界大小: {world_size}")
    print(f"二分类LoRA模型路径: {args.lora_model}")
    print(f"测试数据路径: {args.test_data}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.max_length}")
    print(f"使用混合精度: {args.fp16}")
    print(f"模型类型: 二分类 (正常 vs 违规)")
    
    # 启动多进程测试
    mp.spawn(test_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()