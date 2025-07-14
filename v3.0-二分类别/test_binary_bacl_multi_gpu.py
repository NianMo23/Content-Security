import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch.multiprocessing as mp
from train_binary_bacl_multi_gpu import (
    BACLBinaryQwen3ForSequenceClassification,
    BinaryClassificationDataset,
    set_seed,
    setup_distributed,
    cleanup_distributed
)

def setup_test_logging(rank, output_dir):
    """设置测试日志记录配置"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_file = os.path.join(output_dir, 'bacl_test_results.log')
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

def comprehensive_bacl_analysis(y_true, y_pred, y_prob, original_labels=None):
    """
    全面的BACL模型性能分析
    
    Args:
        y_true: 真实标签 (0=正常, 1=违规)
        y_pred: 预测标签 (0=正常, 1=违规)
        y_prob: 预测概率 [N, 2]
        original_labels: 原始6分类标签（用于细粒度分析）
    """
    # 基础分类指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )
    
    # 计算每个类别的详细指标
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 核心业务指标
    normal_false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    violation_miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    violation_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    normal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 置信度分析
    violation_probs = y_prob[:, 1]
    max_probs = np.max(y_prob, axis=1)
    
    # 不同置信度阈值下的性能
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_analysis = {}
    
    for thresh in thresholds:
        thresh_pred = (violation_probs >= thresh).astype(int)
        thresh_acc = accuracy_score(y_true, thresh_pred)
        thresh_precision, thresh_recall, thresh_f1, _ = precision_recall_fscore_support(
            y_true, thresh_pred, average='binary', pos_label=1
        )
        
        # 计算在该阈值下的关键指标
        thresh_cm = confusion_matrix(y_true, thresh_pred, labels=[0, 1])
        thresh_tn, thresh_fp, thresh_fn, thresh_tp = thresh_cm.ravel()
        
        thresh_normal_fp_rate = thresh_fp / (thresh_tn + thresh_fp) if (thresh_tn + thresh_fp) > 0 else 0.0
        thresh_violation_miss_rate = thresh_fn / (thresh_tp + thresh_fn) if (thresh_tp + thresh_fn) > 0 else 0.0
        
        threshold_analysis[thresh] = {
            'accuracy': thresh_acc,
            'precision': thresh_precision,
            'recall': thresh_recall,
            'f1': thresh_f1,
            'normal_false_positive_rate': thresh_normal_fp_rate,
            'violation_miss_rate': thresh_violation_miss_rate,
            'predicted_positive_ratio': np.mean(thresh_pred),
            'high_confidence_ratio': np.mean(violation_probs >= thresh)
        }
    
    # 置信度分布统计
    confidence_stats = {
        'avg_confidence': np.mean(max_probs),
        'confidence_std': np.std(max_probs),
        'min_confidence': np.min(max_probs),
        'max_confidence': np.max(max_probs),
        'median_confidence': np.median(max_probs),
        'high_confidence_ratio': np.mean(max_probs > 0.8),
        'medium_confidence_ratio': np.mean((max_probs >= 0.6) & (max_probs <= 0.8)),
        'low_confidence_ratio': np.mean(max_probs < 0.6)
    }
    
    # 各类别置信度分析
    normal_indices = np.where(y_true == 0)[0]
    violation_indices = np.where(y_true == 1)[0]
    
    class_confidence_stats = {}
    if len(normal_indices) > 0:
        normal_confidences = max_probs[normal_indices]
        class_confidence_stats['normal'] = {
            'avg_confidence': np.mean(normal_confidences),
            'confidence_std': np.std(normal_confidences),
            'high_confidence_ratio': np.mean(normal_confidences > 0.8)
        }
    
    if len(violation_indices) > 0:
        violation_confidences = max_probs[violation_indices]
        class_confidence_stats['violation'] = {
            'avg_confidence': np.mean(violation_confidences),
            'confidence_std': np.std(violation_confidences),
            'high_confidence_ratio': np.mean(violation_confidences > 0.8)
        }
    
    # 错误样本分析
    error_analysis = {
        'false_positives': {
            'count': int(fp),
            'avg_confidence': np.mean(violation_probs[y_true == 0]) if len(violation_probs[y_true == 0]) > 0 else 0.0,
            'high_confidence_errors': np.sum((y_true == 0) & (violation_probs > 0.8))
        },
        'false_negatives': {
            'count': int(fn),
            'avg_confidence': np.mean(violation_probs[y_true == 1]) if len(violation_probs[y_true == 1]) > 0 else 0.0,
            'low_confidence_errors': np.sum((y_true == 1) & (violation_probs < 0.2))
        }
    }
    
    # 原始标签细粒度分析
    original_analysis = None
    if original_labels is not None:
        original_analysis = analyze_original_labels_performance(
            y_true, y_pred, y_prob, original_labels
        )
    
    return {
        'basic_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support.tolist() if hasattr(support, 'tolist') else support
        },
        'class_metrics': {
            'normal': {
                'precision': class_precision[0],
                'recall': class_recall[0],
                'f1': class_f1[0],
                'support': int(class_support[0])
            },
            'violation': {
                'precision': class_precision[1],
                'recall': class_recall[1],
                'f1': class_f1[1],
                'support': int(class_support[1])
            }
        },
        'business_metrics': {
            'normal_false_positive_rate': normal_false_positive_rate,
            'violation_miss_rate': violation_miss_rate,
            'violation_recall': violation_recall,
            'normal_specificity': normal_specificity
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'matrix': cm.tolist()
        },
        'threshold_analysis': threshold_analysis,
        'confidence_stats': confidence_stats,
        'class_confidence_stats': class_confidence_stats,
        'error_analysis': error_analysis,
        'original_analysis': original_analysis
    }

def analyze_original_labels_performance(y_true, y_pred, y_prob, original_labels):
    """分析原始6分类标签在二分类模型上的表现"""
    # 原始标签映射
    original_label_names = ['正常', '歧视', '违法违规', '政治安全', '暴恐', '色情低俗']
    
    # 统计各原始标签的性能
    original_stats = {}
    
    for i, label_name in enumerate(original_label_names):
        label_indices = np.where(original_labels == i)[0]
        
        if len(label_indices) > 0:
            # 该标签的真实二分类标签
            true_binary = y_true[label_indices]
            pred_binary = y_pred[label_indices]
            prob_binary = y_prob[label_indices]
            
            # 计算该标签的准确率
            accuracy = accuracy_score(true_binary, pred_binary)
            
            # 计算平均置信度
            max_probs = np.max(prob_binary, axis=1)
            avg_confidence = np.mean(max_probs)
            
            # 如果是违规类，计算召回率
            if i > 0:  # 违规类
                recall = np.mean(pred_binary)  # 预测为违规的比例
            else:  # 正常类
                recall = np.mean(pred_binary == 0)  # 预测为正常的比例
            
            original_stats[label_name] = {
                'count': len(label_indices),
                'accuracy': accuracy,
                'recall': recall,
                'avg_confidence': avg_confidence,
                'correct_predictions': np.sum(true_binary == pred_binary)
            }
    
    return original_stats

def save_detailed_results(results, output_dir, rank):
    """保存详细的测试结果"""
    if rank == 0:
        import json
        import time
        
        # 保存JSON格式结果
        results_with_metadata = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'BACL_Binary_Classification',
            'results': results
        }
        
        json_file = os.path.join(output_dir, 'bacl_detailed_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        # 保存易读的文本格式结果
        txt_file = os.path.join(output_dir, 'bacl_detailed_results.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BACL二分类模型详细测试结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"测试时间: {results_with_metadata['timestamp']}\n")
            f.write(f"模型类型: {results_with_metadata['model_type']}\n\n")
            
            # 基础指标
            basic = results['basic_metrics']
            f.write("【基础分类指标】\n")
            f.write(f"准确率: {basic['accuracy']:.4f}\n")
            f.write(f"精确率: {basic['precision']:.4f}\n")
            f.write(f"召回率: {basic['recall']:.4f}\n")
            f.write(f"F1分数: {basic['f1']:.4f}\n\n")
            
            # 各类别指标
            f.write("【各类别详细指标】\n")
            class_metrics = results['class_metrics']
            for class_name, metrics in class_metrics.items():
                f.write(f"{class_name}类:\n")
                f.write(f"  精确率: {metrics['precision']:.4f}\n")
                f.write(f"  召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                f.write(f"  样本数: {metrics['support']}\n")
            f.write("\n")
            
            # 关键业务指标
            business = results['business_metrics']
            f.write("【关键业务指标】\n")
            f.write(f"正常类误报率: {business['normal_false_positive_rate']:.4f}\n")
            f.write(f"违规类漏报率: {business['violation_miss_rate']:.4f}\n")
            f.write(f"违规类召回率: {business['violation_recall']:.4f}\n")
            f.write(f"正常类特异性: {business['normal_specificity']:.4f}\n\n")
            
            # 混淆矩阵
            cm = results['confusion_matrix']
            f.write("【混淆矩阵】\n")
            f.write(f"真正常(TN): {cm['true_negative']}\n")
            f.write(f"假违规(FP): {cm['false_positive']}\n")
            f.write(f"假正常(FN): {cm['false_negative']}\n")
            f.write(f"真违规(TP): {cm['true_positive']}\n\n")
            
            # 置信度统计
            conf_stats = results['confidence_stats']
            f.write("【置信度统计】\n")
            f.write(f"平均置信度: {conf_stats['avg_confidence']:.4f}\n")
            f.write(f"置信度标准差: {conf_stats['confidence_std']:.4f}\n")
            f.write(f"最小置信度: {conf_stats['min_confidence']:.4f}\n")
            f.write(f"最大置信度: {conf_stats['max_confidence']:.4f}\n")
            f.write(f"高置信度比例(>0.8): {conf_stats['high_confidence_ratio']:.4f}\n")
            f.write(f"中等置信度比例(0.6-0.8): {conf_stats['medium_confidence_ratio']:.4f}\n")
            f.write(f"低置信度比例(<0.6): {conf_stats['low_confidence_ratio']:.4f}\n\n")
            
            # 阈值分析
            f.write("【不同阈值下的性能分析】\n")
            threshold_analysis = results['threshold_analysis']
            for thresh, metrics in threshold_analysis.items():
                f.write(f"阈值 {thresh}:\n")
                f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
                f.write(f"  精确率: {metrics['precision']:.4f}\n")
                f.write(f"  召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                f.write(f"  正常误报率: {metrics['normal_false_positive_rate']:.4f}\n")
                f.write(f"  违规漏报率: {metrics['violation_miss_rate']:.4f}\n")
                f.write(f"  预测为违规比例: {metrics['predicted_positive_ratio']:.4f}\n")
                f.write("\n")
            
            # 错误分析
            error_analysis = results['error_analysis']
            f.write("【错误样本分析】\n")
            f.write(f"假阳性错误:\n")
            f.write(f"  数量: {error_analysis['false_positives']['count']}\n")
            f.write(f"  平均置信度: {error_analysis['false_positives']['avg_confidence']:.4f}\n")
            f.write(f"  高置信度错误: {error_analysis['false_positives']['high_confidence_errors']}\n")
            f.write(f"假阴性错误:\n")
            f.write(f"  数量: {error_analysis['false_negatives']['count']}\n")
            f.write(f"  平均置信度: {error_analysis['false_negatives']['avg_confidence']:.4f}\n")
            f.write(f"  低置信度错误: {error_analysis['false_negatives']['low_confidence_errors']}\n\n")
            
            # 原始标签分析
            if results['original_analysis'] is not None:
                f.write("【原始标签细粒度分析】\n")
                original_analysis = results['original_analysis']
                for label_name, stats in original_analysis.items():
                    f.write(f"{label_name}:\n")
                    f.write(f"  样本数: {stats['count']}\n")
                    f.write(f"  准确率: {stats['accuracy']:.4f}\n")
                    f.write(f"  召回率: {stats['recall']:.4f}\n")
                    f.write(f"  平均置信度: {stats['avg_confidence']:.4f}\n")
                    f.write(f"  正确预测数: {stats['correct_predictions']}\n")
                    f.write("\n")
        
        return json_file, txt_file
    return None, None

def test_bacl_binary_model(model, dataloader, device, rank, use_fp16=True):
    """测试BACL二分类模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_original_labels = []
    
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 获取原始标签（如果有的话）
            original_labels = batch.get('original_labels', None)
            
            # 混合精度推理
            with torch.amp.autocast('cuda', enabled=use_fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                result, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            logits = result.logits
            
            # 获取预测和概率
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().float().numpy())
            
            if original_labels is not None:
                all_original_labels.extend(original_labels.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_original_labels = np.array(all_original_labels) if all_original_labels else None
    
    return all_predictions, all_labels, all_probabilities, all_original_labels

def aggregate_results_across_gpus(predictions, labels, probabilities, original_labels=None):
    """聚合多个GPU的测试结果"""
    world_size = dist.get_world_size()
    
    # 收集所有GPU的结果
    all_predictions = [None] * world_size
    all_labels = [None] * world_size
    all_probabilities = [None] * world_size
    all_original_labels = [None] * world_size if original_labels is not None else None
    
    # 聚合predictions
    dist.all_gather_object(all_predictions, predictions)
    dist.all_gather_object(all_labels, labels)
    dist.all_gather_object(all_probabilities, probabilities)
    
    if original_labels is not None:
        dist.all_gather_object(all_original_labels, original_labels)
    
    # 合并结果
    final_predictions = np.concatenate(all_predictions)
    final_labels = np.concatenate(all_labels)
    final_probabilities = np.concatenate(all_probabilities)
    final_original_labels = np.concatenate(all_original_labels) if all_original_labels[0] is not None else None
    
    return final_predictions, final_labels, final_probabilities, final_original_labels

def test_worker(rank, world_size, args):
    """每个GPU上的测试工作进程"""
    actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
    
    setup_distributed(rank, world_size)
    logger = setup_test_logging(rank, args.output_dir)
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"开始BACL二分类模型测试，使用 {world_size} 个GPU")
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"测试数据: {args.test_data}")
    
    torch.cuda.set_device(actual_gpu_id)
    set_seed(args.seed)
    device = torch.device(f'cuda:{actual_gpu_id}')
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载BACL模型
    if is_main_process:
        logger.info("正在加载BACL二分类模型...")
    
    model = BACLBinaryQwen3ForSequenceClassification.from_pretrained(
        args.model_path,
        feature_dim=args.feature_dim,
        use_adversarial=args.use_adversarial
    )
    
    # 将模型移到设备并包装为DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id)
    
    # 准备测试数据集
    if is_main_process:
        logger.info("加载测试数据集...")
    
    test_dataset = BinaryClassificationDataset(args.test_data, tokenizer, args.max_length)
    
    if is_main_process:
        logger.info(f"测试集样本数: {len(test_dataset)}")
    
    # 创建分布式采样器
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    # 执行测试
    if is_main_process:
        logger.info("开始测试...")
    
    predictions, labels, probabilities, original_labels = test_bacl_binary_model(
        model, test_loader, device, rank, args.fp16
    )
    
    # 聚合所有GPU的结果
    if is_main_process:
        logger.info("聚合多GPU测试结果...")
    
    final_predictions, final_labels, final_probabilities, final_original_labels = aggregate_results_across_gpus(
        predictions, labels, probabilities, original_labels
    )
    
    # 主进程进行结果分析和保存
    if is_main_process:
        logger.info("开始进行全面性能分析...")
        
        # 进行全面分析
        results = comprehensive_bacl_analysis(
            final_labels, final_predictions, final_probabilities, final_original_labels
        )
        
        # 打印关键结果
        logger.info("=" * 60)
        logger.info("BACL二分类模型测试结果")
        logger.info("=" * 60)
        
        # 基础指标
        basic = results['basic_metrics']
        logger.info(f"基础指标:")
        logger.info(f"  准确率: {basic['accuracy']:.4f}")
        logger.info(f"  精确率: {basic['precision']:.4f}")
        logger.info(f"  召回率: {basic['recall']:.4f}")
        logger.info(f"  F1分数: {basic['f1']:.4f}")
        
        # 关键业务指标
        business = results['business_metrics']
        logger.info(f"\n关键业务指标:")
        logger.info(f"  正常类误报率: {business['normal_false_positive_rate']:.4f}")
        logger.info(f"  违规类漏报率: {business['violation_miss_rate']:.4f}")
        logger.info(f"  违规类召回率: {business['violation_recall']:.4f}")
        logger.info(f"  正常类特异性: {business['normal_specificity']:.4f}")
        
        # 混淆矩阵
        cm = results['confusion_matrix']
        logger.info(f"\n混淆矩阵:")
        logger.info(f"  真正常(TN): {cm['true_negative']}")
        logger.info(f"  假违规(FP): {cm['false_positive']}")
        logger.info(f"  假正常(FN): {cm['false_negative']}")
        logger.info(f"  真违规(TP): {cm['true_positive']}")
        
        # 置信度统计
        conf_stats = results['confidence_stats']
        logger.info(f"\n置信度统计:")
        logger.info(f"  平均置信度: {conf_stats['avg_confidence']:.4f}")
        logger.info(f"  置信度标准差: {conf_stats['confidence_std']:.4f}")
        logger.info(f"  高置信度比例(>0.8): {conf_stats['high_confidence_ratio']:.4f}")
        logger.info(f"  低置信度比例(<0.6): {conf_stats['low_confidence_ratio']:.4f}")
        
        # 最优阈值分析
        best_f1_thresh = max(results['threshold_analysis'].keys(), 
                           key=lambda x: results['threshold_analysis'][x]['f1'])
        best_balance_thresh = min(results['threshold_analysis'].keys(),
                                key=lambda x: results['threshold_analysis'][x]['normal_false_positive_rate'] + 
                                             results['threshold_analysis'][x]['violation_miss_rate'])
        
        logger.info(f"\n阈值分析:")
        logger.info(f"  最佳F1阈值: {best_f1_thresh} (F1={results['threshold_analysis'][best_f1_thresh]['f1']:.4f})")
        logger.info(f"  最佳平衡阈值: {best_balance_thresh} (误报+漏报={results['threshold_analysis'][best_balance_thresh]['normal_false_positive_rate'] + results['threshold_analysis'][best_balance_thresh]['violation_miss_rate']:.4f})")
        
        # 原始标签分析
        if results['original_analysis'] is not None:
            logger.info(f"\n原始标签细粒度分析:")
            for label_name, stats in results['original_analysis'].items():
                logger.info(f"  {label_name}: 准确率={stats['accuracy']:.4f}, 样本数={stats['count']}")
        
        # 保存详细结果
        json_file, txt_file = save_detailed_results(results, args.output_dir, rank)
        
        if json_file:
            logger.info(f"\n详细结果已保存到:")
            logger.info(f"  JSON格式: {json_file}")
            logger.info(f"  文本格式: {txt_file}")
        
        # 性能评估总结
        logger.info("\n" + "=" * 60)
        logger.info("BACL模型性能评估总结")
        logger.info("=" * 60)
        
        # 判断是否达到预期目标
        target_normal_fp = 0.04  # 目标正常误报率 < 4%
        target_violation_mr = 0.06  # 目标违规漏报率 < 6%
        target_accuracy = 0.96  # 目标准确率 >= 96%
        
        normal_fp_achieved = business['normal_false_positive_rate'] < target_normal_fp
        violation_mr_achieved = business['violation_miss_rate'] < target_violation_mr
        accuracy_achieved = basic['accuracy'] >= target_accuracy
        
        logger.info(f"目标达成情况:")
        logger.info(f"  正常误报率 < 4%: {'✓' if normal_fp_achieved else '✗'} (实际: {business['normal_false_positive_rate']:.4f})")
        logger.info(f"  违规漏报率 < 6%: {'✓' if violation_mr_achieved else '✗'} (实际: {business['violation_miss_rate']:.4f})")
        logger.info(f"  准确率 >= 96%: {'✓' if accuracy_achieved else '✗'} (实际: {basic['accuracy']:.4f})")
        
        all_targets_met = normal_fp_achieved and violation_mr_achieved and accuracy_achieved
        logger.info(f"\n整体目标达成: {'✓ 所有目标均已达成' if all_targets_met else '✗ 部分目标未达成'}")
        
        if not all_targets_met:
            logger.info(f"\n建议优化方向:")
            if not normal_fp_achieved:
                logger.info(f"  - 增加正常类特征提取能力")
                logger.info(f"  - 调整BACL温度参数以增强类间分离")
            if not violation_mr_achieved:
                logger.info(f"  - 提高违规类召回率")
                logger.info(f"  - 增强对抗性特征解耦")
            if not accuracy_achieved:
                logger.info(f"  - 整体模型优化")
                logger.info(f"  - 考虑增加训练数据或调整超参数")
    
    # 确保所有进程都完成测试后再清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="BACL二分类模型多GPU测试脚本")
    
    # 基础配置
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的BACL模型路径')
    parser.add_argument('--test_data', type=str, required=True,
                       help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default="./bacl_test_results",
                       help='测试结果输出目录')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='测试批次大小')
    parser.add_argument('--max_length', type=int, default=256,
                       help='最大序列长度')
    
    # BACL模型参数
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='特征维度（需与训练时一致）')
    parser.add_argument('--use_adversarial', action='store_true', default=True,
                       help='是否使用对抗性特征解耦（需与训练时一致）')
    
    # 系统配置
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='是否使用混合精度')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5",
                       help='指定使用的GPU ID，用逗号分隔')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='指定使用的GPU数量，从GPU 0开始使用')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行多GPU测试")
    
    # 处理GPU配置
    total_gpus = torch.cuda.device_count()
    print(f"系统检测到 {total_gpus} 个GPU")
    
    # 解析GPU配置
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        for gpu_id in gpu_ids:
            if gpu_id >= total_gpus:
                raise ValueError(f"指定的GPU ID {gpu_id} 超出可用范围 (0-{total_gpus-1})")
        args.gpu_ids = gpu_ids
        world_size = len(gpu_ids)
        print(f"使用指定的GPU: {gpu_ids}")
    elif args.num_gpus is not None:
        if args.num_gpus > total_gpus:
            raise ValueError(f"指定的GPU数量 {args.num_gpus} 超出可用GPU数量 {total_gpus}")
        args.gpu_ids = list(range(args.num_gpus))
        world_size = args.num_gpus
        print(f"使用前 {args.num_gpus} 个GPU: {args.gpu_ids}")
    else:
        args.gpu_ids = list(range(total_gpus))
        world_size = total_gpus
        print(f"使用所有GPU: {args.gpu_ids}")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        raise ValueError(f"模型路径不存在: {args.model_path}")
    
    # 检查测试数据
    if not os.path.exists(args.test_data):
        raise ValueError(f"测试数据路径不存在: {args.test_data}")
    
    print(f"开始BACL二分类模型多GPU测试...")
    print(f"模型路径: {args.model_path}")
    print(f"测试数据: {args.test_data}")
    print(f"使用 {world_size} 个GPU进行测试")
    
    # 使用spawn方法启动多进程
    mp.spawn(test_worker, args=(world_size, args), nprocs=world_size, join=True)
    
    print("BACL二分类模型测试完成！")

if __name__ == "__main__":
    main()