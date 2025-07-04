import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import argparse
from tqdm import tqdm
from qwen3_classification_direct import Qwen3ForSequenceClassification
from train_lora import ClassificationDataset, evaluate

def test_model(model_path, test_data_path, checkpoint_path, batch_size=8, max_length=512):
    """
    测试LoRA微调后的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"从 {model_path} 加载模型...")
    model = Qwen3ForSequenceClassification(checkpoint_path, num_labels=6)
    
    # 配置LoRA（需要与训练时相同）
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["base_model.lm_head", "dropout", "classifier"]
    )
    
    # 应用LoRA并加载权重
    model = get_peft_model(model, lora_config)
    
    # 加载微调后的权重
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    model.to(device)
    model.eval()
    
    # 准备测试数据
    test_dataset = ClassificationDataset(test_data_path, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 评估模型
    print("开始评估...")
    results = evaluate(model, test_loader, device)
    
    # 打印结果
    print("\n=== 测试结果 ===")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    # 打印混淆矩阵
    cm = confusion_matrix(results['labels'], results['predictions'])
    print("\n混淆矩阵:")
    print(cm)
    
    # 打印详细的分类报告
    print("\n分类报告:")
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=[f'Class {i}' for i in range(6)]))
    
    # 分析错误案例
    analyze_errors(test_dataset, results['predictions'], results['labels'], tokenizer, num_examples=10)
    
    return results

def analyze_errors(dataset, predictions, labels, tokenizer, num_examples=10):
    """
    分析预测错误的案例
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 找出错误的索引
    error_indices = np.where(predictions != labels)[0]
    
    if len(error_indices) == 0:
        print("\n没有错误的预测！")
        return
    
    print(f"\n=== 错误案例分析 (共 {len(error_indices)} 个错误) ===")
    print(f"显示前 {min(num_examples, len(error_indices))} 个错误案例:\n")
    
    for i, idx in enumerate(error_indices[:num_examples]):
        text = dataset.data.iloc[idx]['text']
        true_label = labels[idx]
        pred_label = predictions[idx]
        
        print(f"案例 {i+1}:")
        print(f"文本: {text[:200]}..." if len(text) > 200 else f"文本: {text}")
        print(f"真实标签: {true_label}")
        print(f"预测标签: {pred_label}")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="测试LoRA微调后的模型")
    parser.add_argument('--model_path', type=str, default="zzz/lora_model/best_model",
                       help="微调后的模型路径")
    parser.add_argument('--test_data', type=str, default="zzz/val-r456-6-3000.xlsx",
                       help="测试数据路径")
    parser.add_argument('--checkpoint', type=str,
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                       help="基础模型路径")
    parser.add_argument('--batch_size', type=int, default=8,
                       help="批次大小")
    parser.add_argument('--max_length', type=int, default=512,
                       help="最大序列长度")
    
    args = parser.parse_args()
    
    # 测试模型
    test_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()