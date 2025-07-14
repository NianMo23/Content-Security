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

# 自定义数据集类
class ClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        # 读取CSV文件
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 获取所有数据用于推理
        self.qa_ids = self.data.iloc[:, 0].tolist()  # 第一列：问答id
        self.inputs = self.data.iloc[:, 1].tolist()  # 第二列：大模型的输入内容
        self.responses = self.data.iloc[:, 2].tolist()  # 第三列：生成的回答
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 对生成的回答进行分类
        text = str(self.responses[idx])
        
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
            'idx': idx  # 保存索引以便后续使用
        }

# 推理函数
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            indices = batch['idx']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_indices.extend(indices.numpy())
    
    return all_preds, all_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B",
                        help="原始模型的路径")
    parser.add_argument('--lora_model', type=str, required=False, default="./lora_model-2015/checkpoint-5000",
                        help="保存的LoRA模型路径")
    parser.add_argument('--test_data', type=str, required=False, default="./S5_stage_one_data.csv",
                        help="测试数据集路径（CSV文件）")
    parser.add_argument('--output_file', type=str, required=False, default="./cert_test_results.xlsx",
                        help="输出结果文件路径")
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--max_length', type=int, default=256)
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = Qwen3ForSequenceClassification(args.checkpoint, num_labels=6)
    
    # 直接从保存的LoRA模型加载
    print(f"加载LoRA模型: {args.lora_model}")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    
    # 将模型移到设备
    model.to(device)
    model.eval()
    
    print("LoRA模型加载完成！")
    print(f"模型已加载到 {device}")
    
    # 准备测试数据集
    print("加载测试数据集...")
    test_dataset = ClassificationDataset(args.test_data, tokenizer, args.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 进行预测
    print("\n开始预测...")
    predictions, indices = predict(model, test_loader, device)
    
    # 定义标签映射（数字到文本）
    label_map_reverse = {
        0: '正常',
        1: '歧视',
        2: '违法违规',
        3: '政治安全',
        4: '暴恐',
        5: '色情低俗'
    }
    
    # 准备输出数据
    results = []
    for idx in range(len(predictions)):
        original_idx = indices[idx]
        qa_id = test_dataset.qa_ids[original_idx]
        response = test_dataset.responses[original_idx]
        pred_label = label_map_reverse[predictions[idx]]
        
        results.append({
            '问答id': qa_id,
            '生成的回答': response,
            '标签': pred_label
        })
    
    # 保存结果到Excel文件
    print(f"\n保存结果到 {args.output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_excel(args.output_file, index=False, engine='openpyxl')
    
    print(f"预测完成！结果已保存到 {args.output_file}")
    
    # 打印一些统计信息
    print("\n标签分布统计:")
    label_counts = results_df['标签'].value_counts()
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(results)*100:.2f}%)")

if __name__ == "__main__":
    main()