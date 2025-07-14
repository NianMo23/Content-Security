import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import time
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from qwen3_classification_direct import Qwen3ForSequenceClassification
import torch.multiprocessing as mp

# =========================== 边界感知对比损失 (BACL) ===========================
class BoundaryAwareContrastiveLoss(nn.Module):
    """
    边界感知对比损失 (Boundary-Aware Contrastive Loss)
    
    核心思想：
    1. 通过对比学习增强类间边界
    2. 使用温度参数控制分布sharpness
    3. 结合边界感知机制提升判别能力
    """
    def __init__(self, temperature=0.1, margin=0.5, alpha=1.0, beta=0.5):
        super(BoundaryAwareContrastiveLoss, self).__init__()
        self.temperature = temperature  # 温度参数，控制对比强度
        self.margin = margin           # 边界margin
        self.alpha = alpha            # 对比损失权重
        self.beta = beta              # 边界损失权重
        
    def forward(self, features, logits, labels):
        """
        Args:
            features: [N, D] 特征表示
            logits: [N, 2] 分类logits
            labels: [N] 标签 (0=正常, 1=违规)
        """
        batch_size = features.size(0)
        device = features.device
        
        # 1. 计算对比损失
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        negative_mask = (labels_expanded != labels_expanded.T).float()
        
        # 去除对角线（自身）
        identity_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - identity_mask)
        
        # 对比损失计算
        # 正样本对：相同类别
        positive_exp = torch.exp(similarity_matrix) * positive_mask
        # 负样本对：不同类别 + 所有样本（作为分母）
        all_exp = torch.exp(similarity_matrix) * (1 - identity_mask)
        
        # 避免除零
        positive_exp = positive_exp + 1e-8
        all_exp = all_exp.sum(dim=1, keepdim=True) + 1e-8
        
        # 计算对比损失
        contrastive_loss = -torch.log(positive_exp / all_exp)
        contrastive_loss = contrastive_loss * positive_mask
        contrastive_loss = contrastive_loss.sum() / (positive_mask.sum() + 1e-8)
        
        # 2. 计算边界感知损失
        # 获取预测概率
        probs = F.softmax(logits, dim=1)
        
        # 计算置信度（最大概率）
        max_probs, _ = torch.max(probs, dim=1)
        
        # 边界感知：低置信度样本应该远离决策边界
        boundary_loss = torch.mean((1 - max_probs) * torch.abs(probs[:, 1] - 0.5))
        
        # 3. 类别分离损失
        # 增强不同类别特征的分离度
        normal_features = features[labels == 0]  # 正常类特征
        violation_features = features[labels == 1]  # 违规类特征
        
        separation_loss = 0.0
        if normal_features.size(0) > 0 and violation_features.size(0) > 0:
            # 计算类内紧密度
            normal_center = torch.mean(normal_features, dim=0)
            violation_center = torch.mean(violation_features, dim=0)
            
            # 类间距离应该大于margin
            inter_class_distance = torch.norm(normal_center - violation_center)
            separation_loss = F.relu(self.margin - inter_class_distance)
        
        # 4. 组合损失
        total_loss = (self.alpha * contrastive_loss + 
                     self.beta * boundary_loss + 
                     0.3 * separation_loss)
        
        return total_loss, {
            'contrastive_loss': contrastive_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'separation_loss': separation_loss.item() if isinstance(separation_loss, torch.Tensor) else separation_loss
        }

# =========================== 双通道特征提取器 ===========================
class DualChannelFeatureExtractor(nn.Module):
    """
    双通道特征提取器
    
    通道1：全局语义特征（整体理解）
    通道2：局部关键特征（关键词识别）
    """
    def __init__(self, hidden_size, feature_dim=256):
        super(DualChannelFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        
        # 全局特征通道
        self.global_channel = nn.Sequential(
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 局部特征通道（注意力机制）
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            dropout=0.1, 
            batch_first=True
        )
        self.local_channel = nn.Sequential(
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. 全局特征：使用平均池化
        if attention_mask is not None:
            # 考虑padding的加权平均
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            global_repr = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_repr = hidden_states.mean(dim=1)
        
        global_features = self.global_channel(global_repr)
        
        # 2. 局部特征：使用自注意力机制
        local_repr, _ = self.local_attention(hidden_states, hidden_states, hidden_states, 
                                           key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # 对局部特征进行加权平均
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(local_repr)
            local_repr = (local_repr * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            local_repr = local_repr.mean(dim=1)
            
        local_features = self.local_channel(local_repr)
        
        # 3. 特征融合
        combined_features = torch.cat([global_features, local_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        return fused_features, global_features, local_features

# =========================== 对抗性特征解耦器 ===========================
class AdversarialFeatureDecoupler(nn.Module):
    """
    对抗性特征解耦器
    
    目标：将特征分解为：
    1. 任务相关特征（用于分类）
    2. 任务无关特征（领域共享）
    
    通过对抗训练增强特征的判别能力
    """
    def __init__(self, feature_dim, num_domains=2):
        super(AdversarialFeatureDecoupler, self).__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        
        # 特征分解器
        self.task_relevant_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 2)
        )
        
        self.task_irrelevant_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 2)
        )
        
        # 领域判别器（用于对抗训练）
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_domains)
        )
        
        # 梯度反转层参数
        self.lambda_reverse = 1.0
        
    def forward(self, features, reverse_gradient=False):
        """
        Args:
            features: [batch_size, feature_dim]
            reverse_gradient: 是否反转梯度（对抗训练）
        """
        # 分解特征
        task_relevant = self.task_relevant_encoder(features)
        task_irrelevant = self.task_irrelevant_encoder(features)
        
        # 域判别（用于对抗训练）
        if reverse_gradient:
            # 梯度反转
            reversed_irrelevant = GradientReversalFunction.apply(task_irrelevant, self.lambda_reverse)
            domain_logits = self.domain_discriminator(reversed_irrelevant)
        else:
            domain_logits = self.domain_discriminator(task_irrelevant)
        
        return task_relevant, task_irrelevant, domain_logits

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转函数"""
    @staticmethod
    def forward(ctx, x, lambda_reverse):
        ctx.lambda_reverse = lambda_reverse
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_reverse, None

# =========================== BACL优化的二分类模型 ===========================
class BACLBinaryQwen3ForSequenceClassification(nn.Module):
    """
    基于边界感知对比损失的二分类Qwen3模型
    
    架构：
    1. Qwen3基础模型
    2. 双通道特征提取器
    3. 对抗性特征解耦器
    4. BACL损失函数
    """
    def __init__(self, model_path, feature_dim=256, use_adversarial=True):
        super().__init__()
        
        # 加载基础模型
        self.base_model_wrapper = Qwen3ForSequenceClassification(model_path, num_labels=2)
        self.base_model = self.base_model_wrapper.base_model
        self.dropout = self.base_model_wrapper.dropout
        self.config = self.base_model_wrapper.config
        self.num_labels = 2
        self.hidden_size = self.base_model_wrapper.hidden_size
        
        # 双通道特征提取器
        self.feature_extractor = DualChannelFeatureExtractor(
            hidden_size=self.hidden_size, 
            feature_dim=feature_dim
        )
        
        # 对抗性特征解耦器
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.feature_decoupler = AdversarialFeatureDecoupler(
                feature_dim=feature_dim,
                num_domains=2
            )
            self.classifier = nn.Linear(feature_dim // 2, 2)  # 使用任务相关特征
        else:
            self.classifier = nn.Linear(feature_dim, 2)
        
        # BACL损失函数
        self.bacl_loss = BoundaryAwareContrastiveLoss(
            temperature=0.1,
            margin=0.5,
            alpha=1.0,
            beta=0.5
        )
        
        # 辅助损失权重
        self.adversarial_weight = 0.1
        self.ce_weight = 0.5
        self.bacl_weight = 0.5
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 从kwargs中提取参数
        position_ids = kwargs.pop('position_ids', None)
        past_key_values = kwargs.pop('past_key_values', None)
        inputs_embeds = kwargs.pop('inputs_embeds', None)
        use_cache = kwargs.pop('use_cache', None)
        output_attentions = kwargs.pop('output_attentions', None)
        output_hidden_states = kwargs.pop('output_hidden_states', None)
        return_dict = kwargs.pop('return_dict', None)
        
        # 获取基础模型输出
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
        
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 双通道特征提取
        fused_features, global_features, local_features = self.feature_extractor(
            hidden_states, attention_mask
        )
        
        # 对抗性特征解耦
        if self.use_adversarial:
            task_relevant, task_irrelevant, domain_logits = self.feature_decoupler(
                fused_features, reverse_gradient=self.training
            )
            # 使用任务相关特征进行分类
            logits = self.classifier(task_relevant)
        else:
            logits = self.classifier(fused_features)
            task_relevant = fused_features
            domain_logits = None
        
        # 计算损失
        loss = None
        loss_details = {}
        
        if labels is not None:
            # 1. 标准交叉熵损失
            ce_loss = F.cross_entropy(logits, labels)
            
            # 2. BACL损失
            bacl_loss, bacl_details = self.bacl_loss(task_relevant, logits, labels)
            
            # 3. 对抗损失（如果使用）
            adversarial_loss = 0.0
            if self.use_adversarial and domain_logits is not None:
                # 创建虚假域标签（促进域不变性）
                domain_labels = torch.randint(0, 2, (labels.size(0),), device=labels.device)
                adversarial_loss = F.cross_entropy(domain_logits, domain_labels)
            
            # 组合损失
            loss = (self.ce_weight * ce_loss + 
                   self.bacl_weight * bacl_loss + 
                   self.adversarial_weight * adversarial_loss)
            
            loss_details = {
                'total_loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'bacl_loss': bacl_loss.item(),
                'adversarial_loss': adversarial_loss.item() if isinstance(adversarial_loss, torch.Tensor) else adversarial_loss,
                **bacl_details
            }
        
        # 返回结果
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if kwargs.get('output_hidden_states', False) else None,
            attentions=outputs.attentions if kwargs.get('output_attentions', False) else None
        ), loss_details

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 二分类数据集类（与之前相同）
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
            try:
                original_label = int(label_text)
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

# 统计数据分布（与之前相同）
def analyze_binary_data_distribution(train_data_path):
    df = pd.read_excel(train_data_path)
    
    normal_count = 0
    violation_count = 0
    original_distribution = {}
    
    for label_text in df['extracted_label']:
        label_str = str(label_text).strip()
        original_distribution[label_str] = original_distribution.get(label_str, 0) + 1
        
        if label_str == '正常':
            normal_count += 1
        else:
            violation_count += 1
    
    total_samples = normal_count + violation_count
    
    data_info = {
        'normal_count': normal_count,
        'violation_count': violation_count,
        'total_samples': total_samples,
        'normal_ratio': normal_count / total_samples,
        'violation_ratio': violation_count / total_samples,
        'class_balance_ratio': min(normal_count, violation_count) / max(normal_count, violation_count),
        'original_distribution': original_distribution
    }
    
    return data_info

# BACL评估函数
def evaluate_bacl_binary(model, dataloader, device, rank, use_fp16=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    loss_details_sum = {}
    
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度推理
            with torch.amp.autocast('cuda', enabled=use_fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                result, loss_details = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = result.loss
            logits = result.logits
            
            total_loss += loss.item()
            
            # 累计损失详情
            for key, value in loss_details.items():
                if key not in loss_details_sum:
                    loss_details_sum[key] = 0
                loss_details_sum[key] += value
            
            # 获取预测和概率
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().float().numpy())
    
    # 计算平均损失详情
    num_batches = len(dataloader)
    for key in loss_details_sum:
        loss_details_sum[key] /= num_batches
    
    # 计算基础指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 计算关键业务指标
    normal_false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    violation_miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    violation_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    normal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 计算置信度统计
    all_probs = np.array(all_probs)
    violation_probs = all_probs[:, 1]
    max_probs = np.max(all_probs, axis=1)
    
    avg_confidence = np.mean(max_probs)
    confidence_std = np.std(max_probs)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        # 业务关键指标
        'normal_false_positive_rate': normal_false_positive_rate,
        'violation_miss_rate': violation_miss_rate,
        'violation_recall': violation_recall,
        'normal_specificity': normal_specificity,
        # 混淆矩阵详情
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'support': {
            'normal': int(tn + fp),
            'violation': int(tp + fn)
        },
        # 置信度统计
        'confidence_stats': {
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'high_confidence_ratio': np.mean(max_probs > 0.8),
            'low_confidence_ratio': np.mean(max_probs < 0.6)
        },
        # BACL损失详情
        'loss_details': loss_details_sum
    }

def setup_logging(rank, output_dir):
    """设置日志记录配置"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_file = os.path.join(output_dir, 'bacl_training.log')
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

def save_hyperparameters(args, output_dir, rank, data_info=None):
    """保存超参数配置到文件"""
    if rank == 0:
        import json
        import time
        
        hyperparams = {
            "experiment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_type": "binary_classification_bacl",
                "checkpoint": args.checkpoint,
                "train_data": args.train_data,
                "val_data": args.val_data,
                "output_dir": args.output_dir,
                "optimization": "boundary_aware_contrastive_learning"
            },
            
            "training_hyperparameters": {
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_steps": args.warmup_steps,
                "logging_steps": args.logging_steps,
                "eval_steps": args.eval_steps,
                "save_steps": args.save_steps
            },
            
            "bacl_config": {
                "feature_dim": args.feature_dim,
                "use_adversarial": args.use_adversarial,
                "temperature": args.temperature,
                "margin": args.margin,
                "alpha": args.alpha,
                "beta": args.beta,
                "adversarial_weight": args.adversarial_weight,
                "ce_weight": args.ce_weight,
                "bacl_weight": args.bacl_weight
            },
            
            "lora_config": {
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules
            },
            
            "optimizer_config": {
                "weight_decay": args.weight_decay,
                "adam_epsilon": args.adam_epsilon,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2
            },
            
            "scheduler_config": {
                "scheduler_type": args.scheduler_type,
                "min_lr": getattr(args, 'min_lr', None),
                "poly_power": getattr(args, 'poly_power', None)
            },
            
            "early_stopping": {
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_metric": args.early_stopping_metric
            },
            
            "system_config": {
                "fp16": args.fp16,
                "seed": args.seed,
                "gpu_ids": args.gpu_ids,
                "clear_cache": args.clear_cache
            }
        }
        
        if data_info is not None:
            hyperparams["data_analysis"] = data_info
        
        config_file = os.path.join(output_dir, 'bacl_hyperparameters.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(hyperparams, f, indent=2, ensure_ascii=False)
        
        txt_file = os.path.join(output_dir, 'bacl_hyperparameters.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BACL二分类训练超参数配置\n")
            f.write("=" * 60 + "\n")
            f.write(f"训练时间: {hyperparams['experiment_info']['timestamp']}\n\n")
            
            for section_name, section_data in hyperparams.items():
                if section_name == "experiment_info":
                    continue
                f.write(f"[{section_name.upper()}]\n")
                for key, value in section_data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        return config_file, txt_file
    return None, None

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """每个GPU上的训练工作进程"""
    actual_gpu_id = args.gpu_ids[rank] if args.gpu_ids else rank
    
    setup_distributed(rank, world_size)
    logger = setup_logging(rank, args.output_dir)
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"开始BACL二分类多GPU训练，使用 {world_size} 个GPU")
        logger.info(f"主进程运行在 rank {rank}，使用 GPU {actual_gpu_id}")
        logger.info(f"优化方法: 边界感知对比学习 (BACL)")
        logger.info(f"特征维度: {args.feature_dim}")
        logger.info(f"使用对抗训练: {args.use_adversarial}")
    
    torch.cuda.set_device(actual_gpu_id)
    set_seed(args.seed)
    device = torch.device(f'cuda:{actual_gpu_id}')
    
    if args.clear_cache:
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"已清理GPU {actual_gpu_id} 缓存")
    
    if is_main_process:
        logger.info(f"使用设备: {device}")
        logger.info(f"世界大小: {world_size}")
        logger.info(f"当前排名: {rank}")
        
        if torch.cuda.is_available():
            for i, gpu_id in enumerate(args.gpu_ids if args.gpu_ids else range(world_size)):
                if gpu_id < torch.cuda.device_count():
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    logger.info(f"GPU {gpu_id}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")
    
    # 分析数据分布
    if is_main_process:
        logger.info("分析二分类数据分布...")
        data_info = analyze_binary_data_distribution(args.train_data)
        logger.info(f"正常样本数: {data_info['normal_count']}")
        logger.info(f"违规样本数: {data_info['violation_count']}")
        logger.info(f"正常样本比例: {data_info['normal_ratio']:.4f}")
        logger.info(f"违规样本比例: {data_info['violation_ratio']:.4f}")
        logger.info(f"类别平衡比例: {data_info['class_balance_ratio']:.4f}")
    else:
        data_info = None
    
    # 保存超参数配置
    config_file, txt_file = save_hyperparameters(args, args.output_dir, rank, data_info)
    
    if is_main_process and config_file:
        logger.info(f"超参数已保存到: {config_file} 和 {txt_file}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载BACL二分类模型
    if is_main_process:
        logger.info("正在加载BACL二分类模型...")
        logger.info(f"特征维度: {args.feature_dim}")
        logger.info(f"对抗训练: {args.use_adversarial}")
        logger.info(f"BACL参数: temperature={args.temperature}, margin={args.margin}")
    
    model = BACLBinaryQwen3ForSequenceClassification(
        args.checkpoint,
        feature_dim=args.feature_dim,
        use_adversarial=args.use_adversarial
    )
    
    # 设置损失权重
    model.adversarial_weight = args.adversarial_weight
    model.ce_weight = args.ce_weight
    model.bacl_weight = args.bacl_weight
    
    # 设置BACL参数
    model.bacl_loss.temperature = args.temperature
    model.bacl_loss.margin = args.margin
    model.bacl_loss.alpha = args.alpha
    model.bacl_loss.beta = args.beta
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        modules_to_save=["classifier", "feature_extractor", "feature_decoupler"]
    )
    
    model = get_peft_model(model, lora_config)
    
    if is_main_process:
        logger.info("LoRA配置:")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 将模型移到设备并包装为DDP
    model.to(device)
    model = DDP(model, device_ids=[actual_gpu_id], output_device=actual_gpu_id, find_unused_parameters=True)
    
    # 准备数据集
    if is_main_process:
        logger.info("加载二分类数据集...")
    
    train_dataset = BinaryClassificationDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = BinaryClassificationDataset(args.val_data, tokenizer, args.max_length)
    
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
    
    if is_main_process:
        logger.info(f"训练配置:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  批次大小: {args.batch_size}")
        logger.info(f"  梯度累积步数: {args.gradient_accumulation_steps}")
        logger.info(f"  每步处理样本数: {samples_per_step}")
        logger.info(f"  每轮步数: {steps_per_epoch}")
        logger.info(f"  总训练步数: {total_steps}")
        logger.info(f"  训练轮数: {args.num_epochs}")
    
    # 学习率调度器
    if args.scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    elif args.scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    elif args.scheduler_type == "polynomial":
        from transformers import get_polynomial_decay_schedule_with_warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            power=args.poly_power
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
    best_val_loss = float('inf')
    best_violation_recall = 0
    global_step = 0
    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"===== 开始第 {epoch + 1}/{args.num_epochs} 轮BACL训练 =====")
        
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        train_loss_details = {}
        train_pbar = tqdm(train_loader, desc="Training", disable=(rank != 0))
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度训练
            with torch.amp.autocast('cuda', enabled=args.fp16, dtype=torch.float16 if model_dtype != torch.bfloat16 else torch.bfloat16):
                result, loss_details = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = result.loss / args.gradient_accumulation_steps
            
            if use_amp_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # 累计损失详情
            for key, value in loss_details.items():
                if key not in train_loss_details:
                    train_loss_details[key] = 0
                train_loss_details[key] += value
            
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
                    
                    # 计算平均损失详情
                    avg_loss_details = {k: v / (step + 1) for k, v in train_loss_details.items()}
                    
                    logger.info(f"Step {global_step} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    logger.info(f"  损失详情: CE={avg_loss_details.get('ce_loss', 0):.4f}, "
                              f"BACL={avg_loss_details.get('bacl_loss', 0):.4f}, "
                              f"Adv={avg_loss_details.get('adversarial_loss', 0):.4f}")
                    logger.info(f"  BACL细节: Contrastive={avg_loss_details.get('contrastive_loss', 0):.4f}, "
                              f"Boundary={avg_loss_details.get('boundary_loss', 0):.4f}, "
                              f"Separation={avg_loss_details.get('separation_loss', 0):.4f}")
                    
                    train_pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}', 
                        'lr': f'{current_lr:.2e}',
                        'bacl': f'{avg_loss_details.get("bacl_loss", 0):.4f}'
                    })
                
                # 评估
                if global_step % args.eval_steps == 0:
                    val_results = evaluate_bacl_binary(model, val_loader, device, rank)
                    
                    if is_main_process:
                        logger.info(f"验证集结果 - Step {global_step}:")
                        logger.info(f"  Loss: {val_results['loss']:.4f}")
                        logger.info(f"  准确率: {val_results['accuracy']:.4f}")
                        logger.info(f"  F1: {val_results['f1']:.4f}")
                        logger.info(f"  精确率: {val_results['precision']:.4f}")
                        logger.info(f"  召回率: {val_results['recall']:.4f}")
                        
                        # 关键业务指标
                        logger.info(f"\n  【关键业务指标】")
                        logger.info(f"    正常误报率: {val_results['normal_false_positive_rate']:.4f}")
                        logger.info(f"    违规漏报率: {val_results['violation_miss_rate']:.4f}")
                        logger.info(f"    违规召回率: {val_results['violation_recall']:.4f}")
                        logger.info(f"    正常特异性: {val_results['normal_specificity']:.4f}")
                        
                        # 置信度统计
                        conf_stats = val_results['confidence_stats']
                        logger.info(f"\n  【置信度统计】")
                        logger.info(f"    平均置信度: {conf_stats['avg_confidence']:.4f}")
                        logger.info(f"    置信度标准差: {conf_stats['confidence_std']:.4f}")
                        logger.info(f"    高置信度比例(>0.8): {conf_stats['high_confidence_ratio']:.4f}")
                        logger.info(f"    低置信度比例(<0.6): {conf_stats['low_confidence_ratio']:.4f}")
                        
                        # BACL损失详情
                        loss_details = val_results['loss_details']
                        logger.info(f"\n  【BACL损失详情】")
                        logger.info(f"    总损失: {loss_details.get('total_loss', 0):.4f}")
                        logger.info(f"    CE损失: {loss_details.get('ce_loss', 0):.4f}")
                        logger.info(f"    BACL损失: {loss_details.get('bacl_loss', 0):.4f}")
                        logger.info(f"    对抗损失: {loss_details.get('adversarial_loss', 0):.4f}")
                        logger.info(f"    对比损失: {loss_details.get('contrastive_loss', 0):.4f}")
                        logger.info(f"    边界损失: {loss_details.get('boundary_loss', 0):.4f}")
                        logger.info(f"    分离损失: {loss_details.get('separation_loss', 0):.4f}")
                        
                        # 混淆矩阵
                        cm = val_results['confusion_matrix']
                        logger.info(f"\n  【混淆矩阵】")
                        logger.info(f"    正常->正常: {cm['true_negative']}")
                        logger.info(f"    正常->违规: {cm['false_positive']}")
                        logger.info(f"    违规->正常: {cm['false_negative']}")
                        logger.info(f"    违规->违规: {cm['true_positive']}")
                        
                        # 保存最佳模型
                        improved = False
                        if args.early_stopping_metric == "f1" and val_results['f1'] > best_val_f1:
                            best_val_f1 = val_results['f1']
                            improved = True
                        elif args.early_stopping_metric == "loss" and val_results['loss'] < best_val_loss:
                            best_val_loss = val_results['loss']
                            improved = True
                        elif args.early_stopping_metric == "recall" and val_results['violation_recall'] > best_violation_recall:
                            best_violation_recall = val_results['violation_recall']
                            improved = True
                        
                        if improved:
                            patience_counter = 0
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(os.path.join(args.output_dir, 'best_bacl_model'))
                            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_bacl_model'))
                            logger.info(f"保存最佳BACL模型，F1: {val_results['f1']:.4f}, "
                                      f"违规召回率: {val_results['violation_recall']:.4f}")
                        else:
                            patience_counter += 1
                            logger.info(f"验证指标未改善，耐心计数: {patience_counter}/{early_stopping_patience}")
                            
                        # 早停检查
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"触发早停，在步骤 {global_step} 停止训练")
                            if torch.distributed.is_initialized():
                                torch.distributed.barrier()
                            cleanup_distributed()
                            return
                    
                    model.train()
                
                # 定期保存检查点
                if global_step % args.save_steps == 0 and is_main_process:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-bacl-{global_step}')
                    model_to_save.save_pretrained(checkpoint_dir)
                    logger.info(f"保存BACL检查点到: {checkpoint_dir}")
        
        # Epoch结束后的评估
        val_results = evaluate_bacl_binary(model, val_loader, device, rank)
        
        if is_main_process:
            logger.info(f"Epoch {epoch + 1} 验证集最终结果:")
            logger.info(f"  Loss: {val_results['loss']:.4f}")
            logger.info(f"  准确率: {val_results['accuracy']:.4f}")
            logger.info(f"  F1: {val_results['f1']:.4f}")
            logger.info(f"  关键指标:")
            logger.info(f"    正常误报率: {val_results['normal_false_positive_rate']:.4f}")
            logger.info(f"    违规漏报率: {val_results['violation_miss_rate']:.4f}")
            logger.info(f"    平均置信度: {val_results['confidence_stats']['avg_confidence']:.4f}")
            
            # 每个epoch结束后保存模型
            epoch_output_dir = os.path.join(args.output_dir, f'epoch-bacl-{epoch + 1}')
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            logger.info(f"已保存第 {epoch + 1} 个epoch的BACL模型到: {epoch_output_dir}")
    
    # 确保所有进程都完成训练后再清理
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="BACL二分类LoRA多GPU训练脚本")
    # 基础配置
    parser.add_argument('--checkpoint', type=str,
                       default="/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B")
    parser.add_argument('--train_data', type=str, default="./data/r789-b-50000_train.xlsx")
    parser.add_argument('--val_data', type=str, default="./data/r789-b-50000_val.xlsx")
    parser.add_argument('--test_data', type=str, default="./data/r789-b-50000_test.xlsx")
    parser.add_argument('--output_dir', type=str, default="./lora-bacl-binary-classification")
    
    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=12)  # 减小批次以适应更复杂的模型
    parser.add_argument('--learning_rate', type=float, default=1e-5)  # 更小的学习率用于精细调优
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6)  # 增加梯度累积
    
    # BACL特定参数
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='特征提取器输出维度')
    parser.add_argument('--use_adversarial', action='store_true', default=True,
                       help='是否使用对抗性特征解耦')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比学习温度参数')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='边界感知margin')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='对比损失权重')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='边界损失权重')
    parser.add_argument('--adversarial_weight', type=float, default=0.1,
                       help='对抗损失权重')
    parser.add_argument('--ce_weight', type=float, default=0.3,
                       help='交叉熵损失权重')
    parser.add_argument('--bacl_weight', type=float, default=0.7,
                       help='BACL损失权重')
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int, default=32,  # 增大秩以适应复杂模型
                       help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=64,  # 相应增大alpha
                       help='LoRA缩放因子')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout率')
    parser.add_argument('--target_modules', type=str, nargs='+',
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help='LoRA目标模块')
    
    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                       help='Adam优化器的epsilon参数')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                       help='Adam优化器的beta1参数')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                       help='Adam优化器的beta2参数')
    
    # 学习率调度
    parser.add_argument('--scheduler_type', type=str, default="linear",
                       choices=["linear", "cosine", "polynomial"],
                       help='学习率调度器类型')
    parser.add_argument('--warmup_steps', type=int, default=200)  # 增加warmup步数
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='最小学习率，用于cosine调度器')
    parser.add_argument('--poly_power', type=float, default=1.0,
                       help='多项式调度器的幂次')
    
    # 早停机制
    parser.add_argument('--early_stopping_patience', type=int, default=4,
                       help='早停耐心值')
    parser.add_argument('--early_stopping_metric', type=str, default="f1",
                       choices=["f1", "loss", "recall"],
                       help='早停监控指标')
    
    # 日志和保存
    parser.add_argument('--logging_steps', type=int, default=30)
    parser.add_argument('--eval_steps', type=int, default=80)
    parser.add_argument('--save_steps', type=int, default=150)
    
    # 系统配置
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5",
                       help='指定使用的GPU ID，用逗号分隔')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='指定使用的GPU数量，从GPU 0开始使用')
    parser.add_argument('--clear_cache', action='store_true', default=True,
                       help='训练前清理GPU缓存')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行多GPU训练")
    
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
    
    # 检查数据文件
    if not os.path.exists(args.train_data):
        raise ValueError(f"训练数据路径不存在: {args.train_data}")
    
    if not os.path.exists(args.val_data):
        raise ValueError(f"验证数据路径不存在: {args.val_data}")
    
    print(f"开始BACL二分类模型多GPU训练...")
    print(f"训练数据: {args.train_data}")
    print(f"验证数据: {args.val_data}")
    print(f"使用 {world_size} 个GPU进行训练")
    print(f"特征维度: {args.feature_dim}")
    print(f"对抗训练: {args.use_adversarial}")
    
    # 使用spawn方法启动多进程
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    
    print("BACL二分类模型训练完成！")

if __name__ == "__main__":
    main()