import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ViolationNormalContrastiveLoss(nn.Module):
    """
    专门针对违法违规(2)和正常(0)类别的对比损失
    通过对比学习增强这两个类别的区分度
    """
    def __init__(self, temperature=0.07, margin=1.0, lambda_contrast=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.lambda_contrast = lambda_contrast
        
    def forward(self, features, labels, base_loss):
        """
        features: [batch_size, hidden_dim] 特征向量
        labels: [batch_size] 标签
        base_loss: 原始的交叉熵损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 筛选出违法违规(2)和正常(0)的样本
        violation_mask = (labels == 2)
        normal_mask = (labels == 0)
        
        if not violation_mask.any() or not normal_mask.any():
            # 如果批次中没有这两类样本，直接返回原始损失
            return base_loss
        
        violation_features = features[violation_mask]
        normal_features = features[normal_mask]
        
        # L2标准化特征
        violation_features = F.normalize(violation_features, dim=1)
        normal_features = F.normalize(normal_features, dim=1)
        
        # 计算违法违规样本内部的相似度（应该高）
        violation_sim = torch.matmul(violation_features, violation_features.T) / self.temperature
        violation_sim.fill_diagonal_(float('-inf'))  # 去除自身相似度
        
        # 计算正常样本内部的相似度（应该高）
        normal_sim = torch.matmul(normal_features, normal_features.T) / self.temperature
        normal_sim.fill_diagonal_(float('-inf'))
        
        # 计算违法违规和正常之间的相似度（应该低）
        cross_sim = torch.matmul(violation_features, normal_features.T) / self.temperature
        
        # 对比损失
        contrast_loss = 0.0
        
        # 1. 违法违规样本应该彼此相似
        if violation_features.shape[0] > 1:
            violation_pos_sim = violation_sim.max(dim=1)[0]  # 最相似的违法违规样本
            violation_neg_sim = cross_sim.max(dim=1)[0]      # 最相似的正常样本
            violation_contrast = F.relu(violation_neg_sim - violation_pos_sim + self.margin)
            contrast_loss += violation_contrast.mean()
        
        # 2. 正常样本应该彼此相似
        if normal_features.shape[0] > 1:
            normal_pos_sim = normal_sim.max(dim=1)[0]        # 最相似的正常样本
            normal_neg_sim = cross_sim.max(dim=0)[0]         # 最相似的违法违规样本
            normal_contrast = F.relu(normal_neg_sim - normal_pos_sim + self.margin)
            contrast_loss += normal_contrast.mean()
        
        # 3. 违法违规和正常应该尽可能不相似
        separation_loss = F.relu(cross_sim + self.margin).mean()
        contrast_loss += separation_loss
        
        # 组合损失
        total_loss = base_loss + self.lambda_contrast * contrast_loss
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'contrast_loss': contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else contrast_loss,
            'separation_loss': separation_loss.item()
        }

class ViolationNormalFocalLoss(nn.Module):
    """
    专门针对违法违规和正常类别的Focal Loss
    对这两个类别给予特殊的权重和gamma值
    """
    def __init__(self, alpha_normal=1.0, alpha_violation=3.0, gamma_normal=2.0, gamma_violation=3.0):
        super().__init__()
        self.alpha_normal = alpha_normal
        self.alpha_violation = alpha_violation
        self.gamma_normal = gamma_normal
        self.gamma_violation = gamma_violation
        
    def forward(self, logits, labels):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 为不同类别设置不同的alpha和gamma
        alpha = torch.ones_like(labels, dtype=torch.float, device=labels.device)
        gamma = torch.ones_like(labels, dtype=torch.float, device=labels.device) * 2.0
        
        # 正常类别
        normal_mask = (labels == 0)
        alpha[normal_mask] = self.alpha_normal
        gamma[normal_mask] = self.gamma_normal
        
        # 违法违规类别
        violation_mask = (labels == 2)
        alpha[violation_mask] = self.alpha_violation
        gamma[violation_mask] = self.gamma_violation
        
        # 计算focal loss
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()

class ViolationNormalBoundaryLoss(nn.Module):
    """
    边界损失 - 专门增强违法违规和正常类别之间的决策边界
    """
    def __init__(self, margin=2.0, lambda_boundary=0.3):
        super().__init__()
        self.margin = margin
        self.lambda_boundary = lambda_boundary
        
    def forward(self, logits, labels, base_loss):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]
        base_loss: 原始损失
        """
        device = logits.device
        batch_size = logits.shape[0]
        
        # 获取正常(0)和违法违规(2)的logits
        normal_logits = logits[:, 0]  # 正常类别的logits
        violation_logits = logits[:, 2]  # 违法违规类别的logits
        
        # 计算边界损失
        boundary_loss = 0.0
        
        # 对于正常样本，应该满足：normal_logits > violation_logits + margin
        normal_mask = (labels == 0)
        if normal_mask.any():
            normal_boundary = F.relu(violation_logits[normal_mask] - normal_logits[normal_mask] + self.margin)
            boundary_loss += normal_boundary.mean()
        
        # 对于违法违规样本，应该满足：violation_logits > normal_logits + margin
        violation_mask = (labels == 2)
        if violation_mask.any():
            violation_boundary = F.relu(normal_logits[violation_mask] - violation_logits[violation_mask] + self.margin)
            boundary_loss += violation_boundary.mean()
        
        # 组合损失
        total_loss = base_loss + self.lambda_boundary * boundary_loss
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'boundary_loss': boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss
        }

class ViolationNormalTripletLoss(nn.Module):
    """
    三元组损失 - 确保正常样本更接近正常锚点，违法违规样本更接近违法违规锚点
    """
    def __init__(self, margin=1.0, lambda_triplet=0.2):
        super().__init__()
        self.margin = margin
        self.lambda_triplet = lambda_triplet
        
    def forward(self, features, labels, base_loss):
        """
        features: [batch_size, hidden_dim]
        labels: [batch_size]
        base_loss: 原始损失
        """
        device = features.device
        
        # 筛选正常和违法违规样本
        normal_mask = (labels == 0)
        violation_mask = (labels == 2)
        
        if not normal_mask.any() or not violation_mask.any():
            return base_loss, {'base_loss': base_loss.item(), 'triplet_loss': 0.0}
        
        normal_features = features[normal_mask]
        violation_features = features[violation_mask]
        
        # L2标准化
        normal_features = F.normalize(normal_features, dim=1)
        violation_features = F.normalize(violation_features, dim=1)
        
        # 计算中心点（质心）
        normal_center = normal_features.mean(dim=0, keepdim=True)
        violation_center = violation_features.mean(dim=0, keepdim=True)
        
        triplet_loss = 0.0
        
        # 对于正常样本：d(sample, normal_center) < d(sample, violation_center) + margin
        if normal_features.shape[0] > 0:
            normal_to_normal_dist = F.pairwise_distance(normal_features, normal_center.expand_as(normal_features))
            normal_to_violation_dist = F.pairwise_distance(normal_features, violation_center.expand_as(normal_features))
            normal_triplet = F.relu(normal_to_normal_dist - normal_to_violation_dist + self.margin)
            triplet_loss += normal_triplet.mean()
        
        # 对于违法违规样本：d(sample, violation_center) < d(sample, normal_center) + margin
        if violation_features.shape[0] > 0:
            violation_to_violation_dist = F.pairwise_distance(violation_features, violation_center.expand_as(violation_features))
            violation_to_normal_dist = F.pairwise_distance(violation_features, normal_center.expand_as(violation_features))
            violation_triplet = F.relu(violation_to_violation_dist - violation_to_normal_dist + self.margin)
            triplet_loss += violation_triplet.mean()
        
        # 组合损失
        total_loss = base_loss + self.lambda_triplet * triplet_loss
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'triplet_loss': triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else triplet_loss
        }

class CombinedViolationNormalLoss(nn.Module):
    """
    组合损失函数 - 结合多种策略来增强违法违规和正常类别的区分
    """
    def __init__(self, 
                 use_focal=True,
                 use_contrastive=True, 
                 use_boundary=True,
                 use_triplet=True,
                 focal_weight=1.0,
                 contrastive_weight=0.5,
                 boundary_weight=0.3,
                 triplet_weight=0.2):
        super().__init__()
        
        self.use_focal = use_focal
        self.use_contrastive = use_contrastive
        self.use_boundary = use_boundary
        self.use_triplet = use_triplet
        
        if use_focal:
            self.focal_loss = ViolationNormalFocalLoss()
        if use_contrastive:
            self.contrastive_loss = ViolationNormalContrastiveLoss(lambda_contrast=contrastive_weight)
        if use_boundary:
            self.boundary_loss = ViolationNormalBoundaryLoss(lambda_boundary=boundary_weight)
        if use_triplet:
            self.triplet_loss = ViolationNormalTripletLoss(lambda_triplet=triplet_weight)
            
        self.focal_weight = focal_weight
        
    def forward(self, logits, features, labels):
        """
        logits: [batch_size, num_classes]
        features: [batch_size, hidden_dim] 
        labels: [batch_size]
        """
        loss_dict = {}
        
        # 基础损失
        if self.use_focal:
            base_loss = self.focal_loss(logits, labels)
            loss_dict['focal_loss'] = base_loss.item()
        else:
            base_loss = F.cross_entropy(logits, labels)
            loss_dict['ce_loss'] = base_loss.item()
        
        total_loss = self.focal_weight * base_loss
        
        # 对比损失
        if self.use_contrastive and features is not None:
            contrastive_total, contrastive_dict = self.contrastive_loss(features, labels, base_loss)
            total_loss = contrastive_total
            loss_dict.update(contrastive_dict)
        
        # 边界损失
        if self.use_boundary:
            boundary_total, boundary_dict = self.boundary_loss(logits, labels, total_loss)
            total_loss = boundary_total
            loss_dict.update(boundary_dict)
        
        # 三元组损失
        if self.use_triplet and features is not None:
            triplet_total, triplet_dict = self.triplet_loss(features, labels, total_loss)
            total_loss = triplet_total
            loss_dict.update(triplet_dict)
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

class ConfidenceCalibrationLoss(nn.Module):
    """
    置信度校准损失 - 提高模型对违法违规和正常类别预测的置信度
    """
    def __init__(self, lambda_calibration=0.1, target_confidence=0.9):
        super().__init__()
        self.lambda_calibration = lambda_calibration
        self.target_confidence = target_confidence
        
    def forward(self, logits, labels, base_loss):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]
        base_loss: 原始损失
        """
        probs = F.softmax(logits, dim=1)
        
        # 获取正确类别的预测概率
        correct_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 只对违法违规和正常类别进行置信度校准
        target_mask = (labels == 0) | (labels == 2)
        
        if target_mask.any():
            target_correct_probs = correct_probs[target_mask]
            
            # 置信度损失：鼓励模型对这两个类别给出高置信度预测
            confidence_loss = F.mse_loss(target_correct_probs, 
                                       torch.full_like(target_correct_probs, self.target_confidence))
            
            total_loss = base_loss + self.lambda_calibration * confidence_loss
            
            return total_loss, {
                'base_loss': base_loss.item(),
                'confidence_loss': confidence_loss.item(),
                'avg_target_confidence': target_correct_probs.mean().item()
            }
        else:
            return base_loss, {'base_loss': base_loss.item(), 'confidence_loss': 0.0}