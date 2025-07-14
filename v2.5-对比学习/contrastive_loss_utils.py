import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleContrastiveLoss(nn.Module):
    """
    简单的对比损失实现，专注于拉近同类样本、推远异类样本
    """
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim] L2归一化的特征
            labels: [batch_size] 标签
        Returns:
            对比损失值
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0., device=device)
        
        # 计算余弦相似度
        sim_matrix = torch.mm(features, features.T)
        
        # 创建正负样本掩码
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 排除对角线
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 应用温度缩放
        logits = sim_matrix / self.temperature
        
        # 计算log_prob
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 计算对比损失
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 计算正样本的平均log概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # 温度调整
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()
        
        return loss


class FocalContrastiveLoss(nn.Module):
    """
    带有焦点机制的对比损失，对困难样本给予更多关注
    """
    def __init__(self, temperature=0.1, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        
    def forward(self, features, labels):
        """
        计算focal对比损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0., device=device)
        
        # 计算相似度矩阵
        similarity = torch.mm(features, features.T) / self.temperature
        
        # 标签掩码
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        neg_mask = 1 - pos_mask
        
        # 移除对角线
        pos_mask.fill_diagonal_(0)
        
        # 计算损失
        loss = 0
        for i in range(batch_size):
            pos_sim = similarity[i][pos_mask[i] == 1]
            neg_sim = similarity[i][neg_mask[i] == 1]
            
            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue
            
            # 对每个正样本计算损失
            pos_loss = 0
            for pos in pos_sim:
                # softmax分母
                neg_exp = torch.exp(neg_sim).sum()
                pos_exp = torch.exp(pos)
                
                # focal weight
                pt = pos_exp / (pos_exp + neg_exp + 1e-12)
                focal_weight = (1 - pt) ** self.gamma
                
                # focal对比损失
                loss_i = -focal_weight * torch.log(pt + 1e-12)
                pos_loss += loss_i
            
            loss += pos_loss / len(pos_sim)
        
        return loss / batch_size


class MarginContrastiveLoss(nn.Module):
    """
    带边界的对比损失，确保正负样本间有明确的间隔
    """
    def __init__(self, margin=0.2, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        计算带边界的对比损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0., device=device)
        
        # 相似度矩阵
        similarity = torch.mm(features, features.T)
        
        # 标签掩码
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        neg_mask = 1 - pos_mask
        
        # 移除对角线
        pos_mask.fill_diagonal_(0)
        
        # 计算损失
        loss = 0
        num_valid = 0
        
        for i in range(batch_size):
            pos_sim = similarity[i][pos_mask[i] == 1]
            neg_sim = similarity[i][neg_mask[i] == 1]
            
            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue
            
            # 对每个正样本
            for p_sim in pos_sim:
                # 对每个负样本
                for n_sim in neg_sim:
                    # margin loss: max(0, margin + neg_sim - pos_sim)
                    loss += torch.relu(self.margin + n_sim - p_sim)
            
            num_valid += 1
        
        if num_valid > 0:
            loss = loss / (num_valid * len(pos_sim) * len(neg_sim))
        
        return loss


def test_losses():
    """测试各种对比损失函数"""
    # 创建测试数据
    batch_size = 8
    feature_dim = 128
    features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    # 测试各种损失
    losses = {
        'Simple': SimpleContrastiveLoss(),
        'Focal': FocalContrastiveLoss(),
        'Margin': MarginContrastiveLoss()
    }
    
    print("对比损失测试结果:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(features, labels)
        print(f"{name} Loss: {loss_value.item():.4f}")
        assert not torch.isnan(loss_value), f"{name} loss is NaN!"
        assert loss_value >= 0, f"{name} loss is negative!"
    
    print("所有损失函数测试通过!")


if __name__ == "__main__":
    test_losses()