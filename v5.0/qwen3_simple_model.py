import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class Qwen3ForClassification(nn.Module):
    """
    使用Qwen3模型进行分类的简单实现
    """
    def __init__(self, checkpoint):
        super().__init__()
        
        # 加载预训练的Qwen3模型
        print("正在加载Qwen3模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=None  # 不使用自动设备映射
        )
        
        # 保存配置信息
        self.config = self.model.config
        
        # 添加分类头
        self.classifier = nn.Linear(self.config.hidden_size, 6)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        前向传播
        """
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.hidden_states[-1]
        
        # 获取序列中最后一个token的隐藏状态
        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        # 确保pad_token_id有效，如果为None则使用eos_token_id
        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.eos_token_id  # 使用EOS作为替代
        
        # 计算每个序列的有效长度（非padding部分）
        attention_mask_sum = attention_mask.sum(dim=1) - 1  # 减1获取最后一个非padding位置
        batch_size = input_ids.shape[0]
        
        # 获取每个序列中最后一个token的隐藏状态
        last_hidden_states = hidden_states[torch.arange(batch_size), attention_mask_sum]
        
        # 通过分类头获取logits
        logits = self.classifier(last_hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # 返回结果
        return type('Output', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states
        })()
    
    def generate_classification(self, input_text, tokenizer):
        """
        生成分类结果
        """
        # 编码输入
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(next(self.parameters()).device)
        
        # 获取分类结果
        with torch.no_grad():
            outputs = self(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            # 获取预测的类别
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            
        # 类别映射
        class_names = ["正常", "歧视", "违法违规", "政治安全", "暴恐", "色情低俗"]
        return predicted_class, class_names[predicted_class]