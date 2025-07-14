from email.policy import strict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

print("当前工作目录:", os.getcwd())
checkpoint = "/home/users/sx_zhuzz/folder/LLaMA-Factory/mymodels/Qwen3-1.7B"

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")

# 查看模型结构
print("=== 模型结构 ===")
print(model)

# 查看关键的输出层参数
print("\n=== 关键输出层参数 ===")
for name, param in model.named_parameters():
    if any(key in name.lower() for key in ['lm_head', 'embed_out', 'output', 'norm']):
        print(f"Parameter name: {name}, shape: {param.size()}")

print(f"\nLoaded in {time.time() - start_time:.2f} seconds")

# 准备输入
chat = [
    {"role": "user", "content": "Hey"}
]

tool_prompt = tokenizer.apply_chat_template(
    chat,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)
tool_prompt = tool_prompt.to(model.device)

print(f"\n输入token IDs: {tool_prompt['input_ids']}")
print(f"输入tokens: {tokenizer.convert_ids_to_tokens(tool_prompt['input_ids'][0])}")

# ========== 方法1：使用完整模型获取词元向量 ==========
print("\n=== 方法1：使用完整模型 ===")

# 获取完整输出
outputs = model(**tool_prompt, output_hidden_states=True)

# 隐藏状态（最后一层Transformer的输出）
hidden_states = outputs.hidden_states
last_hidden_state = hidden_states[-1]  # [batch_size, seq_len, hidden_size]
print(f"最后层隐藏状态形状: {last_hidden_state.shape}")

# Logits（通过lm_head后的输出）
logits = outputs.logits  # [batch_size, seq_len, vocab_size]
print(f"Logits形状: {logits.shape}")

# ========== 方法2：直接访问基础模型获取纯隐藏状态 ==========
print("\n=== 方法2：使用基础模型 ===")

# 获取基础模型（不含lm_head）
base_model = model.model  # 对于大多数模型，基础模型在 .model 属性中

# 直接运行基础模型
base_outputs = base_model(**tool_prompt, output_hidden_states=True)
last_hidden_state_base = base_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
print(f"基础模型最后隐藏状态形状: {last_hidden_state_base.shape}")

# ========== 词元向量的位置 ==========
# 继续上面的代码...

print("\n=== 词元向量位置总结 ===")
print("1. 最后一层隐藏状态（用于分类任务的特征）:")
print(f"   - 形状: {last_hidden_state.shape}")
print(f"   - 位置: hidden_states[-1] 或 base_model.last_hidden_state")
print("\n2. Logits（词汇表上的分数）:")
print(f"   - 形状: {logits.shape}")
print(f"   - 位置: outputs.logits")

# ========== 为分类任务修改词元向量 ==========
print("\n=== 构建分类头 ===")

# 获取隐藏层维度
hidden_size = last_hidden_state.shape[-1]
print(f"隐藏层维度: {hidden_size}")

# 定义分类任务的类别数
num_classes = 3  # 例如：正面、中性、负面

# 创建分类头
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, hidden_states):
        # 使用最后一个token的隐藏状态进行分类
        # 也可以使用pooling（mean, max等）
        pooled_output = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        return logits

# 创建分类头实例
classification_head = ClassificationHead(hidden_size, num_classes).to(model.device)

# ========== 使用词元向量进行分类 ==========
print("\n=== 分类示例 ===")

# 方法1：使用最后一个token的隐藏状态
last_token_hidden = last_hidden_state[:, -1, :]  # [batch_size, hidden_size]
print(f"最后token隐藏状态形状: {last_token_hidden.shape}")

# 通过分类头获取分类logits
classification_logits = classification_head(last_hidden_state)
print(f"分类logits形状: {classification_logits.shape}")

# 获取分类概率
classification_probs = torch.softmax(classification_logits, dim=-1)
print(f"分类概率: {classification_probs}")

# ========== 完整的分类模型包装 ==========
print("\n=== 完整分类模型 ===")

class LLMForClassification(nn.Module):
    def __init__(self, base_model, num_classes, pooling_strategy='last'):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        
        # 获取隐藏层维度
        self.hidden_size = base_model.config.hidden_size
        
        # 分类头
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        # 获取基础模型输出
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 根据pooling策略获取句子表示
        if self.pooling_strategy == 'last':
            # 使用最后一个token
            pooled_output = hidden_states[:, -1, :]
        elif self.pooling_strategy == 'first':
            # 使用第一个token（类似BERT的[CLS]）
            pooled_output = hidden_states[:, 0, :]
        elif self.pooling_strategy == 'mean':
            # 使用所有token的平均值
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # 应用dropout和分类器
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# 创建分类模型实例
classification_model = LLMForClassification(model, num_classes=3, pooling_strategy='last')
classification_model.to(model.device)

# 测试分类模型
with torch.no_grad():
    class_logits = classification_model(
        input_ids=tool_prompt['input_ids'],
        attention_mask=tool_prompt.get('attention_mask')
    )
    class_probs = torch.softmax(class_logits, dim=-1)
    print(f"分类logits: {class_logits}")
    print(f"分类概率: {class_probs}")

# ========== 训练示例 ==========
print("\n=== 训练设置示例 ===")

# 冻结基础模型参数（可选）
for param in model.parameters():
    param.requires_grad = False

# 只训练分类头
for param in classification_model.classifier.parameters():
    param.requires_grad = True

# 设置优化器
optimizer = torch.optim.AdamW(classification_model.classifier.parameters(), lr=1e-4)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 模拟训练数据
batch_size = 2
seq_length = 10
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(model.device)
dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(model.device)

print(f"训练输入形状: {dummy_input_ids.shape}")
print(f"标签形状: {dummy_labels.shape}")

# 训练步骤示例
classification_model.train()
optimizer.zero_grad()

# 前向传播
logits = classification_model(input_ids=dummy_input_ids)
loss = criterion(logits, dummy_labels)

print(f"训练损失: {loss.item():.4f}")


# 反向传播（注意：这里只是示例，实际训练需要完整的数据集）
loss.backward()
optimizer.step()

print(f"训练损失: {loss.item():.4f}")

# ========== 实际使用示例 ==========
print("\n=== 实际使用示例 ===")

# 定义一个完整的分类pipeline
def classify_text(text, model, tokenizer, classification_head, device):
    """
    对输入文本进行分类
    """
    # 准备输入
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        # 使用分类头
        logits = classification_head(hidden_states)
        probs = torch.softmax(logits, dim=-1)
    
    return logits, probs

# 测试分类函数
test_texts = [
    "This movie is absolutely fantastic!",
    "The weather is okay today.",
    "I'm really disappointed with the service."
]

print("\n分类结果：")
for text in test_texts:
    logits, probs = classify_text(text, model, tokenizer, classification_head, model.device)
    predicted_class = torch.argmax(probs, dim=-1).item()
    print(f"\n文本: '{text}'")
    print(f"预测类别: {predicted_class}")
    print(f"概率分布: {probs.squeeze().tolist()}")

# ========== 提取和保存词元向量 ==========
print("\n=== 提取词元向量用于下游任务 ===")

def extract_embeddings(texts, model, tokenizer, device, pooling='mean'):
    """
    从文本中提取词元向量
    """
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            
            if pooling == 'mean':
                # 平均池化
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
                pooled = torch.sum(hidden_states * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            elif pooling == 'last':
                # 最后一个token
                pooled = hidden_states[:, -1, :]
            else:
                # 第一个token
                pooled = hidden_states[:, 0, :]
                
            embeddings.append(pooled.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

# 提取示例文本的嵌入
import numpy as np
embeddings = extract_embeddings(test_texts, model, tokenizer, model.device)
print(f"\n提取的嵌入形状: {embeddings.shape}")

# ========== 总结：词元向量的关键位置 ==========
print("\n=== 词元向量关键位置总结 ===")
print("""
1. 原始词嵌入层:
   - 位置: model.model.embed_tokens
   - 用途: 将token ID转换为初始嵌入
   
2. Transformer层的隐藏状态:
   - 位置: outputs.hidden_states[layer_idx]
   - 用途: 每一层的中间表示
   
3. 最后一层隐藏状态（最常用于分类）:
   - 位置: outputs.hidden_states[-1] 或 outputs.last_hidden_state
   - 形状: [batch_size, seq_len, hidden_size]
   - 用途: 作为分类任务的特征向量
   
4. LM头输出（词汇表logits）:
   - 位置: outputs.logits
   - 形状: [batch_size, seq_len, vocab_size]
   - 用途: 预测下一个token的概率分布
   
5. 获取方式:
   - 完整模型: model(**inputs, output_hidden_states=True)
   - 基础模型: model.model(**inputs, output_hidden_states=True)
""")

