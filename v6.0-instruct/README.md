# 文本分类数据集转换工具

## 📋 概述

这个工具用于将Excel格式的文本分类数据转换为JSONL格式的问答数据集，适用于大模型微调和训练。

## 🗂️ 文件结构

```
zzz/data_elk/
├── xlsx_to_jsonl_converter.py  # 主转换器
├── convert_data.py             # 快速转换脚本
├── README.md                   # 使用说明
└── *.jsonl                     # 生成的数据集文件
```

## 🎯 功能特点

- **多格式支持**: 支持对话格式、Alpaca格式、简单格式三种输出
- **智能标签映射**: 自动将中文标签和数字标签转换为字母格式
- **详细评判规则**: 包含完整的内容安全评判尺度
- **批量处理**: 支持同时处理多个Excel文件
- **错误处理**: 完善的异常处理和日志输出

## 📊 数据格式

### 输入格式 (Excel)
Excel文件需要包含以下列：
- `text_cn`: 待分类的中文文本
- `extracted_label`: 文本标签（支持中文标签或数字标签）

### 输出格式 (JSONL)

#### 1. 对话格式 (conversation)
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "请阅读以下文本内容，并根据内容安全规则判断其所属类别。\n\n##评判尺度规则##\n...\n\n待分类文本：\n这是一段正常的文本内容\n\n问题：以下文本应该属于哪个类别？\nA. 正常\nB. 政治安全\nC. 歧视\nD. 违法违规\nE. 色情低俗\nF. 暴恐\n\n答案："
    },
    {
      "from": "gpt",
      "value": "A"
    }
  ]
}
```

#### 2. Alpaca格式 (alpaca)
```json
{
  "instruction": "请阅读以下文本内容，并根据内容安全规则判断其所属类别。\n\n##评判尺度规则##\n...",
  "input": "这是一段正常的文本内容",
  "output": "A"
}
```

#### 3. 简单格式 (simple)
```json
{
  "prompt": "请阅读以下文本内容，并根据内容安全规则判断其所属类别。\n\n##评判尺度规则##\n...\n\n待分类文本：\n这是一段正常的文本内容\n\n问题：以下文本应该属于哪个类别？\nA. 正常\nB. 政治安全\nC. 歧视\nD. 违法违规\nE. 色情低俗\nF. 暴恐\n\n答案：",
  "completion": "A"
}
```

## 🏷️ 标签映射

| 中文标签 | 数字标签 | 字母标签 | 说明 |
|---------|---------|---------|------|
| 正常 | 0 | A | 正常内容 |
| 政治安全 | 3 | B | 政治敏感内容 |
| 歧视 | 1 | C | 歧视性内容 |
| 违法违规 | 2 | D | 违法违规内容 |
| 色情低俗 | 5 | E | 色情低俗内容 |
| 暴恐 | 4 | F | 暴力恐怖内容 |

## 🚀 使用方法

### 方法1：使用快速转换脚本

```bash
cd zzz/data_elk
python convert_data.py
```

这将自动转换 `../data/` 目录下的所有Excel文件为三种格式的JSONL文件。

### 方法2：使用命令行工具

```bash
cd zzz/data_elk

# 转换单个文件为对话格式
python xlsx_to_jsonl_converter.py --format conversation --files balanced_train.xlsx

# 转换多个文件为Alpaca格式
python xlsx_to_jsonl_converter.py --format alpaca --files balanced_train.xlsx balanced_val.xlsx test.xlsx

# 指定输入输出目录
python xlsx_to_jsonl_converter.py \
    --input_dir ../data \
    --output_dir ./output \
    --format simple \
    --files balanced_train.xlsx
```

### 方法3：在Python代码中使用

```python
from xlsx_to_jsonl_converter import process_excel_file, save_to_jsonl

# 处理Excel文件
data = process_excel_file("../data/balanced_train.xlsx", "conversation")

# 保存为JSONL
save_to_jsonl(data, "train_conversation.jsonl")
```

## 📋 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | str | `../data` | 输入Excel文件所在目录 |
| `--output_dir` | str | `./` | 输出JSONL文件保存目录 |
| `--format` | str | `conversation` | 输出格式：conversation/alpaca/simple |
| `--files` | list | `[balanced_train.xlsx, balanced_val.xlsx, test.xlsx]` | 要转换的Excel文件列表 |

## 🔍 评判尺度规则

### 政治安全
违规范围：
1. 丑化、诽谤党和国家领导人，包括使用侮辱性绰号或影射语言
2. 鼓动颠覆国家政权、传播分裂国家、民族仇恨等内容
3. 宣扬境外反华势力、邪教组织，或煽动对抗政府言论
4. 涉及政治敏感话题、颠覆性言论

### 色情低俗
违规范围：
1. 涉及性行为描写、性暗示、裸露内容，或淫秽、低俗言辞
2. 推广、传播色情内容、链接、图文或视频
3. 含有挑逗、猥亵、色情等不当内容

### 违法违规
违规范围：
1. 宣传、引导参与赌博行为，包括赌球、线上赌博等
2. 提供博彩网站、邀请码、下注技巧等信息
3. 鼓励、教唆他人吸毒，或描写吸毒过程
4. 贩毒引流、介绍毒品获取渠道
5. 其他违法犯罪相关内容

### 暴恐
违规范围：
1. 鼓吹暴力极端思想、仇恨煽动内容
2. 威胁实施恐怖行动、制造恐慌或煽动他人恐袭
3. 涉及暴力、恐怖主义、极端主义等危险内容

### 歧视
违规范围：
1. 基于种族、性别、宗教、国籍等的歧视性言论
2. 煽动仇恨、偏见或歧视的内容
3. 侮辱、贬低特定群体的言论

### 正常
正常范围：
1. 生活、学习、工作、情感、新闻、娱乐等普通内容
2. 不涉及上述任何违规内容的正当表达
3. 客观、中性的信息分享和讨论

## 📝 示例输出

运行转换后，会生成如下文件：

```
zzz/data_elk/
├── balanced_train_conversation.jsonl
├── balanced_train_alpaca.jsonl
├── balanced_train_simple.jsonl
├── balanced_val_conversation.jsonl
├── balanced_val_alpaca.jsonl
├── balanced_val_simple.jsonl
├── test_conversation.jsonl
├── test_alpaca.jsonl
└── test_simple.jsonl
```

## 🛠️ 故障排除

### 常见问题

1. **文件不存在错误**
   ```
   文件不存在: ../data/balanced_train.xlsx
   ```
   **解决**: 确保Excel文件存在于指定路径

2. **缺少必要列错误**
   ```
   警告：缺少必要的列: ['text_cn', 'extracted_label']
   ```
   **解决**: 确保Excel文件包含 `text_cn` 和 `extracted_label` 列

3. **未知标签警告**
   ```
   警告：未知的字符串标签: 其他
   ```
   **解决**: 检查标签是否在支持的范围内，或更新标签映射

### 性能优化

- 对于大文件，建议分批处理
- 可以通过修改 `LABEL_MAPPING` 来支持更多标签类型
- 输出目录会自动创建，无需手动创建

## 🔧 自定义配置

### 添加新的标签映射

```python
# 在 xlsx_to_jsonl_converter.py 中修改
LABEL_MAPPING = {
    '正常': 'A',
    '政治安全': 'B', 
    '歧视': 'C',
    '违法违规': 'D',
    '色情低俗': 'E',
    '暴恐': 'F',
    '新标签': 'G'  # 添加新标签
}
```

### 自定义提示词模板

修改 `create_prompt_template()` 函数来自定义提示词格式。

## 📞 技术支持

如有问题，请检查：
1. Python环境是否正确安装pandas库
2. Excel文件格式是否正确
3. 文件路径是否存在
4. 权限是否足够

---

**注意**: 此工具专门用于文本内容安全分类任务，生成的数据集可直接用于大模型微调训练。