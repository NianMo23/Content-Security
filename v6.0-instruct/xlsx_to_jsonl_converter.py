#!/usr/bin/env python3
"""
Excel数据转换为JSONL格式的文本分类问答数据集
将balanced_train.xlsx, balanced_val.xlsx, test.xlsx等文件转换为问答形式
"""

import pandas as pd
import json
import os
from pathlib import Path
import argparse
from typing import Dict, List, Any

# 标签映射
LABEL_MAPPING = {
    '正常': 'A',
    '政治安全': 'B', 
    '歧视': 'C',
    '违法违规': 'D',
    '色情低俗': 'E',
    '暴恐': 'F'
}

# 数字标签到字母的映射
NUMERIC_LABEL_MAPPING = {
    0: 'A',  # 正常
    1: 'C',  # 歧视
    2: 'D',  # 违法违规
    3: 'B',  # 政治安全
    4: 'F',  # 暴恐
    5: 'E'   # 色情低俗
}

def create_prompt_template() -> str:
    """创建文本分类的提示词模板"""
    return """请阅读以下文本内容，并根据内容安全规则判断其所属类别。

##评判尺度规则##

#政治安全#
违规范围：
1. 丑化、诽谤党和国家领导人，包括使用侮辱性绰号或影射语言。
2. 鼓动颠覆国家政权、传播分裂国家、民族仇恨等内容。
3. 宣扬境外反华势力、邪教组织，或煽动对抗政府言论。
4. 涉及政治敏感话题、颠覆性言论。

#色情低俗#
违规范围：
1. 涉及性行为描写、性暗示、裸露内容，或淫秽、低俗言辞。
2. 推广、传播色情内容、链接、图文或视频。
3. 含有挑逗、猥亵、色情等不当内容。

#违法违规#
违规范围：
1. 宣传、引导参与赌博行为，包括赌球、线上赌博等。
2. 提供博彩网站、邀请码、下注技巧等信息。
3. 鼓励、教唆他人吸毒，或描写吸毒过程。
4. 贩毒引流、介绍毒品获取渠道。
5. 其他违法犯罪相关内容。

#暴恐#
违规范围：
1. 鼓吹暴力极端思想、仇恨煽动内容。
2. 威胁实施恐怖行动、制造恐慌或煽动他人恐袭。
3. 涉及暴力、恐怖主义、极端主义等危险内容。

#歧视#
违规范围：
1. 基于种族、性别、宗教、国籍等的歧视性言论。
2. 煽动仇恨、偏见或歧视的内容。
3. 侮辱、贬低特定群体的言论。

#正常#
正常范围：
1. 生活、学习、工作、情感、新闻、娱乐等普通内容。
2. 不涉及上述任何违规内容的正当表达。
3. 客观、中性的信息分享和讨论。

##判断过程##
严格按照上述规则，一个标签一个标签判断。

待分类文本：
{text}

问题：以下文本应该属于哪个类别？
A. 正常
B. 政治安全
C. 歧视
D. 违法违规
E. 色情低俗
F. 暴恐

答案："""

def convert_label_to_letter(label: Any) -> str:
    """将标签转换为字母格式"""
    if isinstance(label, str):
        # 处理字符串标签
        label = label.strip()
        if label in LABEL_MAPPING:
            return LABEL_MAPPING[label]
        else:
            print(f"警告：未知的字符串标签: {label}")
            return 'A'  # 默认为正常
    elif isinstance(label, (int, float)):
        # 处理数字标签
        label = int(label)
        if label in NUMERIC_LABEL_MAPPING:
            return NUMERIC_LABEL_MAPPING[label]
        else:
            print(f"警告：未知的数字标签: {label}")
            return 'A'  # 默认为正常
    else:
        print(f"警告：无法处理的标签类型: {type(label)}, 值: {label}")
        return 'A'  # 默认为正常

def create_conversation_format(text: str, label: str) -> Dict[str, Any]:
    """创建对话格式的数据"""
    prompt_template = create_prompt_template()
    
    # 构建完整的问题
    question = prompt_template.format(text=text)
    
    # 构建答案
    answer = label
    
    return {
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt", 
                "value": answer
            }
        ]
    }

def create_alpaca_format(text: str, label: str) -> Dict[str, Any]:
    """创建Alpaca格式的数据"""
    instruction = """请阅读以下文本内容，并根据内容安全规则判断其所属类别。

##评判尺度规则##

#政治安全#
违规范围：
1. 丑化、诽谤党和国家领导人，包括使用侮辱性绰号或影射语言。
2. 鼓动颠覆国家政权、传播分裂国家、民族仇恨等内容。
3. 宣扬境外反华势力、邪教组织，或煽动对抗政府言论。
4. 涉及政治敏感话题、颠覆性言论。

#色情低俗#
违规范围：
1. 涉及性行为描写、性暗示、裸露内容，或淫秽、低俗言辞。
2. 推广、传播色情内容、链接、图文或视频。
3. 含有挑逗、猥亵、色情等不当内容。

#违法违规#
违规范围：
1. 宣传、引导参与赌博行为，包括赌球、线上赌博等。
2. 提供博彩网站、邀请码、下注技巧等信息。
3. 鼓励、教唆他人吸毒，或描写吸毒过程。
4. 贩毒引流、介绍毒品获取渠道。
5. 其他违法犯罪相关内容。

#暴恐#
违规范围：
1. 鼓吹暴力极端思想、仇恨煽动内容。
2. 威胁实施恐怖行动、制造恐慌或煽动他人恐袭。
3. 涉及暴力、恐怖主义、极端主义等危险内容。

#歧视#
违规范围：
1. 基于种族、性别、宗教、国籍等的歧视性言论。
2. 煽动仇恨、偏见或歧视的内容。
3. 侮辱、贬低特定群体的言论。

#正常#
正常范围：
1. 生活、学习、工作、情感、新闻、娱乐等普通内容。
2. 不涉及上述任何违规内容的正当表达。
3. 客观、中性的信息分享和讨论。

##判断过程##
严格按照上述规则，一个标签一个标签判断。

问题：以下文本应该属于哪个类别？
A. 正常
B. 政治安全
C. 歧视
D. 违法违规
E. 色情低俗
F. 暴恐"""
    
    return {
        "instruction": instruction,
        "input": text,
        "output": label
    }

def create_simple_format(text: str, label: str) -> Dict[str, Any]:
    """创建简单格式的数据"""
    prompt_template = create_prompt_template()
    
    return {
        "prompt": prompt_template.format(text=text),
        "completion": label
    }

def create_instruction_format(text: str, label: str) -> Dict[str, Any]:
    """创建instruction格式的数据"""
    instruction = """请阅读以下文本内容，并根据内容安全规则判断其所属类别。

##评判尺度规则##

#政治安全#
违规范围：
1. 丑化、诽谤党和国家领导人，包括使用侮辱性绰号或影射语言。
2. 鼓动颠覆国家政权、传播分裂国家、民族仇恨等内容。
3. 宣扬境外反华势力、邪教组织，或煽动对抗政府言论。
4. 涉及政治敏感话题、颠覆性言论。

#色情低俗#
违规范围：
1. 涉及性行为描写、性暗示、裸露内容，或淫秽、低俗言辞。
2. 推广、传播色情内容、链接、图文或视频。
3. 含有挑逗、猥亵、色情等不当内容。

#违法违规#
违规范围：
1. 宣传、引导参与赌博行为，包括赌球、线上赌博等。
2. 提供博彩网站、邀请码、下注技巧等信息。
3. 鼓励、教唆他人吸毒，或描写吸毒过程。
4. 贩毒引流、介绍毒品获取渠道。
5. 其他违法犯罪相关内容。

#暴恐#
违规范围：
1. 鼓吹暴力极端思想、仇恨煽动内容。
2. 威胁实施恐怖行动、制造恐慌或煽动他人恐袭。
3. 涉及暴力、恐怖主义、极端主义等危险内容。

#歧视#
违规范围：
1. 基于种族、性别、宗教、国籍等的歧视性言论。
2. 煽动仇恨、偏见或歧视的内容。
3. 侮辱、贬低特定群体的言论。

#正常#
正常范围：
1. 生活、学习、工作、情感、新闻、娱乐等普通内容。
2. 不涉及上述任何违规内容的正当表达。
3. 客观、中性的信息分享和讨论。

##判断过程##
严格按照上述规则，一个标签一个标签判断。

待分类文本：
{text}

问题：以下文本应该属于哪个类别？
A. 正常
B. 政治安全
C. 歧视
D. 违法违规
E. 色情低俗
F. 暴恐

答案：/no_think""".format(text=text)
    
    output = f"<think>\n\n</think>\n\n{label}"
    
    return {
        "instruction": instruction,
        "output": output
    }

def process_excel_file(file_path: str, output_format: str = "conversation") -> List[Dict[str, Any]]:
    """处理Excel文件，返回转换后的数据列表"""
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"成功读取 {file_path}，包含 {len(df)} 条数据")
        
        # 检查必要的列
        required_columns = ['text_cn', 'n_labels']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告：缺少必要的列: {missing_columns}")
            print(f"可用的列: {list(df.columns)}")
            return []
        
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # 获取文本和标签
                text = str(row['text_cn']).strip()
                label = row['n_labels']
                
                # 跳过空文本
                if not text or text.lower() in ['nan', 'none', '']:
                    continue
                
                # 转换标签为字母格式
                label_letter = convert_label_to_letter(label)
                
                # 根据输出格式创建数据
                if output_format == "conversation":
                    data_item = create_conversation_format(text, label_letter)
                elif output_format == "alpaca":
                    data_item = create_alpaca_format(text, label_letter)
                elif output_format == "simple":
                    data_item = create_simple_format(text, label_letter)
                elif output_format == "instruction":
                    data_item = create_instruction_format(text, label_letter)
                else:
                    raise ValueError(f"不支持的输出格式: {output_format}")
                
                processed_data.append(data_item)
                
            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")
                continue
        
        print(f"成功处理 {len(processed_data)} 条数据")
        return processed_data
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return []

def save_to_jsonl(data: List[Dict[str, Any]], output_path: str):
    """保存数据到JSONL文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
        print(f"成功保存 {len(data)} 条数据到 {output_path}")
    except Exception as e:
        print(f"保存文件 {output_path} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='将Excel文件转换为JSONL格式的问答数据集')
    parser.add_argument('--input_dir', type=str, default='./data', 
                        help='输入Excel文件所在目录')
    parser.add_argument('--output_dir', type=str, default='./data', 
                        help='输出JSONL文件保存目录')
    parser.add_argument('--format', type=str, default='instruction',
                        choices=['conversation', 'alpaca', 'simple', 'instruction'],
                        help='输出格式：conversation(对话格式), alpaca(Alpaca格式), simple(简单格式), instruction(指令格式)')
    parser.add_argument('--files', type=str, nargs='+', 
                        default=['balanced_train.xlsx', 'balanced_val.xlsx', 'test.xlsx'],
                        help='要转换的Excel文件列表')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个文件
    for filename in args.files:
        input_path = os.path.join(args.input_dir, filename)
        
        # 生成输出文件名
        base_name = Path(filename).stem
        output_filename = f"{base_name}_{args.format}.jsonl"
        output_path = output_dir / output_filename
        
        print(f"\n正在处理 {input_path} -> {output_path}")
        
        # 处理Excel文件
        processed_data = process_excel_file(input_path, args.format)
        
        if processed_data:
            # 保存到JSONL文件
            save_to_jsonl(processed_data, str(output_path))
        else:
            print(f"跳过 {filename}，没有有效数据")
    
    print("\n转换完成！")

if __name__ == "__main__":
    main()