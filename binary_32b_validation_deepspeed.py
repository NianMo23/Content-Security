from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl import Workbook

# 初始化
import os
import logging
import sys

# 禁用详细日志输出
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 禁用所有不必要的输出
import warnings
warnings.filterwarnings("ignore")

# ==================== GPU配置 ====================
# 设置要使用的GPU数量（根据需要修改）
USE_GPU_COUNT = 4  # 指定使用的GPU数量

# 设置可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# 检测可用GPU数量
def get_gpu_count():
    try:
        return torch.cuda.device_count()
    except:
        return 1

available_gpu_count = get_gpu_count()
actual_gpu_count = min(USE_GPU_COUNT, available_gpu_count)

print(f"系统可用GPU数量: {available_gpu_count}")
print(f"配置使用GPU数量: {actual_gpu_count}")

# DeepSpeed配置
deepspeed_config = {
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 100,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "fp16": {
        "enabled": True,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 1000
        }
    }
}

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
ds_engine = None

def initialize_deepspeed_model():
    """初始化DeepSpeed模型"""
    global model, tokenizer, ds_engine
    
    model_path = "/home/users/sx_zhuzz/folder/llms-from-scratch/Qwen3"
    
    try:
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None  # DeepSpeed会管理设备分配
        )
        
        print("初始化DeepSpeed...")
        # 创建虚拟优化器参数（推理时不需要真实优化器）
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        
        # 初始化DeepSpeed引擎
        ds_engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=dummy_optimizer,
            config=deepspeed_config,
            dist_init_required=True
        )
        
        print(f"成功使用DeepSpeed初始化32B模型！")
        print(f"DeepSpeed引擎状态: {ds_engine.global_rank}/{ds_engine.world_size}")
        return True
        
    except Exception as e:
        print(f"DeepSpeed初始化失败: {e}")
        return False

def read_csv_data(file_path: str) -> pd.DataFrame:
    """读取CSV数据文件，自动检测编码格式"""
    
    # 尝试多种编码格式
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
    
    for encoding in encodings:
        try:
            print(f"尝试使用 {encoding} 编码读取CSV文件...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取CSV文件，共读取 {len(df)} 条数据")
            print(f"CSV文件列名: {list(df.columns)}")
            return df
                
        except UnicodeDecodeError:
            print(f"{encoding} 编码失败，尝试下一种编码...")
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时发生错误: {e}")
            continue
    
    # 如果所有编码都失败，尝试自动检测编码
    try:
        import chardet
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # 读取前10000字节用于检测
            detected = chardet.detect(raw_data)
            detected_encoding = detected['encoding']
            confidence = detected['confidence']
            
            print(f"检测到文件编码: {detected_encoding} (置信度: {confidence:.2f})")
            
            if detected_encoding and confidence > 0.7:
                try:
                    df = pd.read_csv(file_path, encoding=detected_encoding)
                    print(f"成功使用检测到的编码 {detected_encoding} 读取CSV文件，共读取 {len(df)} 条数据")
                    return df
                    
                except Exception as e:
                    print(f"使用检测到的编码 {detected_encoding} 失败: {e}")
            
    except ImportError:
        print("chardet库未安装，无法自动检测编码")
    except Exception as e:
        print(f"自动编码检测失败: {e}")
    
    print("所有编码尝试都失败了，请检查文件格式或手动指定正确的编码")
    return None

def classify_batch_deepspeed(texts: list) -> list:
    """使用DeepSpeed模型批量分类文本"""
    global ds_engine, tokenizer
    
    prompts = []
    for text in texts:
        prompt = f"""你是一个文本内容合规审核员，你的任务是根据平台的评判尺度规则判断用户的输入是否合规，仔细阅读下面的评判尺度规则。

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

##输出要求##
只能输出'正常'或'政治安全'或'色情低俗'或'违法违规'或'暴恐'或'歧视'这6个标签中的一个，不要输出任何其他额外的信息或解释。

【{text}】

请判断上述文本的标签："""
        prompts.append(prompt)
    
    try:
        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # 将输入移到第一个GPU（DeepSpeed会处理分布）
        input_ids = inputs.input_ids.to(f"cuda:{ds_engine.local_rank}")
        attention_mask = inputs.attention_mask.to(f"cuda:{ds_engine.local_rank}")
        
        # 使用DeepSpeed进行推理
        with torch.no_grad():
            outputs = ds_engine.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        results = []
        valid_labels = ['正常', '政治安全', '色情低俗', '违法违规', '暴恐', '歧视']
        
        for i, output in enumerate(outputs):
            # 去除输入部分，只保留生成的内容
            generated_ids = output[len(input_ids[i]):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 提取标签
            found_label = "正常"  # 默认标签
            for label in valid_labels:
                if label in generated_text:
                    found_label = label
                    break
            
            results.append(found_label)
        
        return results
        
    except Exception as e:
        print(f"DeepSpeed模型批量调用错误: {e}")
        return ["正常"] * len(texts)  # 默认返回正常

def analyze_disagreement(original_label, predicted_label, verification_label):
    """分析当原始标签和预测标签不一致时，32B模型的判断倾向"""
    if original_label == predicted_label:
        return "一致", "无需比较"
    
    # 原始标签和预测标签不一致的情况
    if verification_label == original_label:
        return "不一致", "支持原始标签"
    elif verification_label == predicted_label:
        return "不一致", "支持预测标签"
    else:
        return "不一致", "都不支持"

def main():
    # 初始化DeepSpeed
    if not initialize_deepspeed_model():
        print("DeepSpeed初始化失败，退出程序")
        return
    
    # 读取CSV数据文件
    data_file = './detailed_predictions_20250709_112207.csv'
    print("读取CSV数据文件...")
    
    try:
        df = read_csv_data(data_file)
        if df is None:
            print(f"无法读取文件 {data_file}")
            return
    except FileNotFoundError:
        print(f"文件 {data_file} 不存在")
        return
    
    # 检查必要的列是否存在
    # 推测可能的列名（考虑中英文和不同的命名方式）
    text_column = None
    original_label_column = None
    predicted_label_column = None
    
    for col in df.columns:
        if '原始文本' in col or '文本' in col or 'text' in col.lower():
            text_column = col
        elif '原始标签' in col or 'true' in col.lower() or 'original' in col.lower():
            original_label_column = col
        elif '预测标签' in col or 'pred' in col.lower() or 'predict' in col.lower():
            predicted_label_column = col
    
    if text_column is None:
        print(f"未找到文本列，可用列名: {list(df.columns)}")
        return
    
    print(f"使用文本列: {text_column}")
    print(f"原始标签列: {original_label_column}")
    print(f"预测标签列: {predicted_label_column}")
    
    total = len(df)
    print(f"开始处理 {total} 条数据...")
    
    # 批量处理 - DeepSpeed优化的批次大小
    # 根据GPU数量和内存情况调整，DeepSpeed可以处理更大的批次
    batch_size = 32 * actual_gpu_count  # DeepSpeed下的批次大小
    print(f"使用DeepSpeed批处理大小: {batch_size}")
    
    # 存储结果
    verification_labels = []
    
    # 创建进度条
    progress_bar = tqdm(range(0, total, batch_size),
                       desc="DeepSpeed 32B模型检验进度",
                       unit="batch",
                       ncols=120,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [已处理: {postfix}]')
    
    processed_count = 0
    import time
    start_time = time.time()
    
    for i in progress_bar:
        batch_start_time = time.time()
        batch_df = df.iloc[i:i+batch_size]
        texts = batch_df[text_column].astype(str).tolist()
        
        # 使用DeepSpeed模型批量分类
        batch_verification_labels = classify_batch_deepspeed(texts)
        verification_labels.extend(batch_verification_labels)
        
        processed_count += len(batch_df)
        batch_time = time.time() - batch_start_time
        avg_time_per_sample = batch_time / len(batch_df)
        throughput = len(batch_df) / batch_time
        
        # 更新进度条显示信息
        progress_bar.set_postfix({
            '样本': f'{processed_count}/{total}',
            '批次大小': len(batch_df),
            '吞吐量': f'{throughput:.1f}样本/秒',
            '批次耗时': f'{batch_time:.1f}秒'
        })
    
    progress_bar.close()
    
    # 添加32B模型检验标签列
    df['32B模型检验标签'] = verification_labels
    
    # 分析不一致情况（如果有原始标签和预测标签列）
    disagreement_analysis = {
        '支持原始标签': 0,
        '支持预测标签': 0, 
        '都不支持': 0,
        '总不一致数': 0
    }
    
    if original_label_column and predicted_label_column:
        for idx, row in df.iterrows():
            original = row[original_label_column]
            predicted = row[predicted_label_column]
            verification = row['32B模型检验标签']
            
            consistency, support = analyze_disagreement(original, predicted, verification)
            
            if consistency == "不一致":
                disagreement_analysis['总不一致数'] += 1
                if support == "支持原始标签":
                    disagreement_analysis['支持原始标签'] += 1
                elif support == "支持预测标签":
                    disagreement_analysis['支持预测标签'] += 1
                elif support == "都不支持":
                    disagreement_analysis['都不支持'] += 1
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"zzz/enhanced_with_deepspeed_verification_{timestamp}.csv"
    
    # 保存增强的CSV文件
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"✅ DeepSpeed增强的CSV文件已保存: {output_filename}")
    
    # 输出统计结果
    print(f"\n=== DeepSpeed 32B模型检验结果统计 ===")
    print(f"总样本数: {total}")
    print(f"DeepSpeed 32B模型检验完成")
    
    if original_label_column and predicted_label_column:
        print(f"\n=== 不一致情况分析 ===")
        print(f"原始标签与预测标签不一致的样本数: {disagreement_analysis['总不一致数']}")
        
        if disagreement_analysis['总不一致数'] > 0:
            print(f"在不一致的情况下：")
            print(f"  32B模型支持原始标签: {disagreement_analysis['支持原始标签']} ({disagreement_analysis['支持原始标签']/disagreement_analysis['总不一致数']:.1%})")
            print(f"  32B模型支持预测标签: {disagreement_analysis['支持预测标签']} ({disagreement_analysis['支持预测标签']/disagreement_analysis['总不一致数']:.1%})")
            print(f"  32B模型都不支持: {disagreement_analysis['都不支持']} ({disagreement_analysis['都不支持']/disagreement_analysis['总不一致数']:.1%})")
        
        # 计算32B模型与原始标签、预测标签的一致性
        if original_label_column:
            original_agreement = sum(1 for i in range(len(df)) if df.iloc[i][original_label_column] == df.iloc[i]['32B模型检验标签'])
            original_agreement_rate = original_agreement / total
            print(f"\n32B模型与原始标签一致率: {original_agreement_rate:.2%} ({original_agreement}/{total})")
        
        if predicted_label_column:
            predicted_agreement = sum(1 for i in range(len(df)) if df.iloc[i][predicted_label_column] == df.iloc[i]['32B模型检验标签'])
            predicted_agreement_rate = predicted_agreement / total
            print(f"32B模型与预测标签一致率: {predicted_agreement_rate:.2%} ({predicted_agreement}/{total})")
    
    # 统计各标签分布
    print(f"\n=== DeepSpeed 32B模型检验标签分布 ===")
    label_counts = df['32B模型检验标签'].value_counts()
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/total:.1%})")
    
    # 保存详细统计报告
    report_data = {
        'timestamp': timestamp,
        'total_samples': total,
        'disagreement_analysis': disagreement_analysis,
        'label_distribution': label_counts.to_dict(),
        'acceleration_method': 'DeepSpeed'
    }
    
    if original_label_column and predicted_label_column:
        report_data['original_agreement_rate'] = original_agreement_rate
        report_data['predicted_agreement_rate'] = predicted_agreement_rate
    
    report_filename = f"zzz/deepspeed_verification_report_{timestamp}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 文件保存总结 ===")
    print(f"📄 DeepSpeed增强的CSV文件: {output_filename}")
    print(f"   - 原有列 + '32B模型检验标签'列")
    print(f"📋 DeepSpeed统计报告JSON: {report_filename}")
    print(f"   - 详细的一致性和分布统计")

if __name__ == "__main__":
    # 初始化分布式环境
    import deepspeed
    deepspeed.init_distributed()
    
    main()