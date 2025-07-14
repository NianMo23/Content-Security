from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl import Workbook

# 初始化vLLM模型 - 部分GPU并行配置
import os
import logging
import sys

# 禁用vLLM的详细日志输出和进度条
logging.getLogger("vllm").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# 禁用所有不必要的输出
import warnings
warnings.filterwarnings("ignore")

# 重定向stderr来禁用vLLM内部进度条
import contextlib
import io

class SuppressOutput:
    def __init__(self):
        self._original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = io.StringIO()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr

# ==================== GPU配置 ====================
# 设置要使用的GPU数量（根据需要修改）
USE_GPU_COUNT = 4  # 指定使用的GPU数量，例如：只使用2个GPU

# 设置可见的GPU设备（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # 只使用GPU 0和1

# 检测可用GPU数量
def get_gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 1

available_gpu_count = get_gpu_count()
actual_gpu_count = min(USE_GPU_COUNT, available_gpu_count)

print(f"系统可用GPU数量: {available_gpu_count}")
print(f"配置使用GPU数量: {actual_gpu_count}")

try:
    # 部分GPU并行配置
    llm = LLM(
        model="/home/users/sx_zhuzz/folder/llms-from-scratch/Qwen3",
        tensor_parallel_size=actual_gpu_count,  # 使用指定数量的GPU
        gpu_memory_utilization=0.7,  # 多卡情况下可以使用更高的内存利用率
        max_model_len=8192,  # 增加最大长度以支持更长的prompt
        dtype="half",  # 使用半精度
        trust_remote_code=True,  # 如果模型需要自定义代码
        enforce_eager=False  # 多卡时通常可以启用CUDA图
    )
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=32)
    print(f"成功使用 {actual_gpu_count} 个GPU加载32B模型！")
    
except Exception as e:
    print(f"指定GPU数量配置失败: {e}")
    print("尝试使用更保守的配置...")
    
    # 备用方案1：减少GPU数量
    try:
        backup_gpu_count = max(1, actual_gpu_count // 2)
        llm = LLM(
            model="/home/users/sx_zhuzz/folder/llms-from-scratch/Qwen3",
            tensor_parallel_size=backup_gpu_count,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            dtype="half",
            trust_remote_code=True,
            enforce_eager=True
        )
        sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=32)
        print(f"使用备用配置成功加载32B模型（{backup_gpu_count} GPU）！")
        
    except Exception as e2:
        print(f"备用配置也失败了: {e2}")
        print("尝试单卡配置...")
        
        # 备用方案2：单卡配置
        try:
            llm = LLM(
                model="/home/users/sx_zhuzz/folder/llms-from-scratch/Qwen3",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.7,
                max_model_len=4096,  # 增加最大长度
                dtype="half",
                trust_remote_code=True,
                enforce_eager=True
            )
            sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=32)
            print("使用单卡配置加载32B模型成功！")
        except Exception as e3:
            print(f"所有配置都失败了: {e3}")
            print("建议：")
            print("1. 检查GPU状态：nvidia-smi")
            print("2. 清理GPU内存：pkill -f python")
            print("3. 检查模型路径是否正确")
            print("4. 设置环境变量：export CUDA_VISIBLE_DEVICES=0,1")
            exit(1)

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

def classify_batch_32b(texts: list) -> list:
    """使用32B模型批量分类文本以提高效率"""
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
        # 抑制vLLM的内部进度条输出
        with SuppressOutput():
            outputs = llm.generate(prompts, sampling_params)
        
        results = []
        valid_labels = ['正常', '政治安全', '色情低俗', '违法违规', '暴恐', '歧视']
        
        for output in outputs:
            raw_output = output.outputs[0].text.strip()
            
            # 提取标签
            found_label = "正常"  # 默认标签
            for label in valid_labels:
                if label in raw_output:
                    found_label = label
                    break
            
            results.append(found_label)
        
        return results
        
    except Exception as e:
        print(f"32B模型批量调用错误: {e}")
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
    required_columns = ['原始文本']  # 至少需要原始文本列
    
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
    
    # 批量处理以提高效率 - 多卡优化
    # 4张卡，每张卡处理128样本，总批次大小为512
    batch_size = 32 * actual_gpu_count  # 根据GPU数量动态调整
    print(f"使用批处理大小: {batch_size} (每张GPU处理约{batch_size//actual_gpu_count}个样本)")
    print(f"预计单批次可充分利用{actual_gpu_count}张GPU的并行能力")
    
    # 存储结果
    verification_labels = []
    
    # 创建进度条
    progress_bar = tqdm(range(0, total, batch_size),
                       desc="32B模型检验进度",
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
        
        # 使用32B模型批量分类
        batch_verification_labels = classify_batch_32b(texts)
        
        # 确保批次结果长度与批次数据长度一致
        if len(batch_verification_labels) != len(batch_df):
            print(f"警告：批次 {i//batch_size + 1} 结果长度不匹配！")
            print(f"  批次数据长度: {len(batch_df)}")
            print(f"  批次结果长度: {len(batch_verification_labels)}")
            # 截断或填充到正确长度
            if len(batch_verification_labels) > len(batch_df):
                batch_verification_labels = batch_verification_labels[:len(batch_df)]
            else:
                batch_verification_labels.extend(['正常'] * (len(batch_df) - len(batch_verification_labels)))
        
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
    
    # 最终长度检查和修正
    print(f"\n=== 长度检查 ===")
    print(f"DataFrame长度: {len(df)}")
    print(f"验证标签长度: {len(verification_labels)}")
    
    if len(verification_labels) != len(df):
        print(f"⚠️ 长度不匹配！差异: {len(verification_labels) - len(df)}")
        
        if len(verification_labels) > len(df):
            # 如果标签过多，截断到正确长度
            verification_labels = verification_labels[:len(df)]
            print(f"✂️ 已截断验证标签到 {len(verification_labels)} 个")
        else:
            # 如果标签不足，用默认值填充
            missing_count = len(df) - len(verification_labels)
            verification_labels.extend(['正常'] * missing_count)
            print(f"📝 已填充 {missing_count} 个默认标签")
    
    # 再次确认长度匹配
    assert len(verification_labels) == len(df), f"长度仍不匹配: {len(verification_labels)} vs {len(df)}"
    print("✅ 长度检查通过")
    
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
    output_filename = f"zzz/enhanced_with_32b_verification_{timestamp}.csv"
    
    # 保存增强的CSV文件
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 增强的CSV文件已保存: {output_filename}")
    
    # 输出统计结果
    print(f"\n=== 32B模型检验结果统计 ===")
    print(f"总样本数: {total}")
    print(f"32B模型检验完成")
    
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
    print(f"\n=== 32B模型检验标签分布 ===")
    label_counts = df['32B模型检验标签'].value_counts()
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/total:.1%})")
    
    # 保存详细统计报告
    report_data = {
        'timestamp': timestamp,
        'total_samples': total,
        'disagreement_analysis': disagreement_analysis,
        'label_distribution': label_counts.to_dict()
    }
    
    if original_label_column and predicted_label_column:
        report_data['original_agreement_rate'] = original_agreement_rate
        report_data['predicted_agreement_rate'] = predicted_agreement_rate
    
    report_filename = f"zzz/32b_verification_report_{timestamp}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 文件保存总结 ===")
    print(f"📄 增强的CSV文件: {output_filename}")
    print(f"   - 原有列 + '32B模型检验标签'列")
    print(f"📋 统计报告JSON: {report_filename}")
    print(f"   - 详细的一致性和分布统计")

if __name__ == "__main__":
    main()