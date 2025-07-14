import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_mixed_dataset():
    """
    创建混合数据集：
    - balanced_train.xlsx 的 30% 数据
    - r789-b-50000.xlsx 的 100% 数据
    """
    
    # 数据文件路径
    balanced_train_path = "data/balanced_train.xlsx"
    r789_path = "data/r789-b-50000.xlsx"
    output_path = "data/mixed_train_dataset.xlsx"
    
    print("="*60)
    print("创建混合续训数据集")
    print("="*60)
    
    # 1. 读取 balanced_train.xlsx
    print("正在读取 balanced_train.xlsx...")
    if not os.path.exists(balanced_train_path):
        raise FileNotFoundError(f"文件不存在: {balanced_train_path}")
    
    balanced_data = pd.read_excel(balanced_train_path)
    print(f"balanced_train.xlsx 原始数据量: {len(balanced_data)}")
    
    # 显示数据结构
    print(f"balanced_train.xlsx 列名: {list(balanced_data.columns)}")
    if 'extracted_label' in balanced_data.columns:
        print("balanced_train.xlsx 标签分布:")
        print(balanced_data['extracted_label'].value_counts())
    
    # 2. 取 balanced_train.xlsx 的 30%
    print("\n正在提取 balanced_train.xlsx 的 30% 数据...")
    
    # 按标签分层采样，确保每个类别都保持比例
    if 'extracted_label' in balanced_data.columns:
        balanced_sample, _ = train_test_split(
            balanced_data, 
            test_size=0.7,  # 保留30%
            stratify=balanced_data['extracted_label'],
            random_state=42
        )
    else:
        # 如果没有标签列，就随机采样
        balanced_sample = balanced_data.sample(frac=0.3, random_state=42)
    
    print(f"从 balanced_train.xlsx 提取的数据量: {len(balanced_sample)}")
    if 'extracted_label' in balanced_sample.columns:
        print("提取数据的标签分布:")
        print(balanced_sample['extracted_label'].value_counts())
    
    # 3. 读取 r789-b-50000.xlsx
    print(f"\n正在读取 r789-b-50000.xlsx...")
    if not os.path.exists(r789_path):
        raise FileNotFoundError(f"文件不存在: {r789_path}")
    
    r789_data = pd.read_excel(r789_path)
    print(f"r789-b-50000.xlsx 数据量: {len(r789_data)}")
    
    # 显示数据结构
    print(f"r789-b-50000.xlsx 列名: {list(r789_data.columns)}")
    if 'extracted_label' in r789_data.columns:
        print("r789-b-50000.xlsx 标签分布:")
        print(r789_data['extracted_label'].value_counts())
    
    # 4. 检查列名一致性
    print("\n检查数据结构一致性...")
    balanced_cols = set(balanced_sample.columns)
    r789_cols = set(r789_data.columns)
    
    common_cols = balanced_cols.intersection(r789_cols)
    balanced_only = balanced_cols - r789_cols
    r789_only = r789_cols - balanced_cols
    
    print(f"共同列: {sorted(common_cols)}")
    if balanced_only:
        print(f"仅在 balanced_train 中的列: {sorted(balanced_only)}")
    if r789_only:
        print(f"仅在 r789 中的列: {sorted(r789_only)}")
    
    # 5. 对齐列结构
    print("\n对齐数据结构...")
    
    # 使用共同列
    if common_cols:
        use_cols = sorted(common_cols)
        print(f"使用列: {use_cols}")
        
        balanced_aligned = balanced_sample[use_cols].copy()
        r789_aligned = r789_data[use_cols].copy()
    else:
        print("警告: 没有共同列，将尝试使用所有列")
        # 如果没有共同列，尝试重新命名或处理
        balanced_aligned = balanced_sample.copy()
        r789_aligned = r789_data.copy()
    
    # 6. 合并数据
    print("\n合并数据...")
    
    # 添加数据来源标识
    balanced_aligned['data_source'] = 'balanced_train'
    r789_aligned['data_source'] = 'r789'
    
    # 合并
    mixed_data = pd.concat([balanced_aligned, r789_aligned], ignore_index=True)
    
    print(f"合并后总数据量: {len(mixed_data)}")
    print(f"  - 来自 balanced_train (30%): {len(balanced_aligned)}")
    print(f"  - 来自 r789 (100%): {len(r789_aligned)}")
    
    # 7. 显示最终标签分布
    if 'extracted_label' in mixed_data.columns:
        print("\n最终混合数据集标签分布:")
        final_distribution = mixed_data['extracted_label'].value_counts()
        print(final_distribution)
        
        print("\n按数据源的标签分布:")
        source_label_dist = mixed_data.groupby(['data_source', 'extracted_label']).size().unstack(fill_value=0)
        print(source_label_dist)
    
    # 8. 打乱数据
    print("\n打乱数据...")
    mixed_data = mixed_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 9. 保存混合数据集
    print(f"\n保存混合数据集到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mixed_data.to_excel(output_path, index=False)
    
    # 10. 验证保存的文件
    print("\n验证保存的文件...")
    saved_data = pd.read_excel(output_path)
    print(f"保存文件的数据量: {len(saved_data)}")
    
    print("="*60)
    print("混合数据集创建完成！")
    print("="*60)
    print(f"输出文件: {output_path}")
    print(f"总数据量: {len(mixed_data)}")
    print(f"数据来源: balanced_train (30%) + r789 (100%)")
    
    return output_path, mixed_data

if __name__ == "__main__":
    try:
        output_path, mixed_data = create_mixed_dataset()
        print(f"\n✅ 成功创建混合数据集: {output_path}")
    except Exception as e:
        print(f"\n❌ 创建混合数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()