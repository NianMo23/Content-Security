#!/usr/bin/env python3
"""
数据集分割工具
根据标签列均匀地将数据随机分为训练、验证、测试集
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import os

def split_data_by_labels(input_file, label_column=2, train_ratio=0.4, val_ratio=0.1, test_ratio=0.5, random_seed=42):
    """
    根据标签列均匀地分割数据集
    
    Args:
        input_file: 输入Excel文件路径
        label_column: 标签列索引（从0开始）或列名
        train_ratio: 训练集比例
        val_ratio: 验证集比例  
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    print(f"正在读取数据文件: {input_file}")
    
    # 读取数据
    df = pd.read_excel(input_file)
    print(f"总数据量: {len(df)} 条")
    print(f"数据列名: {list(df.columns)}")
    
    # 获取标签列名
    if isinstance(label_column, int):
        label_col_name = df.columns[label_column]
    else:
        label_col_name = label_column
    
    print(f"使用标签列: {label_col_name}")
    
    # 检查比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"警告: 比例总和为 {total_ratio}，不等于1.0，将自动归一化")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    print(f"分割比例 - 训练集: {train_ratio:.1%}, 验证集: {val_ratio:.1%}, 测试集: {test_ratio:.1%}")
    
    # 设置随机种子
    np.random.seed(random_seed)
    print(f"随机种子: {random_seed}")
    
    # 按标签分组处理
    unique_labels = df[label_col_name].unique()
    print(f"发现 {len(unique_labels)} 个不同标签: {list(unique_labels)}")
    
    train_data = []
    val_data = []
    test_data = []
    
    for label in unique_labels:
        label_data = df[df[label_col_name] == label].copy()
        label_count = len(label_data)
        
        print(f"\n处理标签 '{label}': {label_count} 条数据")
        
        if label_count == 0:
            continue
        
        # 计算各集合的目标数量
        train_size = int(label_count * train_ratio)
        val_size = int(label_count * val_ratio)
        test_size = label_count - train_size - val_size  # 剩余的全部给测试集
        
        print(f"  目标分配 - 训练: {train_size}, 验证: {val_size}, 测试: {test_size}")
        
        # 随机打乱数据
        label_data = label_data.sample(frac=1, random_state=random_seed + hash(str(label)) % 1000).reset_index(drop=True)
        
        # 分割数据
        if label_count >= 3:  # 至少需要3条数据才能分成3个集合
            # 先分出训练集
            if train_size > 0:
                train_part = label_data.iloc[:train_size]
                remaining_data = label_data.iloc[train_size:]
            else:
                train_part = pd.DataFrame(columns=label_data.columns)
                remaining_data = label_data
            
            # 再从剩余数据中分出验证集和测试集
            if val_size > 0 and len(remaining_data) > 0:
                val_part = remaining_data.iloc[:val_size]
                test_part = remaining_data.iloc[val_size:]
            else:
                val_part = pd.DataFrame(columns=label_data.columns)
                test_part = remaining_data
            
        else:
            # 数据太少，按优先级分配
            if label_count >= 1:
                if test_size > 0:
                    test_part = label_data.iloc[:1]
                    train_part = pd.DataFrame(columns=label_data.columns)
                    val_part = pd.DataFrame(columns=label_data.columns)
                else:
                    train_part = label_data.iloc[:1]
                    val_part = pd.DataFrame(columns=label_data.columns)
                    test_part = pd.DataFrame(columns=label_data.columns)
            else:
                train_part = pd.DataFrame(columns=label_data.columns)
                val_part = pd.DataFrame(columns=label_data.columns)
                test_part = pd.DataFrame(columns=label_data.columns)
        
        # 添加到对应的集合
        if len(train_part) > 0:
            train_data.append(train_part)
        if len(val_part) > 0:
            val_data.append(val_part)
        if len(test_part) > 0:
            test_data.append(test_part)
        
        print(f"  实际分配 - 训练: {len(train_part)}, 验证: {len(val_part)}, 测试: {len(test_part)}")
    
    # 合并各个集合
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame(columns=df.columns)
    
    # 再次随机打乱每个集合
    if len(train_df) > 0:
        train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    if len(val_df) > 0:
        val_df = val_df.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
    if len(test_df) > 0:
        test_df = test_df.sample(frac=1, random_state=random_seed + 2).reset_index(drop=True)
    
    print(f"\n最终分割结果:")
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 检查每个标签在各集合中的分布
    print(f"\n各标签分布统计:")
    print(f"{'标签':<12} {'总数':<8} {'训练集':<8} {'验证集':<8} {'测试集':<8}")
    print("-" * 50)
    
    for label in unique_labels:
        total_count = len(df[df[label_col_name] == label])
        train_count = len(train_df[train_df[label_col_name] == label]) if len(train_df) > 0 else 0
        val_count = len(val_df[val_df[label_col_name] == label]) if len(val_df) > 0 else 0
        test_count = len(test_df[test_df[label_col_name] == label]) if len(test_df) > 0 else 0
        
        print(f"{str(label):<12} {total_count:<8} {train_count:<8} {val_count:<8} {test_count:<8}")
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(
        description="将数据按标签列均匀分割为训练、验证、测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（4:1:5比例分割）
  python split_data_by_labels.py -i data.xlsx
  
  # 指定标签列和比例
  python split_data_by_labels.py -i data.xlsx -c 2 --train-ratio 0.4 --val-ratio 0.1 --test-ratio 0.5
  
  # 指定随机种子和输出前缀
  python split_data_by_labels.py -i data.xlsx --seed 123 --output-prefix my_dataset
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入Excel文件路径')
    parser.add_argument('-c', '--label-column', default=2, type=int,
                       help='标签列索引（从0开始，默认: 2）')
    parser.add_argument('--train-ratio', type=float, default=0.4,
                       help='训练集比例（默认: 0.4）')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='验证集比例（默认: 0.1）')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                       help='测试集比例（默认: 0.5）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--output-prefix', default=None,
                       help='输出文件前缀（默认: 使用输入文件名）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        return
    
    # 生成输出文件名
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        # 使用输入文件名作为前缀
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_prefix = input_name
    
    train_file = f"{output_prefix}_train.xlsx"
    val_file = f"{output_prefix}_val.xlsx"
    test_file = f"{output_prefix}_test.xlsx"
    
    try:
        # 分割数据
        train_df, val_df, test_df = split_data_by_labels(
            args.input,
            args.label_column,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
        
        # 保存文件
        print(f"\n保存文件...")
        
        if len(train_df) > 0:
            train_df.to_excel(train_file, index=False)
            print(f"✓ 训练集已保存到: {train_file}")
        else:
            print("⚠ 训练集为空，跳过保存")
        
        if len(val_df) > 0:
            val_df.to_excel(val_file, index=False)
            print(f"✓ 验证集已保存到: {val_file}")
        else:
            print("⚠ 验证集为空，跳过保存")
        
        if len(test_df) > 0:
            test_df.to_excel(test_file, index=False)
            print(f"✓ 测试集已保存到: {test_file}")
        else:
            print("⚠ 测试集为空，跳过保存")
        
        print(f"\n数据分割完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()