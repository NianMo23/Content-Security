#!/usr/bin/env python3
"""
数据转换示例脚本
快速将Excel文件转换为不同格式的JSONL数据集
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from xlsx_to_jsonl_converter import process_excel_file, save_to_jsonl

def main():
    """主函数：演示如何使用转换器"""
    
    print("🚀 开始数据转换任务...")
    print("=" * 50)
    
    # 配置
    input_dir = "../data"
    output_dir = "./"
    
    # 要转换的文件列表
    files_to_convert = [
        "balanced_train.xlsx",
        "balanced_val.xlsx", 
        "test.xlsx"
    ]
    
    # 输出格式
    formats = ["conversation", "alpaca", "simple"]
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换每个文件的每种格式
    for filename in files_to_convert:
        input_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"⚠️ 文件不存在，跳过: {filename}")
            continue
            
        print(f"\n📁 处理文件: {filename}")
        
        for format_type in formats:
            print(f"  🔄 转换格式: {format_type}")
            
            # 生成输出文件名
            base_name = Path(filename).stem
            output_filename = f"{base_name}_{format_type}.jsonl"
            output_path = os.path.join(output_dir, output_filename)
            
            # 处理数据
            processed_data = process_excel_file(input_path, format_type)
            
            if processed_data:
                # 保存数据
                save_to_jsonl(processed_data, output_path)
                print(f"    ✅ 成功生成: {output_filename}")
            else:
                print(f"    ❌ 处理失败: {filename}")
    
    print("\n🎉 数据转换完成！")
    print("\n生成的文件列表:")
    
    # 显示生成的文件
    jsonl_files = list(Path(output_dir).glob("*.jsonl"))
    for file_path in sorted(jsonl_files):
        size = file_path.stat().st_size
        print(f"  📄 {file_path.name} ({size:,} bytes)")

if __name__ == "__main__":
    main()