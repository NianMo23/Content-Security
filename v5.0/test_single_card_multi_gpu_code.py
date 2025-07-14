#!/usr/bin/env python3
"""
单卡运行多GPU代码的便捷脚本
这个脚本使用多GPU的代码架构，但只在一张GPU卡上运行
"""

import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="在单张GPU上运行多GPU测试代码")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='指定使用的GPU ID (默认: 0)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小 (默认: 2)')
    parser.add_argument('--test_data', type=str, default="../data/r789-b-50000.xlsx",
                        help='测试数据路径')
    parser.add_argument('--lora_model', type=str, default="../simple-model-output/best_model",
                        help='LoRA模型路径')
    parser.add_argument('--max_length', type=int, default=256,
                        help='最大序列长度')
    
    args = parser.parse_args()
    
    print(f"🚀 在GPU {args.gpu_id} 上运行多GPU测试代码...")
    print(f"📊 批次大小: {args.batch_size}")
    print(f"📁 测试数据: {args.test_data}")
    print(f"🤖 LoRA模型: {args.lora_model}")
    print("-" * 50)
    
    # 构建命令
    cmd = [
        sys.executable,  # python
        "test_simple_multi_gpu.py",
        "--gpu_ids", str(args.gpu_id),
        "--batch_size", str(args.batch_size),
        "--test_data", args.test_data,
        "--lora_model", args.lora_model,
        "--max_length", str(args.max_length),
        "--fp16",
        "--clear_cache"
    ]
    
    try:
        # 运行命令
        result = subprocess.run(cmd, check=True, text=True)
        print("✅ 测试完成！")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
        return 130

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)