#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比测试脚本
比较原始模型和优化模型的性能差异
"""

import argparse
import os
import json
from datetime import datetime
import subprocess

def run_test(model_path, test_data, output_prefix, gpu_ids="0,1,2,3"):
    """运行单个模型的测试"""
    cmd = [
        "python", "test_saved_lora_multi_gpu.py",
        "--lora_model", model_path,
        "--test_data", test_data,
        "--gpu_ids", gpu_ids,
        "--batch_size", "18",
        "--fp16"
    ]
    
    print(f"正在测试模型: {model_path}")
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"✅ {output_prefix}模型测试完成")
            return True
        else:
            print(f"❌ {output_prefix}模型测试失败")
            print(f"错误信息: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {output_prefix}模型测试超时")
        return False
    except Exception as e:
        print(f"❌ {output_prefix}模型测试异常: {e}")
        return False

def compare_results(original_results_dir, optimized_results_dir):
    """比较两个模型的结果"""
    
    # 查找最新的结果文件
    def find_latest_result(results_dir):
        if not os.path.exists(results_dir):
            return None
        
        files = [f for f in os.listdir(results_dir) if f.startswith('multi_gpu_test_results_with_analysis_') and f.endswith('.json')]
        if not files:
            return None
        
        # 按时间戳排序，取最新的
        files.sort(reverse=True)
        return os.path.join(results_dir, files[0])
    
    original_file = find_latest_result(original_results_dir)
    optimized_file = find_latest_result(optimized_results_dir)
    
    if not original_file or not optimized_file:
        print("❌ 无法找到对比结果文件")
        print(f"原始模型结果文件: {original_file}")
        print(f"优化模型结果文件: {optimized_file}")
        return
    
    # 加载结果
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(optimized_file, 'r', encoding='utf-8') as f:
        optimized_data = json.load(f)
    
    # 提取关键指标
    orig_results = original_data['results']
    opti_results = optimized_data['results']
    
    print("\n" + "="*80)
    print("🎯 模型性能对比报告")
    print("="*80)
    
    # 基础指标对比
    print("\n📊 基础性能指标对比:")
    print("-" * 50)
    metrics = [
        ('6分类准确率', 'accuracy'),
        ('F1分数', 'f1'),
        ('精确率', 'precision'),
        ('召回率', 'recall'),
        ('二分类准确率', 'binary_accuracy'),
        ('损失值', 'loss')
    ]
    
    for metric_name, metric_key in metrics:
        orig_val = orig_results[metric_key]
        opti_val = opti_results[metric_key]
        improvement = opti_val - orig_val
        improvement_pct = (improvement / orig_val) * 100 if orig_val != 0 else 0
        
        if metric_key == 'loss':
            # 对于损失，越小越好
            improvement = orig_val - opti_val
            improvement_pct = (improvement / orig_val) * 100 if orig_val != 0 else 0
            symbol = "📉" if improvement > 0 else "📈"
        else:
            symbol = "📈" if improvement > 0 else "📉"
        
        print(f"{metric_name:12} | 原始: {orig_val:.4f} | 优化: {opti_val:.4f} | "
              f"变化: {improvement:+.4f} ({improvement_pct:+.2f}%) {symbol}")
    
    # 漏报误报对比
    print("\n🚨 关键业务指标对比:")
    print("-" * 50)
    
    # 正常类误报率
    orig_fp_rate = orig_results['false_positive_analysis']['normal_false_positive_rate']
    opti_fp_rate = opti_results['false_positive_analysis']['normal_false_positive_rate']
    fp_improvement = orig_fp_rate - opti_fp_rate  # 误报率越小越好
    fp_improvement_pct = (fp_improvement / orig_fp_rate) * 100 if orig_fp_rate != 0 else 0
    
    print(f"正常内容误报率 | 原始: {orig_fp_rate:.4f} | 优化: {opti_fp_rate:.4f} | "
          f"降低: {fp_improvement:.4f} ({fp_improvement_pct:+.2f}%) {'✅' if fp_improvement > 0 else '❌'}")
    
    # 总漏报率
    orig_miss_rate = orig_results['miss_report_analysis']['total_miss_rate']
    opti_miss_rate = opti_results['miss_report_analysis']['total_miss_rate']
    miss_improvement = orig_miss_rate - opti_miss_rate  # 漏报率越小越好
    miss_improvement_pct = (miss_improvement / orig_miss_rate) * 100 if orig_miss_rate != 0 else 0
    
    print(f"异常内容漏报率 | 原始: {orig_miss_rate:.4f} | 优化: {opti_miss_rate:.4f} | "
          f"降低: {miss_improvement:.4f} ({miss_improvement_pct:+.2f}%) {'✅' if miss_improvement > 0 else '❌'}")
    
    # 各类别漏报率详细对比
    print("\n📋 各异常类别漏报率对比:")
    print("-" * 60)
    class_names = ['歧视', '违法违规', '政治安全', '暴恐', '色情低俗']
    
    for class_name in class_names:
        if class_name in orig_results['miss_report_analysis']['per_class_miss_rate'] and \
           class_name in opti_results['miss_report_analysis']['per_class_miss_rate']:
            
            orig_class_miss = orig_results['miss_report_analysis']['per_class_miss_rate'][class_name]['rate']
            opti_class_miss = opti_results['miss_report_analysis']['per_class_miss_rate'][class_name]['rate']
            class_improvement = orig_class_miss - opti_class_miss
            class_improvement_pct = (class_improvement / orig_class_miss) * 100 if orig_class_miss != 0 else 0
            
            print(f"{class_name:8} | 原始: {orig_class_miss:.4f} | 优化: {opti_class_miss:.4f} | "
                  f"变化: {class_improvement:+.4f} ({class_improvement_pct:+.2f}%) {'✅' if class_improvement > 0 else '❌'}")
    
    # 总结
    print("\n🎯 优化效果总结:")
    print("-" * 50)
    
    # 计算总体改善指标
    accuracy_improved = opti_results['accuracy'] > orig_results['accuracy']
    fp_reduced = opti_fp_rate < orig_fp_rate
    miss_reduced = opti_miss_rate < orig_miss_rate
    
    improvements = [accuracy_improved, fp_reduced, miss_reduced]
    improvement_count = sum(improvements)
    
    if improvement_count >= 2:
        print("🌟 优化效果显著！主要指标均有改善")
    elif improvement_count == 1:
        print("⚡ 优化有一定效果，但仍有提升空间")
    else:
        print("⚠️  优化效果有限，建议调整参数重新训练")
    
    # 保存对比报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"model_comparison_report_{timestamp}.json"
    
    comparison_report = {
        "timestamp": timestamp,
        "original_model": original_file,
        "optimized_model": optimized_file,
        "comparison": {
            "basic_metrics": {
                metric_key: {
                    "original": orig_results[metric_key],
                    "optimized": opti_results[metric_key],
                    "improvement": opti_results[metric_key] - orig_results[metric_key]
                } for _, metric_key in metrics
            },
            "business_metrics": {
                "normal_false_positive_rate": {
                    "original": orig_fp_rate,
                    "optimized": opti_fp_rate,
                    "improvement": fp_improvement
                },
                "total_miss_rate": {
                    "original": orig_miss_rate,
                    "optimized": opti_miss_rate,
                    "improvement": miss_improvement
                }
            },
            "summary": {
                "accuracy_improved": accuracy_improved,
                "false_positive_reduced": fp_reduced,
                "miss_rate_reduced": miss_reduced,
                "improvement_score": improvement_count
            }
        }
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=4, ensure_ascii=False)
    
    print(f"\n📄 详细对比报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="比较原始模型和优化模型的性能")
    parser.add_argument('--original_model', type=str, required=True,
                        help="原始模型路径")
    parser.add_argument('--optimized_model', type=str, required=True,
                        help="优化模型路径")
    parser.add_argument('--test_data', type=str, default="./data/r789-b-50000_test.xlsx",
                        help="测试数据路径")
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3",
                        help="使用的GPU ID")
    parser.add_argument('--skip_test', action='store_true',
                        help="跳过模型测试，直接比较已有结果")
    
    args = parser.parse_args()
    
    print("🚀 开始模型性能对比测试")
    print(f"原始模型: {args.original_model}")
    print(f"优化模型: {args.optimized_model}")
    print(f"测试数据: {args.test_data}")
    print(f"使用GPU: {args.gpu_ids}")
    
    if not args.skip_test:
        print("\n📋 开始运行模型测试...")
        
        # 测试原始模型
        success1 = run_test(args.original_model, args.test_data, "原始", args.gpu_ids)
        
        # 测试优化模型
        success2 = run_test(args.optimized_model, args.test_data, "优化", args.gpu_ids)
        
        if not (success1 and success2):
            print("❌ 模型测试失败，无法进行对比")
            return
        
        print("✅ 所有模型测试完成")
    
    # 比较结果
    print("\n📊 开始分析对比结果...")
    original_results_dir = os.path.dirname(args.original_model)
    optimized_results_dir = os.path.dirname(args.optimized_model)
    
    compare_results(original_results_dir, optimized_results_dir)

if __name__ == "__main__":
    main()