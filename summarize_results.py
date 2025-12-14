"""
AgentHarm 分类结果汇总脚本
读取每个类别的结果，生成汇总统计
"""
import pandas as pd
import json
import os
from typing import Dict, Any

# AgentHarm 的 8 个类别
CATEGORIES = [
    "Copyright",
    "Cybercrime",
    "Disinformation",
    "Drugs",
    "Fraud",
    "Harassment",
    "Hate",
    "Sexual",
]

MODEL_CONFIG = "deepseek-chat_deepseek-chat_deepseek-chat"


def summarize_category_results(category: str, data_type: str) -> Dict[str, Any]:
    """
    汇总单个类别的结果
    
    Args:
        category: 类别名称
        data_type: "harmful" 或 "benign"
    
    Returns:
        包含统计信息的字典
    """
    result_path = f"results/agentharm_{category}_{data_type}/{MODEL_CONFIG}.csv"
    
    if not os.path.exists(result_path):
        return {
            "category": category,
            "type": data_type,
            "status": "not_found",
            "total": 0,
            "safe": 0,
            "unsafe": 0,
            "error": 0,
        }
    
    try:
        df = pd.read_csv(result_path)
        
        total = len(df)
        safe_count = len(df[df["decision"] == "safe"])
        unsafe_count = len(df[df["decision"] == "unsafe"])
        error_count = len(df[df["decision"] == "error"])
        
        # 统计标签分布
        label_0_count = len(df[df["label"] == 0])  # harmful/unsafe
        label_1_count = len(df[df["label"] == 1])  # benign/safe
        
        # 计算准确率（正确拦截有害 + 正确放行无害）
        if data_type == "harmful":
            # harmful 数据应该被拦截 (decision=unsafe)
            correct = unsafe_count
            accuracy = correct / total * 100 if total > 0 else 0
        else:
            # benign 数据应该被放行 (decision=safe)
            correct = safe_count
            accuracy = correct / total * 100 if total > 0 else 0
        
        return {
            "category": category,
            "type": data_type,
            "status": "success",
            "total": total,
            "safe": safe_count,
            "unsafe": unsafe_count,
            "error": error_count,
            "label_0": label_0_count,
            "label_1": label_1_count,
            "correct": correct,
            "accuracy": accuracy,
        }
    
    except Exception as e:
        print(f"Error processing {category}_{data_type}: {str(e)}")
        return {
            "category": category,
            "type": data_type,
            "status": "error",
            "error_message": str(e),
        }


def main():
    print("=" * 80)
    print("AgentHarm 分类结果汇总")
    print("=" * 80)
    print()
    
    all_results = []
    
    # 处理每个类别的 harmful 和 benign 数据
    for category in CATEGORIES:
        print(f"\n{'=' * 40}")
        print(f"类别: {category}")
        print(f"{'=' * 40}")
        
        # Harmful 数据
        harmful_stats = summarize_category_results(category, "harmful")
        all_results.append(harmful_stats)
        
        if harmful_stats["status"] == "success":
            print(f"\nHarmful 数据:")
            print(f"  总数: {harmful_stats['total']}")
            print(f"  判定为 Safe: {harmful_stats['safe']} ({harmful_stats['safe']/harmful_stats['total']*100:.2f}%)")
            print(f"  判定为 Unsafe: {harmful_stats['unsafe']} ({harmful_stats['unsafe']/harmful_stats['total']*100:.2f}%)")
            print(f"  错误: {harmful_stats['error']}")
            print(f"  准确率（正确拦截）: {harmful_stats['accuracy']:.2f}%")
        else:
            print(f"\nHarmful 数据: {harmful_stats['status']}")
        
        # Benign 数据
        benign_stats = summarize_category_results(category, "benign")
        all_results.append(benign_stats)
        
        if benign_stats["status"] == "success":
            print(f"\nBenign 数据:")
            print(f"  总数: {benign_stats['total']}")
            print(f"  判定为 Safe: {benign_stats['safe']} ({benign_stats['safe']/benign_stats['total']*100:.2f}%)")
            print(f"  判定为 Unsafe: {benign_stats['unsafe']} ({benign_stats['unsafe']/benign_stats['total']*100:.2f}%)")
            print(f"  错误: {benign_stats['error']}")
            print(f"  准确率（正确放行）: {benign_stats['accuracy']:.2f}%")
        else:
            print(f"\nBenign 数据: {benign_stats['status']}")
    
    # 生成总体统计
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    
    successful_results = [r for r in all_results if r["status"] == "success"]
    
    if successful_results:
        total_samples = sum(r["total"] for r in successful_results)
        total_safe = sum(r["safe"] for r in successful_results)
        total_unsafe = sum(r["unsafe"] for r in successful_results)
        total_error = sum(r["error"] for r in successful_results)
        total_correct = sum(r["correct"] for r in successful_results)
        
        print(f"\n总样本数: {total_samples}")
        print(f"判定为 Safe: {total_safe} ({total_safe/total_samples*100:.2f}%)")
        print(f"判定为 Unsafe: {total_unsafe} ({total_unsafe/total_samples*100:.2f}%)")
        print(f"错误: {total_error} ({total_error/total_samples*100:.2f}%)")
        print(f"总体准确率: {total_correct/total_samples*100:.2f}%")
        
        # 按类别统计
        harmful_results = [r for r in successful_results if r["type"] == "harmful"]
        benign_results = [r for r in successful_results if r["type"] == "benign"]
        
        if harmful_results:
            harmful_total = sum(r["total"] for r in harmful_results)
            harmful_correct = sum(r["correct"] for r in harmful_results)
            print(f"\nHarmful 数据总体准确率: {harmful_correct/harmful_total*100:.2f}%")
        
        if benign_results:
            benign_total = sum(r["total"] for r in benign_results)
            benign_correct = sum(r["correct"] for r in benign_results)
            print(f"Benign 数据总体准确率: {benign_correct/benign_total*100:.2f}%")
    
    # 保存汇总结果到 JSON
    summary_file = "results/agentharm_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存到: {summary_file}")
    
    # 生成表格格式的汇总
    summary_df = pd.DataFrame(successful_results)
    if not summary_df.empty:
        summary_csv = "results/agentharm_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"CSV 格式汇总已保存到: {summary_csv}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
