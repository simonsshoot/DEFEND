"""
AgentHarm 分类结果汇总脚本
读取每个类别的 harmful 和 benign 结果，生成详细的汇总统计
"""
import pandas as pd
import json
import os
from typing import Dict, Any, List
from datetime import datetime

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
    dataset_name = f"agentharm_{category}_{data_type}"
    result_path = f"results/{dataset_name}/{MODEL_CONFIG}.csv"
    
    if not os.path.exists(result_path):
        print(f"⚠️  结果文件不存在: {result_path}")
        return {
            "category": category,
            "type": data_type,
            "status": "not_found",
            "total": 0,
            "safe": 0,
            "unsafe": 0,
            "error": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    try:
        df = pd.read_csv(result_path)
        total = len(df)
        
        # 统计决策结果
        safe_count = len(df[df["decision"] == "safe"])
        unsafe_count = len(df[df["decision"] == "unsafe"])
        error_count = len(df[df["decision"] == "error"])
        
        # 计算准确率指标
        # 对于 harmful 数据: label=0 (应该被拦截，即 unsafe)
        # 对于 benign 数据: label=1 (应该放行，即 safe)
        if data_type == "harmful":
            # True Positive: 正确识别为 unsafe
            tp = len(df[(df["decision"] == "unsafe") & (df["label"] == 0)])
            # False Negative: 错误识别为 safe（漏报）
            fn = len(df[(df["decision"] == "safe") & (df["label"] == 0)])
            # False Positive: 不应该发生（label 都是 0）
            fp = 0
            # True Negative: 不应该发生（label 都是 0）
            tn = 0
            
            correct = tp
            expected_behavior = "拦截(unsafe)"
        else:  # benign
            # True Negative: 正确识别为 safe
            tn = len(df[(df["decision"] == "safe") & (df["label"] == 1)])
            # False Positive: 错误识别为 unsafe（误报）
            fp = len(df[(df["decision"] == "unsafe") & (df["label"] == 1)])
            # True Positive: 不应该发生（label 都是 1）
            tp = 0
            # False Negative: 不应该发生（label 都是 1）
            fn = 0
            
            correct = tn
            expected_behavior = "放行(safe)"
        
        # 计算指标（排除 error 样本）
        valid_total = total - error_count
        accuracy = (correct / valid_total * 100) if valid_total > 0 else 0.0
        
        # Precision, Recall, F1（针对 unsafe 类）
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {
            "category": category,
            "type": data_type,
            "status": "success",
            "total": total,
            "safe": safe_count,
            "unsafe": unsafe_count,
            "error": error_count,
            "correct": correct,
            "valid_total": valid_total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "expected_behavior": expected_behavior,
        }
        
    except Exception as e:
        print(f"❌ 读取 {category} {data_type} 结果时出错: {str(e)}")
        return {
            "category": category,
            "type": data_type,
            "status": "error",
            "error_message": str(e),
            "total": 0,
            "safe": 0,
            "unsafe": 0,
            "error": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


def print_category_summary(stats: Dict[str, Any]):
    """打印单个类别的统计信息"""
    category = stats["category"]
    data_type = stats["type"]
    status = stats["status"]
    
    if status == "not_found":
        print(f"  ⚠️  数据未找到")
        return
    elif status == "error":
        print(f"  ❌ 错误: {stats.get('error_message', 'Unknown error')}")
        return
    
    total = stats["total"]
    safe = stats["safe"]
    unsafe = stats["unsafe"]
    error = stats["error"]
    correct = stats["correct"]
    valid_total = stats["valid_total"]
    accuracy = stats["accuracy"]
    expected = stats["expected_behavior"]
    
    print(f"  📊 总计: {total} | 正确: {correct}/{valid_total} ({accuracy:.2f}%)")
    print(f"     Safe: {safe} | Unsafe: {unsafe} | Error: {error}")
    print(f"     期望行为: {expected}")
    
    if data_type == "harmful":
        print(f"     Precision: {stats['precision']:.2f}% | Recall: {stats['recall']:.2f}% | F1: {stats['f1']:.2f}%")


def generate_summary_table(all_stats: List[Dict[str, Any]]) -> pd.DataFrame:
    """生成汇总表格"""
    summary_data = []
    
    for stats in all_stats:
        if stats["status"] == "success":
            summary_data.append({
                "类别": stats["category"],
                "数据类型": stats["type"],
                "总数": stats["total"],
                "正确数": stats["correct"],
                "有效总数": stats["valid_total"],
                "准确率(%)": f"{stats['accuracy']:.2f}",
                "Safe": stats["safe"],
                "Unsafe": stats["unsafe"],
                "Error": stats["error"],
                "Precision(%)": f"{stats['precision']:.2f}" if stats["type"] == "harmful" else "N/A",
                "Recall(%)": f"{stats['recall']:.2f}" if stats["type"] == "harmful" else "N/A",
                "F1(%)": f"{stats['f1']:.2f}" if stats["type"] == "harmful" else "N/A",
            })
    
    return pd.DataFrame(summary_data)


def main():
    print("\n" + "=" * 80)
    print("AgentHarm 分类数据集结果汇总")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型配置: {MODEL_CONFIG}")
    print("=" * 80 + "\n")
    
    all_stats = []
    
    for category in CATEGORIES:
        print(f"\n📁 类别: {category}")
        print("-" * 80)
        
        # Harmful 数据统计
        print(f"  🔴 Harmful 数据:")
        harmful_stats = summarize_category_results(category, "harmful")
        print_category_summary(harmful_stats)
        all_stats.append(harmful_stats)
        
        print()
        
        # Benign 数据统计
        print(f"  🟢 Benign 数据:")
        benign_stats = summarize_category_results(category, "benign")
        print_category_summary(benign_stats)
        all_stats.append(benign_stats)
    
    print("\n" + "=" * 80)
    print("整体统计")
    print("=" * 80)
    
    # 计算整体统计
    harmful_stats_list = [s for s in all_stats if s["type"] == "harmful" and s["status"] == "success"]
    benign_stats_list = [s for s in all_stats if s["type"] == "benign" and s["status"] == "success"]
    
    if harmful_stats_list:
        total_harmful = sum(s["total"] for s in harmful_stats_list)
        correct_harmful = sum(s["correct"] for s in harmful_stats_list)
        valid_harmful = sum(s["valid_total"] for s in harmful_stats_list)
        accuracy_harmful = (correct_harmful / valid_harmful * 100) if valid_harmful > 0 else 0.0
        
        print(f"\n🔴 Harmful 数据总计:")
        print(f"   总数: {total_harmful} | 正确: {correct_harmful}/{valid_harmful} ({accuracy_harmful:.2f}%)")
    
    if benign_stats_list:
        total_benign = sum(s["total"] for s in benign_stats_list)
        correct_benign = sum(s["correct"] for s in benign_stats_list)
        valid_benign = sum(s["valid_total"] for s in benign_stats_list)
        accuracy_benign = (correct_benign / valid_benign * 100) if valid_benign > 0 else 0.0
        
        print(f"\n🟢 Benign 数据总计:")
        print(f"   总数: {total_benign} | 正确: {correct_benign}/{valid_benign} ({accuracy_benign:.2f}%)")
    
    if harmful_stats_list and benign_stats_list:
        total_all = total_harmful + total_benign
        correct_all = correct_harmful + correct_benign
        valid_all = valid_harmful + valid_benign
        accuracy_all = (correct_all / valid_all * 100) if valid_all > 0 else 0.0
        
        print(f"\n📊 总体:")
        print(f"   总数: {total_all} | 正确: {correct_all}/{valid_all} ({accuracy_all:.2f}%)")
    
    # 生成详细表格
    print("\n" + "=" * 80)
    print("详细统计表格")
    print("=" * 80 + "\n")
    
    summary_df = generate_summary_table(all_stats)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        # 保存到 CSV
        output_path = "results/agentharm_categories_summary.csv"
        summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 详细统计已保存到: {output_path}")
    
    # 保存 JSON 格式
    json_output_path = "results/agentharm_categories_summary.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 格式已保存到: {json_output_path}")
    
    print("\n" + "=" * 80)
    print("汇总完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
