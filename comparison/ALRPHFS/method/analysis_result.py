"""
分析R-Judge评估结果
从已保存的结果文件中重新计算指标和生成报告
"""
import json
import os
from typing import Dict, Any, List
from collections import defaultdict


def calculate_metrics(results: List[Dict], data_type: str) -> Dict[str, Any]:
    """计算评估指标"""
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for item in results:
        predicted_harmful = item.get("final_is_harmful", False)
        
        if data_type == "harmful":
            actual_harmful = True
        else:
            actual_harmful = False
        
        if actual_harmful and predicted_harmful:
            true_positive += 1
        elif not actual_harmful and not predicted_harmful:
            true_negative += 1
        elif not actual_harmful and predicted_harmful:
            false_positive += 1
        elif actual_harmful and not predicted_harmful:
            false_negative += 1
    
    accuracy = (true_positive + true_negative) / total if total > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0.0
    
    metrics = {
        "total": total,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(false_positive / total, 4) if total > 0 else 0.0,
        "false_negative_rate": round(false_negative / total, 4) if total > 0 else 0.0
    }
    
    return metrics


def analyze_results(results_dir: str):
    """分析结果目录中的所有结果文件"""
    categories = ["Application", "Finance", "IoT", "Program", "Web"]
    data_types = ["benign", "harmful"]
    
    all_metrics = {}
    
    for category in categories:
        all_metrics[category] = {}
        
        for data_type in data_types:
            result_file = os.path.join(results_dir, f"{category}_{data_type}_results.json")
            
            if not os.path.exists(result_file):
                print(f"警告：文件不存在 {result_file}")
                continue
            
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            metrics = calculate_metrics(results, data_type)
            all_metrics[category][data_type] = metrics
            
            print(f"\n{'='*60}")
            print(f"类别: {category} | 数据类型: {data_type}")
            print(f"{'='*60}")
            print(f"总数据量: {metrics['total']}")
            print(f"真阳性 (TP): {metrics['true_positive']}")
            print(f"真阴性 (TN): {metrics['true_negative']}")
            print(f"假阳性 (FP): {metrics['false_positive']} (误报)")
            print(f"假阴性 (FN): {metrics['false_negative']} (漏报)")
            print(f"{'-'*60}")
            print(f"准确率 (Accuracy):    {metrics['accuracy']:.2%}")
            print(f"精确率 (Precision):   {metrics['precision']:.2%}")
            print(f"召回率 (Recall):      {metrics['recall']:.2%}")
            print(f"F1 分数:              {metrics['f1_score']:.2%}")
            print(f"特异性 (Specificity): {metrics['specificity']:.2%}")
            print(f"{'='*60}\n")
    
    # 保存汇总
    summary_file = os.path.join(results_dir, "metrics_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n指标汇总已保存到: {summary_file}")
    
    # 生成详细报告
    generate_detailed_report(all_metrics, results_dir)
    
    return all_metrics


def generate_detailed_report(all_metrics: Dict, output_dir: str):
    """生成详细的Markdown报告"""
    report_file = os.path.join(output_dir, "detailed_report.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# R-Judge评估详细报告\n\n")
        
        # 总体统计
        f.write("## 总体统计\n\n")
        
        total_benign = 0
        total_harmful = 0
        correct_benign = 0
        correct_harmful = 0
        total_fp = 0
        total_fn = 0
        
        for category, data in all_metrics.items():
            if "benign" in data:
                total_benign += data["benign"]["total"]
                correct_benign += data["benign"]["true_negative"]
                total_fp += data["benign"]["false_positive"]
            if "harmful" in data:
                total_harmful += data["harmful"]["total"]
                correct_harmful += data["harmful"]["true_positive"]
                total_fn += data["harmful"]["false_negative"]
        
        total_samples = total_benign + total_harmful
        total_correct = correct_benign + correct_harmful
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        f.write(f"- **总样本数**: {total_samples}\n")
        f.write(f"  - Benign: {total_benign}\n")
        f.write(f"  - Harmful: {total_harmful}\n")
        f.write(f"- **总体准确率**: {overall_accuracy:.2%}\n")
        f.write(f"- **误报总数**: {total_fp} ({total_fp/total_benign:.2%} of benign)\n")
        f.write(f"- **漏报总数**: {total_fn} ({total_fn/total_harmful:.2%} of harmful)\n\n")
        
        # 各类别详细数据
        f.write("## 各类别详细指标\n\n")
        
        for category in ["Application", "Finance", "IoT", "Program", "Web"]:
            if category not in all_metrics:
                continue
            
            f.write(f"### {category}\n\n")
            
            # Benign
            if "benign" in all_metrics[category]:
                m = all_metrics[category]["benign"]
                f.write("#### Benign数据\n\n")
                f.write("| 指标 | 值 | 说明 |\n")
                f.write("|------|----|----- |\n")
                f.write(f"| 总数 | {m['total']} | - |\n")
                f.write(f"| 准确率 | {m['accuracy']:.2%} | 正确识别为benign的比例 |\n")
                f.write(f"| 误报数 | {m['false_positive']} | benign被错误识别为harmful |\n")
                f.write(f"| 误报率 | {m['false_positive_rate']:.2%} | FP/Total |\n")
                f.write(f"| 特异性 | {m['specificity']:.2%} | 1 - 误报率 |\n\n")
            
            # Harmful
            if "harmful" in all_metrics[category]:
                m = all_metrics[category]["harmful"]
                f.write("#### Harmful数据\n\n")
                f.write("| 指标 | 值 | 说明 |\n")
                f.write("|------|----|----- |\n")
                f.write(f"| 总数 | {m['total']} | - |\n")
                f.write(f"| 准确率 | {m['accuracy']:.2%} | 正确识别为harmful的比例 |\n")
                f.write(f"| 召回率 | {m['recall']:.2%} | TP/(TP+FN) |\n")
                f.write(f"| 精确率 | {m['precision']:.2%} | TP/(TP+FP) |\n")
                f.write(f"| F1分数 | {m['f1_score']:.2%} | 精确率和召回率的调和平均 |\n")
                f.write(f"| 漏报数 | {m['false_negative']} | harmful被错误识别为benign |\n")
                f.write(f"| 漏报率 | {m['false_negative_rate']:.2%} | FN/Total |\n\n")
        
        # 跨类别比较
        f.write("## 跨类别比较\n\n")
        
        f.write("### Benign识别准确率\n\n")
        f.write("| 类别 | 准确率 | 误报率 |\n")
        f.write("|------|--------|--------|\n")
        for category in ["Application", "Finance", "IoT", "Program", "Web"]:
            if category in all_metrics and "benign" in all_metrics[category]:
                m = all_metrics[category]["benign"]
                f.write(f"| {category} | {m['accuracy']:.2%} | {m['false_positive_rate']:.2%} |\n")
        f.write("\n")
        
        f.write("### Harmful检测效果\n\n")
        f.write("| 类别 | 召回率 | 精确率 | F1分数 |\n")
        f.write("|------|--------|--------|--------|\n")
        for category in ["Application", "Finance", "IoT", "Program", "Web"]:
            if category in all_metrics and "harmful" in all_metrics[category]:
                m = all_metrics[category]["harmful"]
                f.write(f"| {category} | {m['recall']:.2%} | {m['precision']:.2%} | {m['f1_score']:.2%} |\n")
        f.write("\n")
    
    print(f"详细报告已保存到: {report_file}")


def main():
    results_dir = "./results/rjudge_evaluation"
    
    if not os.path.exists(results_dir):
        print(f"错误：结果目录不存在 {results_dir}")
        print("请先运行 eval_rjudge.py 进行评估")
        return
    
    print("=" * 80)
    print("分析R-Judge评估结果")
    print("=" * 80)
    
    all_metrics = analyze_results(results_dir)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
