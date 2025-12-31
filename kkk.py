"""
从已有的CSV结果中生成误判案例详细记录
用于补充生成之前实验中未保存的误判案例
"""

import os
import json
import pandas as pd
import argparse


def save_misjudged_cases(df_misjudged: pd.DataFrame, dataset: str, misjudge_type: str):
    """保存误判案例到JSONL文件"""
    detail_dir = os.path.join("detail_results", dataset, misjudge_type)
    os.makedirs(detail_dir, exist_ok=True)

    file_path = os.path.join(detail_dir, "misjudged_cases.jsonl")

    # 如果文件已存在，先删除
    if os.path.exists(file_path):
        os.remove(file_path)

    saved_count = 0
    for idx, row in df_misjudged.iterrows():
        misjudge_record = {
            "index": int(idx),
            "misjudge_type": misjudge_type,
            "label": {
                "value": int(row["label"]) if pd.notna(row["label"]) else None,
            },
            "decision": {
                "value": row.get("decision", ""),
            },
            "input": row.get("input", ""),
            "agent_actions": row.get("agent_actions", ""),
            "generated_tools": row.get("generated_tools", ""),
        }

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(misjudge_record, ensure_ascii=False) + "\n")
        saved_count += 1

    print(f"  ✓ Saved {saved_count} cases to: {file_path}")


def process_csv(csv_path: str, dataset: str):
    """处理单个CSV文件"""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 先将字符串label转换为数字：unsafe->0, safe->1
    # 注意：有些数据集的label已经是数字，有些是字符串
    def convert_label(label):
        if pd.isna(label):
            return None
        if isinstance(label, str):
            # 字符串label：unsafe->0, safe->1
            return 0 if label.lower() == "unsafe" else 1
        else:
            # 已经是数字，直接返回
            return int(label)

    df["label"] = df["label"].apply(convert_label)

    print(f"\nProcessing: {dataset}")
    print(f"Total: {len(df)}")

    # 找出误判案例
    # label=0是harmful(应判unsafe), label=1是benign(应判safe)
    benign_judged_harmful = df[(df["label"] == 1) & (df["decision"] == "unsafe")]
    harmful_judged_benign = df[(df["label"] == 0) & (df["decision"] == "safe")]

    print(f"  Benign->Harmful: {len(benign_judged_harmful)}")
    print(f"  Harmful->Benign: {len(harmful_judged_benign)}")

    # 保存误判案例
    if len(benign_judged_harmful) > 0:
        save_misjudged_cases(benign_judged_harmful, dataset, "benign_judged_harmful")
    if len(harmful_judged_benign) > 0:
        save_misjudged_cases(harmful_judged_benign, dataset, "harmful_judged_benign")

    # 计算准确率
    correct = len(df[(df["label"] == 0) & (df["decision"] == "unsafe")]) + len(
        df[(df["label"] == 1) & (df["decision"] == "safe")]
    )
    error = len(df[df["decision"] == "error"])
    valid = len(df) - error
    acc = (correct / valid * 100) if valid > 0 else 0
    print(f"  ACC: {correct}/{valid} ({acc:.2f}%)\n")


def main():
    parser = argparse.ArgumentParser(description="从CSV结果中生成误判案例")
    parser.add_argument("--csv", type=str, help="CSV结果文件路径")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument(
        "--batch", action="store_true", help="批量处理所有ASSEBench结果"
    )

    args = parser.parse_args()

    if args.batch:
        # 批量处理
        results_dir = "results"
        if not os.path.exists(results_dir):
            print(f"Error: {results_dir} not found")
            return

        print(f"\n{'='*60}")
        print("Batch Processing ASSEBench Results")
        print(f"{'='*60}")

        for dirname in os.listdir(results_dir):
            if not dirname.startswith("assebench_"):
                continue

            result_dir = os.path.join(results_dir, dirname)
            if not os.path.isdir(result_dir):
                continue

            # 查找CSV文件
            csv_files = [f for f in os.listdir(result_dir) if f.endswith(".csv")]
            if csv_files:
                csv_path = os.path.join(result_dir, csv_files[0])
                process_csv(csv_path, dirname)

        print(f"{'='*60}")
        print("✓ Batch processing completed!")
        print(f"{'='*60}\n")

    elif args.csv and args.dataset:
        # 处理单个文件
        process_csv(args.csv, args.dataset)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # 处理单个CSV")
        print(
            "  python generate_misjudged_from_csv.py --csv results/assebench_Security/file.csv --dataset assebench_Security"
        )
        print("\n  # 批量处理所有ASSEBench")
        print("  python generate_misjudged_from_csv.py --batch")


if __name__ == "__main__":
    main()
