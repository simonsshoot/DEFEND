
import json
import os
import sys
import logging
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

sys.path.append('./environments')
from defence_online import SecurityEvaluator, parse_envs
from embedding import ChromaDBManager
from Hybrid_retrieval import HybridDBManager
from utils import LLMClient
from config import API_KEY, BASE_URL, BM25_STORAGE_PATH, CHROMA_PERSIST_DIR, THRESOLD_LOW, THRESOLD_HIGH, MODEL_NAME

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def convert_rjudge_to_standard_format(item: Dict) -> Dict:
    """
    将R-Judge数据格式转换为评估器期望的格式
    
    Args:
        item: R-Judge格式的数据项
        
    Returns:
        标准格式的数据项
    """
    # 提取用户初始指令
    instruction = ""
    if item.get("contents") and len(item["contents"]) > 0:
        first_turn = item["contents"][0]
        if first_turn and len(first_turn) > 0:
            user_msg = first_turn[0]
            if user_msg.get("role") == "user":
                instruction = user_msg.get("content", "")
    
    # 添加profile作为系统背景
    profile = item.get("profile", "")
    if profile and instruction:
        instruction = f"{profile}\n\n{instruction}"
    
    # 构建agent_trajectory（智能体轨迹）
    agent_trajectory = []
    for turn_idx, turn in enumerate(item.get("contents", [])):
        for msg in turn:
            role = msg.get("role")
            if role == "user":
                agent_trajectory.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })
            elif role == "agent":
                # R-Judge中的agent消息
                thought = msg.get("thought")
                action = msg.get("action")
                
                # 构建assistant消息内容
                if thought and action:
                    content = f"THOUGHT: {thought}\nACTION: {action}"
                elif action:
                    content = action
                elif thought:
                    content = thought
                else:
                    content = ""
                
                # 如果有实际内容，添加到轨迹中
                if content:
                    agent_trajectory.append({
                        "role": "assistant",
                        "content": content
                    })
                    
            elif role == "environment":
                # 环境响应作为tool返回
                env_content = msg.get("content")
                if env_content:
                    agent_trajectory.append({
                        "role": "tool",
                        "content": env_content,
                        "tool_call_id": f"call_{turn_idx}"
                    })
    
    # 构建标准格式
    standard_item = {
        "instruction": instruction,
        "output": agent_trajectory,
        "environments": [],  # R-Judge数据中没有明确的环境定义
        # 保留原始R-Judge字段用于追溯
        "rjudge_id": item.get("id"),
        "rjudge_scenario": item.get("scenario"),
        "rjudge_profile": item.get("profile"),
        "rjudge_goal": item.get("goal"),
        "rjudge_label": item.get("label"),  # 0=safe, 1=unsafe
        "rjudge_risk_description": item.get("risk_description", ""),
        "rjudge_attack_type": item.get("attack_type", "")
    }
    
    return standard_item


def process_item(item: Dict, config: Dict) -> Dict:
    """处理单个数据项"""

    if "is_judge" in item and item["is_judge"]:
        return item
    
    try:
        llm_client = LLMClient(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"]
        )
        
        hybrid = HybridDBManager(
            collection_name="attack_essence_agentharm",
            persist_dir=config["persist_dir"],
            alpha=0.3,
            bm25_storage_path=os.path.join(config["bm25_storage_path"], "attack_essence")
        )
        
        evaluator = SecurityEvaluator(
            llm_client=llm_client,
            db_manager=hybrid,
            threshold_high=config["threshold_high"],
            threshold_low=config["threshold_low"],
            enable_guardrail=True,
        )
        
        standard_item = convert_rjudge_to_standard_format(item)
        
        result = evaluator.evaluate_query(standard_item)
        
        # 更新结果
        item.update(result)
        item["is_judge"] = True
        item["final_is_harmful"] = item.get("is_harmful", False)
        
        logger.info(f"Item {item.get('id')} evaluated: {item['final_is_harmful']}")
        
    except Exception as e:
        logger.error(f"Error processing item {item.get('id')}: {e}")
        import traceback
        traceback.print_exc()
        item["error"] = str(e)
        item["is_judge"] = False
    
    return item


def process_dataset(
    input_path: str,
    output_path: str,
    data_type: str,
    config: Dict,
    max_workers: int = 1,  # 改为串行处理，避免多进程数据库冲突
    save_interval: int = 5
) -> Dict[str, Any]:
    """
    处理单个数据集文件并返回评估指标
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        data_type: 数据类型（"benign" 或 "harmful"）
        config: 配置字典
        max_workers: 最大并行工作进程数
        save_interval: 保存间隔（处理多少条后保存一次）
        
    Returns:
        评估指标字典
    """
    # 读取数据
    logger.info(f"Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Total items to process: {len(data)}")
    
    # 如果输出文件已存在，加载已有进度
    if os.path.exists(output_path):
        logger.info(f"Loading existing progress from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        # 合并已评估的结果
        for i, item in enumerate(existing_data):
            if i < len(data) and item.get("is_judge"):
                data[i] = item
    
    # 串行处理以避免数据库冲突
    processed_count = sum(1 for item in data if item.get("is_judge"))
    total_count = len(data)
    
    logger.info(f"Already processed: {processed_count}/{total_count}")
    
    for idx, item in enumerate(data):
        # 跳过已评估的项
        if item.get("is_judge"):
            continue
        
        try:
            result = process_item(item, config)
            data[idx] = result
            processed_count += 1
            
            if processed_count % 5 == 0:
                logger.info(f"Processed {processed_count}/{total_count} items")
            
            # 定期保存进度
            if processed_count % save_interval == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Progress saved at {processed_count} items")
                
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            import traceback
            traceback.print_exc()
            # 标记错误但继续处理
            data[idx]["error"] = str(e)
            data[idx]["is_judge"] = False
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Final results saved to {output_path}")
    
    # 计算指标
    metrics = calculate_metrics(data, data_type)
    
    return metrics


def calculate_metrics(results: List[Dict], data_type: str) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # 统计各种情况
    true_positive = 0   # 正确识别为harmful
    true_negative = 0   # 正确识别为benign
    false_positive = 0  # 误报（benign被识别为harmful）
    false_negative = 0  # 漏报（harmful被识别为benign）
    
    for item in results:
        predicted_harmful = item.get("final_is_harmful", False)
        
        # R-Judge的label: 0=safe, 1=unsafe
        # 或者根据data_type判断真实标签
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
    
    # 计算指标
    accuracy = (true_positive + true_negative) / total if total > 0 else 0.0
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 特异性（Specificity）：正确识别benign的能力
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


def print_metrics(category: str, data_type: str, metrics: Dict[str, Any]):
    """打印格式化的指标"""
    print(f"\n{'='*60}")
    print(f"类别: {category} | 数据类型: {data_type}")
    print(f"{'='*60}")
    print(f"总数据量: {metrics['total']}")
    print(f"真阳性 (TP): {metrics['true_positive']}")
    print(f"真阴性 (TN): {metrics['true_negative']}")
    print(f"假阳性 (FP): {metrics['false_positive']}")
    print(f"假阴性 (FN): {metrics['false_negative']}")
    print(f"{'-'*60}")
    print(f"准确率 (Accuracy):    {metrics['accuracy']:.2%}")
    print(f"精确率 (Precision):   {metrics['precision']:.2%}")
    print(f"召回率 (Recall):      {metrics['recall']:.2%}")
    print(f"F1 分数:              {metrics['f1_score']:.2%}")
    print(f"特异性 (Specificity): {metrics['specificity']:.2%}")
    print(f"误报率 (FPR):         {metrics['false_positive_rate']:.2%}")
    print(f"漏报率 (FNR):         {metrics['false_negative_rate']:.2%}")
    print(f"{'='*60}\n")


def save_summary_report(all_metrics: Dict, output_dir: str):

    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summary report saved to {summary_file}")
    
    # 生成Markdown报告
    md_file = os.path.join(output_dir, "evaluation_summary.md")
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# R-Judge数据集评估报告\n\n")
        f.write("## 使用的模型和配置\n\n")
        f.write(f"- 风险模式库: attack_essence_agentharm（经过offline_train训练）\n")
        f.write(f"- 高阈值: {THRESOLD_HIGH}\n")
        f.write(f"- 低阈值: {THRESOLD_LOW}\n\n")
        
        f.write("## 各类别评估结果\n\n")
        
        # 按类别组织
        categories = ["Application", "Finance", "IoT", "Program", "Web"]
        
        for category in categories:
            if category not in all_metrics:
                continue
            
            f.write(f"### {category}\n\n")
            
            # Benign评估
            if "benign" in all_metrics[category]:
                metrics = all_metrics[category]["benign"]
                f.write("#### Benign数据评估\n\n")
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                f.write(f"| 总数 | {metrics['total']} |\n")
                f.write(f"| 准确率 | {metrics['accuracy']:.2%} |\n")
                f.write(f"| 精确率 | {metrics['precision']:.2%} |\n")
                f.write(f"| 召回率 | {metrics['recall']:.2%} |\n")
                f.write(f"| F1分数 | {metrics['f1_score']:.2%} |\n")
                f.write(f"| 误报率 | {metrics['false_positive_rate']:.2%} |\n\n")
            
            # Harmful评估
            if "harmful" in all_metrics[category]:
                metrics = all_metrics[category]["harmful"]
                f.write("#### Harmful数据评估\n\n")
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                f.write(f"| 总数 | {metrics['total']} |\n")
                f.write(f"| 准确率 | {metrics['accuracy']:.2%} |\n")
                f.write(f"| 精确率 | {metrics['precision']:.2%} |\n")
                f.write(f"| 召回率 | {metrics['recall']:.2%} |\n")
                f.write(f"| F1分数 | {metrics['f1_score']:.2%} |\n")
                f.write(f"| 漏报率 | {metrics['false_negative_rate']:.2%} |\n\n")
        
        # 总体统计
        f.write("## 总体统计\n\n")
        
        total_benign = sum(all_metrics[cat].get("benign", {}).get("total", 0) for cat in categories if cat in all_metrics)
        total_harmful = sum(all_metrics[cat].get("harmful", {}).get("total", 0) for cat in categories if cat in all_metrics)
        
        avg_benign_accuracy = sum(all_metrics[cat].get("benign", {}).get("accuracy", 0) for cat in categories if cat in all_metrics) / len([c for c in categories if c in all_metrics and "benign" in all_metrics[c]]) if any("benign" in all_metrics.get(c, {}) for c in categories if c in all_metrics) else 0
        avg_harmful_accuracy = sum(all_metrics[cat].get("harmful", {}).get("accuracy", 0) for cat in categories if cat in all_metrics) / len([c for c in categories if c in all_metrics and "harmful" in all_metrics[c]]) if any("harmful" in all_metrics.get(c, {}) for c in categories if c in all_metrics) else 0
        
        f.write(f"- 总Benign数据量: {total_benign}\n")
        f.write(f"- 总Harmful数据量: {total_harmful}\n")
        f.write(f"- Benign平均准确率: {avg_benign_accuracy:.2%}\n")
        f.write(f"- Harmful平均准确率: {avg_harmful_accuracy:.2%}\n")
    
    logger.info(f"Markdown report saved to {md_file}")


def main():
    print("=" * 80)
    print("R-Judge数据集评估")
    print("使用训练后的AgentHarm风险模式库")
    print("=" * 80)

    config = {
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "model": MODEL_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
        "bm25_storage_path": BM25_STORAGE_PATH,
        "threshold_high": THRESOLD_HIGH,
        "threshold_low": THRESOLD_LOW
    }

    # 初始化数据库管理器
    print("\n初始化数据库...")
    db_manager1 = HybridDBManager(
        collection_name="attack_essence_agentharm",
        persist_dir=CHROMA_PERSIST_DIR,
        alpha=0.3,
        bm25_storage_path=os.path.join(BM25_STORAGE_PATH, "attack_essence")
    )
    db_manager2 = ChromaDBManager(
        collection_name="harmful_result_agentharm",
        persist_dir=CHROMA_PERSIST_DIR
    )
    
    db_count = db_manager1.collection.count()
    print(f"数据库中的风险模式数量: {db_count}")
    
    # 如果数据库为空，尝试从去重数据文件初始化
    if db_count == 0:
        print("\n数据库为空，正在尝试导入去重后的AgentHarm数据...")
        
        # 尝试多个可能的数据文件路径
        possible_data_files = [
            "./deduplicated_agentharm_patterns.json",
            "../deduplicated_agentharm_patterns.json",
            "./results/deduplicated_agentharm_patterns.json"
        ]
        
        harmful_data = []
        data_file_found = None
        
        for data_file in possible_data_files:
            if os.path.exists(data_file):
                print(f"找到数据文件: {data_file}")
                with open(data_file, 'r', encoding='utf-8') as f:
                    harmful_data = json.load(f)
                data_file_found = data_file
                break
        
        if not harmful_data:
            print("\n" + "!" * 80)
            print("错误：风险模式数据库为空且未找到去重数据文件！")
            return
        
        print(f"加载了 {len(harmful_data)} 条去重数据")
        
        # 导入数据到数据库
        imported_count = 0
        for item in harmful_data:
            if "attack_essence" not in item:
                print(f"警告：数据项缺少 attack_essence 字段，跳过")
                continue
            
            db_manager1.add_items("attack_essence", [item])
            db_manager2.add_items("harmful_result", [item])
            imported_count += 1
        
        print(f"✓ 已导入 {imported_count} 条数据到数据库")
        print(f"  - attack_essence_agentharm: {db_manager1.collection.count()} 条")
        print(f"  - harmful_result_agentharm: {db_manager2.collection.count()} 条")
    else:
        print(f"✓ 数据库已就绪，包含 {db_count} 个风险模式\n")

    data_dir = "/home/yx/yx_search/aisafety/DEFEND/data/R-Judge"

    output_dir = "./results/rjudge_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    categories = ["Application", "Finance", "IoT", "Program", "Web"]

    data_types = ["benign", "harmful"]
    
    max_workers = 1  # 串行处理避免数据库冲突
    all_metrics = {}
    
    # 统计要处理的文件
    files_to_process = []
    for category in categories:
        for data_type in data_types:
            input_path = os.path.join(data_dir, category, f"{data_type}.json")
            if os.path.exists(input_path):
                files_to_process.append((category, data_type, input_path))
            else:
                logger.warning(f"文件不存在，跳过: {input_path}")
    
    print(f"\n找到 {len(files_to_process)} 个文件需要处理\n")
    
    # 处理每个文件
    for idx, (category, data_type, input_path) in enumerate(files_to_process, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(files_to_process)}] 处理: {category} - {data_type}")
        print(f"{'=' * 80}\n")
        
        if category not in all_metrics:
            all_metrics[category] = {}
        
        output_path = os.path.join(output_dir, f"{category}_{data_type}_results.json")
        
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}\n")

        try:
            metrics = process_dataset(
                input_path=input_path,
                output_path=output_path,
                data_type=data_type,
                config=config,
                max_workers=max_workers,
                save_interval=5
            )
            
            # 存储指标
            all_metrics[category][data_type] = metrics
            
            # 打印指标
            print_metrics(category, data_type, metrics)
            
            print(f"✓ 完成: {category} - {data_type} [{idx}/{len(files_to_process)}]")
            
        except Exception as e:
            logger.error(f"✗ 处理失败: {category} - {data_type}")
            logger.error(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n继续处理下一个文件...\n")
    
    # 保存汇总报告
    print("\n" + "=" * 80)
    print("生成汇总报告...")
    print("=" * 80 + "\n")
    
    save_summary_report(all_metrics, output_dir)
    
    # 打印总体统计
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80 + "\n")
    
    total_benign = 0
    total_harmful = 0
    total_benign_correct = 0
    total_harmful_correct = 0
    
    for category in categories:
        if category not in all_metrics:
            continue
        
        if "benign" in all_metrics[category]:
            metrics = all_metrics[category]["benign"]
            total_benign += metrics["total"]
            total_benign_correct += metrics["true_negative"]
        
        if "harmful" in all_metrics[category]:
            metrics = all_metrics[category]["harmful"]
            total_harmful += metrics["total"]
            total_harmful_correct += metrics["true_positive"]
    
    overall_accuracy = (total_benign_correct + total_harmful_correct) / (total_benign + total_harmful) if (total_benign + total_harmful) > 0 else 0
    benign_accuracy = total_benign_correct / total_benign if total_benign > 0 else 0
    harmful_accuracy = total_harmful_correct / total_harmful if total_harmful > 0 else 0
    
    print(f"总Benign数据量: {total_benign}")
    print(f"总Harmful数据量: {total_harmful}")
    print(f"Benign正确识别: {total_benign_correct} ({benign_accuracy:.2%})")
    print(f"Harmful正确识别: {total_harmful_correct} ({harmful_accuracy:.2%})")
    print(f"总体准确率: {overall_accuracy:.2%}")
    
    print("\n" + "=" * 80)
    print("✓ R-Judge数据集评估完成！")
    print(f"结果保存在: {output_dir}")
    print(f"  - 详细结果: *_results.json")
    print(f"  - 汇总报告: evaluation_summary.json")
    print(f"  - Markdown报告: evaluation_summary.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
