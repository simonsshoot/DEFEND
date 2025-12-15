
import json
import os
import sys
import logging
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    config: Dict,
    max_workers: int = 3,
    save_interval: int = 5
):
    """
    处理单个数据集文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        config: 配置字典
        max_workers: 最大并行工作进程数
        save_interval: 保存间隔（处理多少条后保存一次）
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
            if item.get("is_judge"):
                data[i] = item
    
    # 使用进程池并行处理
    futures_map = {}
    processed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        for idx, item in enumerate(data):
            # 跳过已评估的项
            if item.get("is_judge"):
                processed_count += 1
                continue
            
            future = executor.submit(process_item, item, config)
            futures_map[future] = idx
        
        # 处理完成的任务
        for future in as_completed(futures_map.keys()):
            idx = futures_map[future]
            try:
                result = future.result()
                data[idx] = result
                processed_count += 1
                
                logger.info(f"Processed {processed_count}/{len(data)} items")
                
                # 定期保存进度
                if processed_count % save_interval == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Progress saved at {processed_count} items")
                    
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Final results saved to {output_path}")


def main():
    print("=" * 80)
    print("R-Judge数据集评估")
    print("=" * 80)
    
    # 配置参数
    config = {
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "model": MODEL_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
        "bm25_storage_path": BM25_STORAGE_PATH,
        "threshold_high": THRESOLD_HIGH,
        "threshold_low": THRESOLD_LOW
    }
    
    # R-Judge数据集目录
    data_dir = "/home/yx/yx_search/aisafety/DEFEND/data/R-Judge"
    
    # 输出目录
    output_dir = "./results/rjudge_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # R-Judge的5个类别
    categories = ["Application", "Finance", "IoT", "Program", "Web"]
    
    # 数据类型
    data_types = ["benign", "harmful"]
    
    # 并行工作进程数
    max_workers = 3
    
    # 处理每个类别的每种数据类型
    for category in categories:
        print(f"\n{'=' * 80}")
        print(f"处理类别: {category}")
        print(f"{'=' * 80}\n")
        
        for data_type in data_types:
            input_path = os.path.join(data_dir, category, f"{data_type}.json")
            
            # 检查文件是否存在
            if not os.path.exists(input_path):
                logger.warning(f"文件不存在，跳过: {input_path}")
                continue
            
            # 设置输出路径
            output_path = os.path.join(output_dir, f"{category}_{data_type}_results.json")
            
            print(f"\n处理文件: {input_path}")
            print(f"输出路径: {output_path}\n")
            
            # 处理数据集
            try:
                process_dataset(
                    input_path=input_path,
                    output_path=output_path,
                    config=config,
                    max_workers=max_workers,
                    save_interval=5
                )
                print(f"✓ 完成: {category} - {data_type}")
            except Exception as e:
                logger.error(f"✗ 处理失败: {category} - {data_type}")
                logger.error(f"错误信息: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ R-Judge数据集评估完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
