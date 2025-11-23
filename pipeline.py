# 这里执行完整的pipeline逻辑
"""
暂时的pipeline：
是自进化出安全工具，这些工具能作为安全防护补丁，保证agent的安全性：
输入一些请求，不管是不是恶意的，比如操作请求，像safeOS那样，然后定向进化：请求先丢给大模型，询问可能暗含哪些风险，拿到风险后，去找现有的安全工具看有没有（可重用优先）；如果没有，让大模型生成安全工具。同时，为了防止生成的安全工具本身也可能存在风险，会有一个末端的怀疑模型，专门进行质疑，当安全工具通过质疑后，才会被加入工具库。
对于已有安全工具本身，会有一个优化器LLM，给定上下文和安全工具，构想是否可以优化，优化后也需要过怀疑模型，不然回退
执行应该放在哪里？应该放在安全工具通过怀疑模型验证之后、真实执行请求之前

一个新的问题是，当没有通过安全工具验证时，拒绝执行，但是，可能用户本身是无意识的，是一刀切的拒绝好，还是将用户请求的风险去除后再执行好？

还有一个问题需要考虑：过度防御的问题  怀疑模型加一个这个功能？
怀疑模型应该是最终汇总，安全工具，用户请求，执行结果，然后进行质疑

还有就是有些工具需要外部依赖的，比如说import re，这个如何解决

加载lifelong_memory可能会涉及到记忆爆炸的问题。此外，当运行较多时，可能输入给模型的token会过长
"""
from configs import pipeline_config
import argparse
import random
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Tuple
from utils import read_data, data_wrapper
from container import Container
from agents import Agent, TarevoAgent, OptimAgent, DoubtAgent
from configs import setup_logger

logger = setup_logger("pipeline")


def pipeline(
    args: argparse.Namespace, data: Dict[str, Any], container: Container
) -> Tuple[List[Dict], str]:
    # Step 1: TarevoAgent - 风险分析和工具生成
    tarevoagent = TarevoAgent(args.tarevo_model, args.risk_memory)
    tarevoagent_data = data_wrapper(data, "tarevo")
    tool_results = tarevoagent.targeted_evo(args, tarevoagent_data)

    # Step 2: OptimAgent - 工具搜索、优化和执行
    optimagent = OptimAgent(args.optim_model, tool_memory_path=args.tool_memory)
    clean_relevant_tools = optimagent.search_tool(tool_results)
    tool_workflow = optimagent.optimize_tool(clean_relevant_tools)
    feedback = optimagent.execute_tool(tool_workflow, tarevoagent_data, container)

    # Step 3: DoubtAgent - 工具质疑和最终决策
    doubtagent = DoubtAgent(args.doubt_model)
    doubt_tool_result = doubtagent.doubt_tool(tool_workflow, feedback, tarevoagent_data)
    tool_set, is_unsafe = doubtagent.doubt_decision(tarevoagent_data, doubt_tool_result)

    decision = "unsafe" if is_unsafe else "safe"
    return tool_set, decision


def run(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = read_data(args.dataset)

    df = pd.DataFrame(
        {
            "input": [None] * len(data),
            "label": [None] * len(data),
            "agent_actions": [None] * len(data),
            "generated_tools": [None] * len(data),
            "decision": [None] * len(data),
            "risk_detected": [None] * len(data),
        }
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{args.dataset}", exist_ok=True)
    result_path = os.path.join(
        "results",
        args.dataset,
        f"{args.tarevo_model}_{args.optim_model}_{args.doubt_model}.csv",
    )
    if args.restart and os.path.exists(result_path):
        logger.info(f"Restart mode: removing existing results at {result_path}")
        os.remove(result_path)

    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path)
        df = existing_df

    for index, item in tqdm(enumerate(data), desc="Pipeline Running", total=len(data)):
        if df.iloc[index]["decision"] is not None and pd.notna(
            df.iloc[index]["decision"]
        ):
            continue

        try:
            container = Container()
            toolset, decision = pipeline(args, item, container)
            df.at[index, "input"] = item.get("description", "")
            df.at[index, "label"] = item.get("labels", "")
            df.at[index, "agent_actions"] = item.get("agent_actions", "")
            df.at[index, "generated_tools"] = json.dumps(
                [tool.get("tool_name") for tool in toolset]
            )
            df.at[index, "decision"] = decision
            df.at[index, "risk_detected"] = 1 if decision == "unsafe" else 0

            del container
            df.to_csv(result_path, index=False)

        except Exception as e:
            print(f"\nError processing item {index}: {str(e)}")
            # 记录错误但继续处理下一条
            df.at[index, "decision"] = "error"
            df.at[index, "risk_detected"] = -1
            df.to_csv(result_path, index=False)
            continue

    logger.info(f"\nPipeline completed! Results saved to {result_path}")

    total = len(df)
    unsafe_count = len(df[df["decision"] == "unsafe"])
    safe_count = len(df[df["decision"] == "safe"])
    error_count = len(df[df["decision"] == "error"])

    print(f"\nStatistics:")
    print(f"Total: {total}")
    print(f"Unsafe: {unsafe_count} ({unsafe_count/total*100:.2f}%)")
    print(f"Safe: {safe_count} ({safe_count/total*100:.2f}%)")
    print(f"Error: {error_count} ({error_count/total*100:.2f}%)")


if __name__ == "__main__":
    args = pipeline_config()
    logger.info(f"Pipeline Configuration: {args}")
    run(args)
