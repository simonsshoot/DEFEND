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

如何抵御提示注入攻击？
提示注入攻击的核心是利用LLM智能体对环境信息的信任依赖。智能体在交互时会从环境（如文件系统、数据库、网页）中读取信息作为输入，攻击者通过在这些信息中隐藏恶意指令，实现“劫持”智能体的目的。例如：
在文件路径或文档中嵌入误导性文本，迫使智能体忽略原始用户请求。
在网页HTML中插入隐藏元素，诱骗智能体泄露用户隐私数据。

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
from utils import (
    read_data,
    data_wrapper,
    simulate_data_wrapper,
    read_simulated_data,
    normalize_is_safe,
)
from container import Container
from agents import (
    Agent,
    TarevoAgent,
    OptimAgent,
    DoubtAgent,
    SimulateAgent,
    SandBoxAgent,
)
from configs import setup_logger

logger = setup_logger("pipeline")


def update_lifelong_library(
    args: argparse.Namespace,
    risk_analysis: Dict[str, Any],
    tool_set: List[Dict],
    doubt_tool_result: List[Tuple[Dict, Dict, bool, bool]],
):
    """
    更新 lifelong library
    1. 添加新识别的风险到 risks.json
    2. 添加新生成的工具或覆盖优化后的工具到 safety_tools.json
    """
    if risk_analysis.get("new_risks") == "yes":
        try:
            with open(args.risk_memory, "r", encoding="utf-8") as f:
                risk_library = json.load(f)

            # 添加新风险
            for risk in risk_analysis.get("risks", []):
                category = risk.get("category")
                description = risk.get("description")

                if category and category not in risk_library:
                    risk_library[category] = {
                        "category": category,
                        "description": description,
                    }
                    logger.info(f"Added new risk category: {category}")
            with open(args.risk_memory, "w", encoding="utf-8") as f:
                json.dump(risk_library, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to update risk library: {str(e)}")

    try:
        with open(args.tool_memory, "r", encoding="utf-8") as f:
            tool_library = json.load(f)

        updated = False

        for item in doubt_tool_result:
            tool_info, doubt_result, is_optimized, execution_result = item

            if normalize_is_safe(doubt_result.get("is_safe", "")) != True:
                continue

            category = tool_info.get("category")
            tool_name = tool_info.get("tool_name")

            if not category or not tool_name:
                continue

            if category not in tool_library:
                tool_library[category] = []

            tool_data = {
                "tool_name": tool_name,
                "tool_description": tool_info.get("tool_description", ""),
                "require": tool_info.get("require", []),
                "tool_code": tool_info.get("tool_code", ""),
                "risk_description": tool_info.get("risk_description", ""),
            }

            if is_optimized:
                found = False
                for i, existing_tool in enumerate(tool_library[category]):
                    if existing_tool.get("tool_name") == tool_name:
                        tool_library[category][i] = tool_data
                        logger.info(
                            f"Updated optimized tool: {tool_name} in category {category}"
                        )
                        found = True
                        updated = True
                        break

                if not found:
                    tool_library[category].append(tool_data)
                    logger.info(
                        f"Added optimized tool (not found original): {tool_name} in category {category}"
                    )
                    updated = True
            else:
                exists = any(
                    t.get("tool_name") == tool_name for t in tool_library[category]
                )

                if not exists:
                    tool_library[category].append(tool_data)
                    logger.info(f"Added new tool: {tool_name} in category {category}")
                    updated = True
        if updated:
            with open(args.tool_memory, "w", encoding="utf-8") as f:
                json.dump(tool_library, f, indent=2, ensure_ascii=False)
            logger.info(f"Tool library updated successfully")

    except Exception as e:
        logger.error(f"Failed to update tool library: {str(e)}")


def get_present_tools(dataset: str, data: Dict[str, Any]) -> List[Dict]:
    """
    根据数据集名称，返回对应的现有工具列表(这里的工具是执行工具，不是安全工具)
    从AgentSafeBench的environments字段中提取工具描述
    从AgentHarm的target_functions字段中提取工具描述
    """
    if dataset == "agentsafebench":
        # 导入AgentSafeBench的环境管理器
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "agentsafebench"))
        from environments.EnvManager import EnvManager

        env_manager = EnvManager()
        present_tools = []

        environments = data.get("environments", [])
        if not environments or not environments[0].get("name"):
            return []

        try:
            # 解析每个环境
            for env_info in environments:
                env_name = env_info.get("name", "")
                if not env_name:
                    continue

                env_params = env_info.get("parameters") or None
                env = env_manager.init_env(env_name, env_params)

                if env is None:
                    logger.warning(f"Environment {env_name} not found.")
                    continue

                tool_names = env_info.get("tools", [])
                # 获取工具描述（参考run.py的parse_envs函数）
                tool_descs = env.get_tool_descs(tool_names)
                for tool_desc in tool_descs:
                    present_tools.append(
                        {
                            "environment": env_name,
                            "tool_name": tool_desc.get("name", ""),
                            "description": tool_desc.get("description", ""),
                            "parameters": tool_desc.get("parameters", {}),
                            "env_instance": env,
                        }
                    )
        except Exception as e:
            logger.error(f"Error parsing present tools: {str(e)}")
            return []

        return present_tools

    elif dataset == "agentharm" or dataset == "agentharm_benign":
        # AgentHarm数据集，从target_functions中提取工具
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agentharm"))

        try:
            # AgentHarm工具是通过@tool装饰器定义的函数
            # 动态导入目标工具
            if dataset == "agentharm":
                tools_package = "benchmark.harmful_tools"
            else:
                tools_package = "benchmark.benign_tools"

            target_functions = data.get("agentharm_target_functions", [])

            present_tools = []
            for func_name in target_functions:
                try:
                    # 导入对应的工具模块
                    module_path = f"{tools_package}.{func_name}"
                    tool_module = __import__(module_path, fromlist=[func_name])

                    # 获取工具函数
                    if hasattr(tool_module, func_name):
                        tool_func = getattr(tool_module, func_name)

                        # 调用tool函数获取Tool对象
                        tool_obj = tool_func()

                        # 提取工具描述和参数
                        import inspect

                        if hasattr(tool_obj, "__wrapped__"):
                            sig = inspect.signature(tool_obj.__wrapped__)
                            doc = inspect.getdoc(tool_obj.__wrapped__) or ""
                        else:
                            sig = inspect.signature(tool_obj)
                            doc = inspect.getdoc(tool_obj) or ""

                        # 构建参数信息
                        params = {}
                        for param_name, param in sig.parameters.items():
                            params[param_name] = {
                                "type": (
                                    str(param.annotation)
                                    if param.annotation != inspect.Parameter.empty
                                    else "any"
                                ),
                                "required": param.default == inspect.Parameter.empty,
                            }

                        present_tools.append(
                            {
                                "environment": "agentharm",
                                "tool_name": func_name,
                                "description": doc,
                                "parameters": {"properties": params},
                                "tool_function": tool_obj,
                            }
                        )
                        logger.info(f"Loaded AgentHarm tool: {func_name}")
                    else:
                        logger.warning(f"Tool function {func_name} not found in module")
                except Exception as e:
                    logger.error(f"Error loading tool {func_name}: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    continue

            return present_tools
        except Exception as e:
            logger.error(f"Error loading AgentHarm tools: {str(e)}")
            import traceback

            traceback.print_exc()
            return []
    else:
        return []


def pipeline(
    args: argparse.Namespace, data: Dict[str, Any], container: Container
) -> Tuple[
    List[Dict], str, Dict[str, Any], List[Tuple[Dict, Dict, bool, bool]], Dict[str, Any]
]:
    # Step 1: 数据预处理
    if args.need_simulate:
        # 需要模拟：先包装数据，然后模拟
        wrap_data = data_wrapper(data, args.dataset)
        present_tools = get_present_tools(args.dataset, wrap_data)
        simulateagent = SimulateAgent(args.simulate_model)
        tarvodata = simulateagent.simulate(wrap_data, present_tools)

        # 如果开启调试模式，保存模拟后的数据
        if args.debug_mode:
            with open(args.debug_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(tarvodata, ensure_ascii=False) + "\n")
    else:
        # 不需要模拟：数据已经包含 agent_actions 和 init
        # 直接使用预模拟的数据
        tarvodata = data
        # 提取 present_tools（仍然需要用于后续流程）
        present_tools = get_present_tools(args.dataset, tarvodata)

    tarevoagent = TarevoAgent(
        args.tarevo_model, args.risk_memory, args.permission_policy
    )

    if "user_level" not in tarvodata:
        user_identity = tarvodata.get("user_identity", "user")
        tarvodata["user_level"] = "root" if user_identity == "root" else "user"
    # new add
    tool_results, risk_analysis = tarevoagent.targeted_evo(
        args, tarvodata, present_tools
    )

    # Step 2: OptimAgent - 工具搜索、优化和执行
    optimagent = OptimAgent(args.optim_model, tool_memory_path=args.tool_memory)
    clean_relevant_tools = optimagent.search_tool(tool_results)
    tool_workflow = optimagent.optimize_tool(clean_relevant_tools)

    sandboxagent = SandBoxAgent(args.sandbox_model)
    # new add
    feedback = sandboxagent.sandbox_execute_tool(
        tool_workflow, tarvodata, present_tools, args.dataset
    )
    agent_action_result = sandboxagent.sandbox_execute_action(
        tarvodata, present_tools, args.dataset
    )
    # feedback = optimagent.execute_tool(tool_workflow, tarvodata, container)
    # agent_action_result = optimagent.execute_action(tarvodata, container)

    final_result = agent_action_result[-1] if len(agent_action_result) > 0 else ""

    # Step 3: DoubtAgent - 工具质疑和最终决策
    doubtagent = DoubtAgent(args.doubt_model, args.tool_memory, args.permission_policy)
    """doublt_tool_result:(tool_info, clean_response, is_optimized, execution_result)"""
    # new add
    doubt_tool_result = doubtagent.doubt_tool(
        tool_workflow, feedback, tarvodata, present_tools
    )
    tool_set, is_unsafe = doubtagent.doubt_decision(
        tarvodata,
        doubt_tool_result,
        final_result,
        args.debug_mode,
        args.debug_doubt_tool_path,
        args.debug_decision_path,
    )

    decision = "unsafe" if is_unsafe else "safe"

    return tool_set, decision, risk_analysis, doubt_tool_result, tarvodata


def run(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.need_simulate:
        data = read_data(args.dataset)
    else:
        data = read_simulated_data(args.dataset)

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

    if not os.path.exists(args.risk_memory):
        with open(args.risk_memory, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
    if not os.path.exists(args.tool_memory):
        with open(args.tool_memory, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)

    new_tool_generated = 0
    optimized_tool = 0
    fail_count = 0
    need_tools_count = 0  # 需要生成工具的次数
    no_tools_needed_count = 0  # 不需要生成工具的次数
    risky_tools_count = 0  # 生成的工具中有风险的次数

    for index, item in tqdm(enumerate(data), desc="Pipeline Running", total=len(data)):
        if df.iloc[index]["decision"] is not None and pd.notna(
            df.iloc[index]["decision"]
        ):
            continue

        try:
            container = Container()
            tool_set, decision, risk_analysis, doubt_tool_result, tarvodata = pipeline(
                args, item, container
            )

            need_tools = risk_analysis.get("need_tools", "no")
            if need_tools == "yes":
                need_tools_count += 1
            else:
                no_tools_needed_count += 1

            for result in doubt_tool_result:
                tool_info, doubt_result, is_optimized, execution_result, risky_tools = (
                    result
                )
                risky_tools_count += risky_tools
                if normalize_is_safe(doubt_result.get("is_safe", "")) == True:
                    if is_optimized:
                        optimized_tool += 1
                    else:
                        new_tool_generated += 1
                elif normalize_is_safe(doubt_result.get("is_safe", "")) == False:
                    fail_count += 1
                    with open(args.fail_tool_debug, "a", encoding="utf-8") as f:
                        debug_info = {
                            "input": item.get("description", ""),
                            "tool_info": tool_info,
                            "doubt_result": doubt_result,
                            "is_optimized": is_optimized,
                            "execution_result": execution_result,
                        }
                        f.write(json.dumps(debug_info, ensure_ascii=False) + "\n")

            if args.dataset == "asb_harmful" or args.dataset == "asb_benign":
                input_text = item.get("instruction", "")
                label = item.get("fulfillable", 0)
            elif args.dataset in ["agentharm", "agentharm_benign"]:
                input_text = item.get("prompt", "")
                label = 0 if args.dataset == "agentharm" else 1
            else:
                input_text = item.get("description", "")
                label = item.get("labels", "")

            df.at[index, "input"] = input_text
            df.at[index, "label"] = label
            df.at[index, "agent_actions"] = tarvodata.get("agent_actions", "")
            df.at[index, "generated_tools"] = json.dumps(
                [tool.get("tool_name") for tool in tool_set]
            )
            df.at[index, "decision"] = decision
            df.at[index, "risk_detected"] = 1 if decision == "unsafe" else 0

            update_lifelong_library(args, risk_analysis, tool_set, doubt_tool_result)

            del container
            df.to_csv(result_path, index=False)

        except Exception as e:
            logger.error(f"\nError processing item {index}: {str(e)}")
            import traceback

            traceback.print_exc()
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
    misjudge_count = len(df[df["label"] == 0][df["decision"] == "unsafe"])

    print(f"\nStatistics:")
    print(f"Total: {total}")
    print(f"Unsafe: {unsafe_count} ({unsafe_count/total*100:.2f}%)")
    print(f"Safe: {safe_count} ({safe_count/total*100:.2f}%)")
    print(f"Error: {error_count} ({error_count/total*100:.2f}%)")
    print(
        f"Misjudge (Safe but judged Unsafe): {misjudge_count} ({misjudge_count/total*100:.2f}%)"
    )

    # 工具生成决策统计
    print(f"\nTool Generation Decision:")
    decision_total = need_tools_count + no_tools_needed_count
    if decision_total > 0:
        print(
            f"Need tools: {need_tools_count}/{decision_total} ({need_tools_count/decision_total*100:.2f}%)"
        )
        print(
            f"No tools needed: {no_tools_needed_count}/{decision_total} ({no_tools_needed_count/decision_total*100:.2f}%)"
        )

    # 工具重用率和失败率统计
    tool_total = new_tool_generated + optimized_tool + fail_count
    if tool_total > 0:
        print(f"\nTool Statistics:")
        print(
            f"Newly generated tool ratio: {new_tool_generated}/{tool_total} ({new_tool_generated/tool_total*100:.2f}%)"
        )
        print(
            f"Optimized tool ratio: {optimized_tool}/{tool_total} ({optimized_tool/tool_total*100:.2f}%)"
        )
        print(
            f"Failed tool ratio: {fail_count}/{tool_total} ({fail_count/tool_total*100:.2f}%)"
        )
        print(
            f"Risky tool ratio: {risky_tools_count}/{tool_total} ({risky_tools_count/tool_total*100:.2f}%)"
        )
    else:
        print("\nNo tools generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="agentharm")
    parser.add_argument(
        "--risk_memory",
        type=str,
        default="lifelong_library/risks_agentharm_new.json",
    )
    parser.add_argument(
        "--tool_memory",
        type=str,
        default="lifelong_library/safety_tools_agentharm_new.json",
    )
    parser.add_argument(
        "--permission_policy",
        type=str,
        default="permission_policy.json",
        help="Path to the permission policy JSON file",
    )
    parser.add_argument("--simulate_model", type=str, default="deepseek-chat")
    parser.add_argument("--tarevo_model", type=str, default="deepseek-chat")
    parser.add_argument("--optim_model", type=str, default="deepseek-chat")
    parser.add_argument("--doubt_model", type=str, default="deepseek-chat")
    parser.add_argument("--sandbox_model", type=str, default="deepseek-chat")
    parser.add_argument(
        "--fail_tool_debug", type=str, default="results/fail_tool_debug2.json"
    )
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument(
        "--debug_file", type=str, default="results/simulate_agentharm.jsonl"
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--need_simulate", action="store_true")
    parser.add_argument(
        "--debug_doubt_tool_path",
        type=str,
        default="debug_doubt_tool_agentharmbenign.log",
        help="Path to save debug tools if in debug mode",
    )
    parser.add_argument(
        "--debug_decision_path",
        type=str,
        default="debugs/debug_decision_agentharmbenign.log",
        help="tool pass but decision refused debug path",
    )
    args = parser.parse_args()
    logger.info(f"Pipeline Configuration: {args}")
    run(args)
