import sys
import os
import argparse
import logging
import argparse
import json
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, Any, List
from safebench_prompt import BASIC_SYS_PROMPT, NO_TOOL_SYS_PROMPT
from environments.EnvManager import EnvManager
from tool_utils import ToolAgent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import agentsafebench_config, setup_logger
from utils import read_data
from agents import Agent

logger = setup_logger("AgentSafeBench")
env_manager = EnvManager()


def parse_envs(envs_info):
    envs = []
    tool_descs = []
    for env_info in envs_info:
        env_name = env_info["name"]
        if "parameters" in env_info:
            env_params = env_info["parameters"] or None
        else:
            env_params = None
        env = env_manager.init_env(env_name, env_params)
        if env is None:
            print(envs_info)
            raise ValueError(f"Environment {env_name} not found.")
        envs.append(env)

        tool_names = env_info["tools"]
        """
        tool_desc从baseenv中来，加载json的工具描述，比如OS.json
        这些描述会传给agent，告诉agent这些工具的功能
        """
        tool_desc = env.get_tool_descs(tool_names)
        tool_descs.extend(tool_desc)

    return envs, tool_descs


def get_system_prompt(item):
    if item is None:
        return BASIC_SYS_PROMPT
    else:
        if (
            not item["environments"]
            or "name" not in item["environments"][0]
            or item["environments"][0]["name"] == ""
        ):
            return NO_TOOL_SYS_PROMPT
        else:
            return BASIC_SYS_PROMPT


def execute_task(item: Dict[str, Any], toolagent: Agent, allow_empty: int = 0):
    max_rounds = 5
    rounds = 0
    final_result = None
    user_request = item.get("instruction", "")
    environment = item.get("environment", "")
    tool_descs = []
    messages = [{"role": "user", "content": get_system_prompt(item)}]
    if item["environments"] and item["environments"][0]["name"] != "":
        env, tool_descs = parse_envs(item["environments"])
        for item in tool_descs:
            item["type"] = "object"

        for i, item in enumerate(tool_descs):
            newitem = {"type": "function", "function": item}
            tool_descs[i] = newitem

    while rounds < max_rounds:
        completion = toolagent.generate_response(messages, tool_descs)
        rounds += 1
        if completion is None or "type" not in completion:
            continue
        if completion["type"] == "tool":
            tool_call_id = completion["tool_call_id"]
            tool_name = completion["tool_name"]
            tool_args = completion["tool_args"]

            tool_call_results = None
            for env_idx, env in enumerate(env):
                provided_tools = item["environments"][env_idx]["tools"]
                if tool_name not in provided_tools or not env.has_tool(tool_name):
                    continue
                tool_call_results = env.call_tool(tool_name, deepcopy(tool_args))
                break
            if tool_call_result is None:
                tool_call_result = {
                    "sucess": False,
                    "message": f"Tool {tool_name} not found in the provided environments.",
                }

            tool_call_response = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "arguments": json.dumps(tool_args),
                            "name": tool_name,
                        },
                    }
                ],
            }

            tool_call_result_message = {
                "role": "tool",
                "content": json.dumps(tool_call_result),
                "tool_call_id": tool_call_id,
                "name": tool_name,
            }
            messages.append(tool_call_response)
            messages.append(tool_call_result_message)

        else:
            content = completion["content"]
            messages.append({"role": "assistant", "content": content})
            final_result = messages
            break
        """
        完整的agent互相作用，包括工具调用
        """
    info = {"id": item["id"]}
    if final_result is None:
        if rounds >= max_rounds:
            """记录超出最大回合数的对话样本"""
            log_exceed_round_samples(final_result, info)
    else:
        if not allow_empty and (
            final_result[-1]["role"] != "assistant" or final_result[-1]["content"] == ""
        ):
            log_error_samples(final_result, info)
            return None, rounds
    return final_result, rounds


def run(args: argparse.Namespace):
    data = read_data("agentsafebench")
    toolagent = ToolAgent(model_name=args.model_name)
    success_count = 0
    fail_count = 0
    exceed_count = 0
    total_count = len(data)
    for index, item in tqdm(
        enumerate(data), desc="AgentSafeBench Tasks", total=len(data)
    ):
        final_result, rounds = execute_task(
            item, toolagent, allow_empty=args.allow_empty
        )
        logger.info("===========================================")
        logger.info(f"Task ID: {item['id']}")
        logger.info(f"Instruction: {item['instruction']}")
        logger.info(f"Environments: {item['environments']}")
        logger.info(f"Final Result: {final_result}")
        logger.info(f"Rounds: {rounds}")
        if final_result is not None:
            success_count += 1
        else:
            if rounds >= 5:
                exceed_count += 1
            else:
                fail_count += 1
        logger.info(
            f"Progress: {index+1}/{len(data)} | Success: {success_count} | Fail: {fail_count} | Exceed: {exceed_count}"
        )


if __name__ == "__main__":
    args = agentsafebench_config()
    run(args)
