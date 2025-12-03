import json
import os
import sys
from typing import Dict, List
from safebench_prompt import TOOL_SYS_PROMPT_WITHOUTJSONFORMAT

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import Agent
from configs import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from openai import OpenAI

os.makedirs("logs", exist_ok=True)
error_log_path = "logs/error_samples.json"
exceed_log_path = "logs/exceed_round_samples.json"


class ToolAgent(Agent):
    def __init__(self, model_name: str, logger="tool_agent_logger"):
        super().__init__(model_name=model_name, logger=logger)

    def __toolcall_deepseek(self, history, tools):
        system_message = {
            "role": "system",
            "content": TOOL_SYS_PROMPT_WITHOUTJSONFORMAT,
        }
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        for _ in range(5):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=history,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                if completion is None or completion.choices is None:
                    self.logger.warning("Received empty completion from DeepSeek API.")
                    continue
                return completion
            except Exception as e:
                self.logger.error(f"Error during DeepSeek API call: {e}")
        return None

    def __toolcall_gemini(self, history, tools):
        pass

    def response(self, messages, tools):
        if not tools:
            tools = None
        if "deepseek" in self.model_name.lower():
            completion = self.__toolcall_deepseek(messages, tools)
            return completion

    def generate_response(
        self, messages: List[Dict], tool_descs: Dict[int, Dict]
    ) -> str:
        completion = self.response(messages, tool_descs)
        if completion is None:
            return None
        """
        agent的工具调用：
        completion.choices[0].message.tool_calls结构解析
        这是 OpenAI API 中工具调用（Function Calling）​ 的响应结构：
        completion.choices[0]- 取第一个（通常也是唯一的）生成结果
        .message- 消息对象
        .tool_calls- 工具调用数组，为 None表示没有工具调用
        当模型决定调用工具时，tool_calls的结构：
        tool_call = {
            "id": "call_abc123",  # 唯一标识符
            "type": "function",
            "function": {
                "name": "get_weather",  # 要调用的函数名
                "arguments": '{"location": "Beijing", "unit": "celsius"}'  # JSON格式参数
            }
}
        """
        if completion.choices[0].message.tool_calls is not None:
            tool_call = completion.choices[0].message.tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments is not None:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            response = {
                "type": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
            return response
        else:
            content = completion.choices[0].message.content
            return {"type": "text", "content": content}
        pass


def filter_data(file_path: str, output_path: str, target_key: str):
    """对于agentsafebench的数据，暂时选取其中特定risks的数据"""
    with open(file_path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            filtered_data = [
                item
                for item in data
                if isinstance(item, dict)
                and "risks" in item
                and target_key in item.get("risks", [])
            ]
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Filtered data saved to {output_path}")


def count_environment_names(file_path: str):
    """统计agentsafebench数据中environment name的分布情况"""
    from collections import Counter

    with open(file_path, "r") as f:
        data = json.load(f)
        env_names = []
        for item in data:
            if "environments" in item:
                for env in item["environments"]:
                    if "name" in env:
                        env_names.append(env["name"])
        counter = Counter(env_names)
        for name, count in counter.items():
            print(f"{name}: {count}")


def log_exceed_round_samples(messages, info):
    try:
        with open(exceed_log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        data = []
    data.append({"messages": messages, "info": info})
    with open(exceed_log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_error_samples(messages, info):
    try:
        with open(error_log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        data = []
    data.append({"messages": messages, "info": info})
    with open(error_log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # data = filter_data(
    #     "data/origin_data.json",
    #     "data/data.json",
    #     "Contribute to harmful / vulnerable code",
    # )
    count_environment_names("data/data.json")
