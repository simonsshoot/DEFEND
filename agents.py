from container import Session
from typing import List, Dict, Any, Tuple
import anthropic
from openai import OpenAI
import os
import json
import re
import time
import argparse
import logging
from agents_prompts import (
    RISK_ANALYSIS_PROMPT,
    TOOL_PLAN_PROMPT,
    TOOL_SEARCH_PROMPT,
    TOOL_OPTIMIZE_PROMPT,
    TOOL_DOUBT_PROMPT,
    TOOL_DOUBT_OPTIMIZE_PROMPT,
    DOUBT_DECISION_PROMPT,
    SIMULATE_AGENT_PROMPT
)
from configs import (
    DEEPSEEK_BASE_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEEPSEEK_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from configs import setup_logger
from container import Container


class Agent:
    def __init__(self, model_name: str, logger: str = "general_logger"):
        self.model_name = model_name.lower()
        self.logger = setup_logger(logger)

    def create_session(self) -> Session:
        return Session(self.inference)

    def inference(self, History: List[dict]) -> str:
        history = History
        for h in history:
            if h["role"] == "agent":
                h["role"] = "assistant"

        if "deepseek" in self.model_name:
            return self.__call_deepseek(history)
        elif "claude" in self.model_name:
            return self.__call_claude(history)
        elif "gemini" in self.model_name:
            return self.__call_gemini(history)
        else:
            return self.__call_openai(history)

    def extract_json(self, response: str) -> Dict[str, Any]:
        cleaned = response.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned
        if "```json" in cleaned:
            match = re.search(r"```json(.*?)```", cleaned, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
                return json_content
        if "```" in cleaned:
            match = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content.startswith("{") and content.endswith("}"):
                    return content
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = cleaned[start : end + 1]
                json.loads(json_str)
                return json_str
        except:
            self.logger.warning(
                "Failed to extract JSON from response. Return the original response."
            )
        return cleaned

    def __call_deepseek(self, history: List[dict]) -> str:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return completion.choices[0].message.content

    def __call_claude(self, history: List[dict]) -> str:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # 注意下history是一个list，里面是多个dict 是这样的嘛：
        """
     {"role": "system", "content": "你是一个专业的 AI 助手"},
        {"role": "user", "content": "解释什么是量子纠缠"}
     """
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return completion.choices[0].message.content

    def __call_gemini(self, history: List[dict]) -> str:
        client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return completion.choices[0].message.content

    def __call_openai(self, history: List[dict]) -> str:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return completion.choices[0].message.content


class TarevoAgent(Agent):
    def __init__(
        self, model_name, risk_memory_path: str = "lifelong_library/risks.json"
    ):
        super().__init__(model_name, logger="tarevo_agent_logger")
        self.risk_memory_path = risk_memory_path

    def __load_riskmemory(self):
        with open(self.risk_memory_path, "r") as f:
            risk_categories_json = json.load(f)
        risk_categories_str = json.dumps(
            risk_categories_json, indent=2, ensure_ascii=False
        )
        return risk_categories_str

    def risk_analysis(
        self, request: str, agent_actions: str, risk_categories: str
    ) -> Dict[str, Any]:
        """
        analysis_result:
        {{
        "risks": [
        {{
        "category": "系统安全风险",
        "description": "具体描述1"
        }},
        {{
        "category": "权限提升风险",
        "description": "具体描述2"
        }}
        ],
        "new_risks": "yes/no"
        }}
        """
        prompt = RISK_ANALYSIS_PROMPT.format(
            request=request,
            agent_actions=agent_actions,
            risk_categories=risk_categories,
        )
        messages = [
            {
                "role": "system",
                "content": "你是一个专业而严谨的安全专家，专注于风险分析。",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.inference(messages)

        clean_response = self.extract_json(response)

        analysis_result = json.loads(clean_response)
        return analysis_result

    def safetytool_definition(
        self, risk: Dict[str, Any], user_request: str, agent_actions: str
    ) -> Dict[str, Any]:
        prompt = TOOL_PLAN_PROMPT.format(
            request=user_request,
            agent_actions=agent_actions,
            risk_analysis=json.dumps(risk, indent=2, ensure_ascii=False),
        )
        messages = [
            {
                "role": "system",
                "content": "你是一个安全工具开发专家，专注于设计防护工具。",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.inference(messages)
        clean_response = self.extract_json(response)
        clean_response = json.loads(clean_response)
        return clean_response

    def targeted_evo(self, args: argparse.Namespace, data: List[Dict[str, Any]]):
        """目标进化"""
        user_request = data["request"]
        agent_actions = data["agent_actions"]
        user_identity = data["user_identity"]
        init_script = data["init"]
        label = data["labels"]

        analysis_result = self.risk_analysis(
            request=user_request,
            agent_actions=agent_actions,
            risk_categories=self.__load_riskmemory(),
        )
        print("===========================================")
        self.logger.info(f"Risk Analysis Result: {analysis_result}")
        time.sleep(2)
        tool_results = self.safetytool_definition(
            analysis_result, user_request, agent_actions
        )
        print("===========================================")
        self.logger.info(f"Generated Safety Tool: {tool_results}")

        # 返回 tool_results 和 analysis_result 用于后续更新 lifelong library
        return tool_results, analysis_result


"""
tool_result:
{{
"tools": [
{{
"category": "System security risk",
"tool_name": "root_ownership_protector",
"tool_description": "A tool that prevents unauthorized changes to the ownership of critical system directories, ensuring system stability.",
"require": ["re"],
"tool_code": "def root_ownership_protector(command: str):\\n cmd = command.strip().lower()\\n high_risk_patterns = [\\n r'\\bchown\\b[^;&|]\\s+-r[^;&|]\\s+/\\s$',\\n r'\\bchown\\b[^;&|]\\s+/\\s$',\\n r'\\bchmod\\b[^;&|]\\s+-r[^;&|]\\s+/\\s$',\\n r'\\bchmod\\b[^;&|]\\s+/\\s$'\\n ]\\n for pat in high_risk_patterns:\\n if re.search(pat, cmd):\\n return True\\n return False"
}}
]
}}
"""


class OptimAgent(Agent):
    def __init__(
        self, model_name, tool_memory_path: str = "lifelong_library/tool_memory.json"
    ):
        super().__init__(model_name, logger="optim_agent_logger")
        self.tool_memory_path = tool_memory_path

    def __load_toolmemory(self):
        with open(self.tool_memory_path, "r") as f:
            tool_categories_json = json.load(f)
        return tool_categories_json

    def load_tool(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        categories = [tool["category"] for tool in tool_result.get("tools", [])]
        relevant_tools: Dict[str, Any] = {}
        tool_memory = self.__load_toolmemory()

        for cat in categories:
            if cat in tool_memory:
                relevant_tools[cat] = tool_memory[cat]

        return relevant_tools

    def search_tool(self, tool_result: Dict[str, Any]):
        """
        tool_result format:
        {{
          "tools": [
            {{
              "category": "<string>",
              "tool_name": "<string>",
              "tool_description": "<string>",
              "require": ["<string>"],
              "tool_code": "<string>",
              "if_match": "yes/no",
              "match_tool_name": "<string>"
            }}
          ]
        }}
        """
        relevant_tools = self.load_tool(tool_result)
        prompt = TOOL_SEARCH_PROMPT.format(
            user_tools=json.dumps(tool_result, indent=2, ensure_ascii=False),
            existing_tools=json.dumps(relevant_tools, indent=2, ensure_ascii=False),
        )
        messages = [
            {
                "role": "system",
                "content": "你是一个检索专家，负责检索现有的安全工具库中有没有与用户提出的工具相似的工具，进而避免后续重复开发。",
            },
            {"role": "user", "content": prompt},
        ]
        time.sleep(2)
        response = self.inference(messages)
        clean_response = self.extract_json(response)
        clean_response = json.loads(clean_response)
        return clean_response

    def optimize_tool(
        self, clean_relevant_tools: Dict[str, Any]
    ) -> List[Tuple[Dict, bool]]:
        # 这里还应该形成最终执行的工具集合
        tool_category_json = self.__load_toolmemory()
        tool_workflow: List[Tuple[Dict, bool]] = []

        for item in clean_relevant_tools["tools"]:
            matched_tool_info = None

            if item.get("is_match") == "yes":
                category = item.get("category")
                matched = item.get("match_tool_name")

                if category in tool_category_json:
                    for cate_tool in tool_category_json[category]:
                        if cate_tool.get("tool_name") == matched:
                            matched_tool_info = cate_tool
                            break

            if matched_tool_info is None:
                # 没有匹配的工具，直接使用新工具
                tool_workflow.append((item, False))
            else:
                # 有匹配的工具，进行优化
                prompt = TOOL_OPTIMIZE_PROMPT.format(
                    user_tools=json.dumps(item, indent=2, ensure_ascii=False),
                    existing_tools=json.dumps(
                        matched_tool_info, indent=2, ensure_ascii=False
                    ),
                )
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个安全工具优化专家，负责根据用户提出的工具，优化现有的相似工具，目的在提高工具库的精简性同时兼容用户提出的工具。",
                    },
                    {"role": "user", "content": prompt},
                ]
                time.sleep(2)
                response = self.inference(messages)
                clean_response = self.extract_json(response)
                clean_response = json.loads(clean_response)
                tool_workflow.append((clean_response, True))

        return tool_workflow

    def execute_tool(
        self,
        tool_workflow: List[Tuple[Dict, bool]],
        data: Dict[str, Any],
        environment: Container,
    ) -> List[Tuple[Dict, bool, bool]]:
        """
        在 Docker 环境中执行安全工具

        Args:
          tool_workflow: List of (tool_info, is_optimized) tuples
          data: 包含请求信息的字典
          environment: Docker 容器环境

        Returns:
          List of tuples: (tool_info, is_optimized, execution_result)
          - tool_info: 工具信息字典
          - is_optimized: 是否为优化后的工具
          - execution_result: True 表示检测到风险，False 表示安全
        """
        user_identity = data["user_identity"]
        agent_identity = "root"
        command = data["agent_actions"]

        test = environment.container.exec_run("whoami", user=user_identity)
        self.logger.info(f"User identity: {test.output.decode().strip()}")
        test = environment.container.exec_run("whoami", user=agent_identity)
        self.logger.info(f"Agent identity: {test.output.decode().strip()}")

        if data["init"] != "":
            self.logger.info(f"Executing init script: {data['init']}")
            environment.execute_init(data["init"], user=user_identity)

        # 执行每个安全工具并收集结果
        execution_results: List[Tuple[Dict, bool, bool]] = []

        for tool_item in tool_workflow:
            tool_info, is_optimized = tool_item

            tool_name = tool_info.get("tool_name", "unknown_tool")
            tool_code = tool_info.get("tool_code", "")
            require = tool_info.get("require", [])

            # self.logger.info(f"Executing tool: {tool_name}")
            # self.logger.info(f"Is optimized: {is_optimized}")

            if require:
                self.logger.info(f"Installing requirements: {require}")
                install_success = environment.install_packages(
                    require, user=agent_identity
                )
                if not install_success:
                    self.logger.warning(
                        f"Failed to install requirements for tool: {tool_name}"
                    )
                    # 安装失败视为检测到风险
                    execution_results.append((tool_info, is_optimized, True))
                    continue

            try:
                success, result, error = environment.execute_python_code(
                    code=tool_code,
                    function_name=tool_name,
                    command=command,
                    require=require,
                    user=user_identity,
                )

                if not success:
                    self.logger.warning(f"Tool {tool_name} execution failed: {error}")
                    # 执行失败视为检测到风险
                    execution_results.append((tool_info, is_optimized, True))
                else:
                    # 工具返回 True 表示检测到风险，False 表示安全
                    risk_detected = bool(result)
                    # if risk_detected:
                    #     self.logger.warning(
                    #         f"Tool {tool_name} detected risk in command: {command}"
                    #     )
                    # else:
                    #     self.logger.info(f"Tool {tool_name} passed: no risk detected")
                    execution_results.append((tool_info, is_optimized, risk_detected))

            except Exception as e:
                self.logger.error(
                    f"Exception while executing tool {tool_name}: {str(e)}"
                )
                execution_results.append((tool_info, is_optimized, True))

        return execution_results

    def debug_tool(self):
        pass


class DoubtAgent(Agent):
    def __init__(
        self, model_name, tool_memory_path: str = "lifelong_library/tool_memory.json"
    ):
        super().__init__(model_name, logger="double_agent_logger")
        self.tool_memory_path = tool_memory_path

    def __load_toolmemory(self):
        with open(self.tool_memory_path, "r") as f:
            tool_categories_json = json.load(f)
        return tool_categories_json

    def search_original_tool(self, optimized_tool: Dict[str, Any]) -> Dict[str, Any]:
        tool_memory = self.__load_toolmemory()
        category_tool_set = optimized_tool.get("category")
        for tool in tool_memory.get(category_tool_set, []):
            if tool.get("tool_name") == optimized_tool.get("tool_name"):
                return tool
        pass

    def doubt_tool(
        self,
        tool_workflow: List[Tuple[Dict, bool]],
        feedback: List[Tuple[Dict, bool, bool]],
        data: Dict[str, Any],
    ) -> List[Tuple[Dict, Dict, bool, bool]]:
        doubt_results = []
        for item in feedback:
            tool_info, is_optimized, execution_result = item
            if not is_optimized:
                prompt = TOOL_DOUBT_PROMPT.format(
                    request=data["request"],
                    agent_actions=data["agent_actions"],
                    tool=json.dumps(tool_info, indent=2, ensure_ascii=False),
                )
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个安全工具审核专家，用户的请求与代理执行的操作可能含有风险，需要使用安全工具进行防护。",
                    },
                    {"role": "user", "content": prompt},
                ]
                time.sleep(2)
                response = self.inference(messages)
                clean_response = self.extract_json(response)
                clean_response = json.loads(clean_response)

            elif is_optimized:
                origin_tool = self.search_original_tool(tool_info)
                prompt = TOOL_DOUBT_OPTIMIZE_PROMPT.format(
                    request=data["request"],
                    agent_actions=data["agent_actions"],
                    tool=json.dumps(origin_tool, indent=2, ensure_ascii=False),
                    optimized_tool=json.dumps(
                        origin_tool, indent=2, ensure_ascii=False
                    ),
                )
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个安全工具改进专家，用户的请求与代理执行的操作可能含有风险，现有的安全工具可被重用和优化来检测这些风险。",
                    },
                    {"role": "user", "content": prompt},
                ]
                time.sleep(2)
                response = self.inference(messages)
                clean_response = self.extract_json(response)
                clean_response = json.loads(clean_response)
            doubt_results.append(
                (tool_info, clean_response, is_optimized, execution_result)
            )
        return doubt_results

    def doubt_decision(
        self,
        data: Dict[str, Any],
        doubt_tool_result: List[Tuple[Dict, Dict, bool, bool]],
    ) -> Tuple[List[Dict], bool]:
        tool_set = []
        for item in doubt_tool_result:
            tool_set.append(item[0])
        for item in doubt_tool_result:
            # execution_result: True 表示检测到风险，False 表示安全
            if item[3] == True:
                return tool_set, True
            if item[1].get("is_safe") == "False":
                return tool_set, True
        prompt = DOUBT_DECISION_PROMPT.format(
            request=data["request"],
            agent_actions=data["agent_actions"],
            approved_tools=json.dumps(tool_set, indent=2, ensure_ascii=False),
        )
        messages = [
            {
                "role": "system",
                "content": "你是一个安全工具最终审核专家",
            },
            {"role": "user", "content": prompt},
        ]
        time.sleep(2)
        response = self.inference(messages)
        clean_response = self.extract_json(response)
        clean_response = json.loads(clean_response)
        if clean_response.get("is_safe") == "False":
            return tool_set, True
        return tool_set, False


"""
tool_info:
{{
  "category": "<string>",
  "tool_name": "<string>",                
  "tool_description": "<string>",         
  "require": ["<string>"],                 
  "tool_code": "<string>"
}}

"""

class SimulateAgent(Agent):
    def __init__(self, model_name, logger = "simulate_logger"):
        super().__init__(model_name, logger)

    def simulate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = SIMULATE_AGENT_PROMPT.format(
            request=data["request"],
        )
        messages = [
            {
                "role": "system",
                "content": "你是一个模拟代理，负责输出代理在执行用户请求时可能会执行的操作。",
            },
            {"role": "user", "content": prompt},
        ]
        time.sleep(2)
        response = self.inference(messages)
        clean_response = self.extract_json(response)
        clean_response = json.loads(clean_response)
        simu_data=data
        simu_data["agent_actions"]=clean_response.get("agent_actions","")
        simu_data["init"]=clean_response.get("environment","")
        return simu_data
    
class SimulateMultiRoundAgent(Agent):
    def __init__(self, model_name, logger = "simulate_multi_round_logger"):
        super().__init__(model_name, logger)
        self.history=[]

    def multi_round_simulate(self, data: Dict[str, Any]) -> Dict[str,Any]:
        imple_data = {**data, "agent_actions": [""], "init": [""]}
        prompts_set=data.get("description","")
        system_prompt = {
            "role": "system", 
            "content": """你是一个模拟代理，负责与用户进行多轮交互。在每轮对话中，你需要根据用户的请求和对话历史，输出代理可能会执行的操作。请确保你的响应是连贯的，并考虑之前的对话上下文。"""
        }
        self.history.append(system_prompt)
        user_initial_prompt = {
            "role": "user",
            "content": SIMULATE_AGENT_PROMPT.format(
                request=data["request"],
            )
        }
        self.history.append(user_initial_prompt)
        for round_num, prompt in enumerate(prompts_set,0):
            try:
                if round_num==0:
                    self.history.append(user_initial_prompt)
                else:
                    user_round_prompt = {
                        "role": "user",
                        "content": prompt
                    }
                    self.history.append(user_round_prompt)
                agent_response=self.inference(self.history)
                clean_response = self.extract_json(agent_response)
                clean_response = json.loads(clean_response)
                imple_data["agent_actions"].append(clean_response.get("agent_actions",""))
                imple_data["init"].append(clean_response.get("environment",""))

                agent_message={
                    "role":"assistant",
                    "content":clean_response    
                }
                self.history.append(agent_message)

            except Exception as e:
                self.logger.error(f"Exception while multi round simulate: {str(e)}")
                break
        return imple_data

    def __warp_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        '''imple_data中多步的agent_actions和init是list，转为str， description也是'''
        description_list=data.get("description",[])
        action_list=data.get("agent_actions",[])
        init_list=data.get("init",[])
        wrap_data=Dict[str,Any]()
        wrap_data["user_identity"]=data.get("user_identity","")
        wrap_data["labels"]=data.get("labels","")
        wrap_data["request"]=" && ".join(description_list)
        wrap_data["agent_actions"]=" && ".join(action_list)
        wrap_data["init"]=" && ".join(init_list)
        return wrap_data

    def simulate(self, data: Dict[str, Any]) -> str:
        pass

