import argparse
import logging

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0

DEEPSEEK_API_KEY = "sk-bcb9f72045274fab98c083a1901dfa31"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

ANTHROPIC_API_KEY = "your_anthropic_api_key"

GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_BASE_URL = "https://api.gemini.com"

OPENAI_API_KEY = "your_openai_api_key"
OPENAI_BASE_URL = "https://api.openai.com"


def setup_logger(logger_name: str = "general_logger") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fh = logging.FileHandler(f"logs/{logger_name}.log")
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def pipeline_config():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--dataset", type=str, default="agentsafebench")
    parser.add_argument(
        "--risk_memory", type=str, default="lifelong_library/risks_agentsafebench.json"
    )
    parser.add_argument(
        "--tool_memory",
        type=str,
        default="lifelong_library/safety_tools_agentsafebench.json",
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
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    return args


def agentsafebench_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--judge_model_name", type=str, default="deepseek-chat")
    parser.add_argument("--greedy", type=int, default=1)
    parser.add_argument("--regen_exceed", type=int, default=0)
    parser.add_argument("--extra_info", type=str, default="")
    parser.add_argument("--allow_empty", type=int, default=0)
    return parser.parse_args()
