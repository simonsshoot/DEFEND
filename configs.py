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
    parser.add_argument("--dataset", type=str, default="system_sabotage")
    parser.add_argument(
        "--risk_memory", type=str, default="lifelong_library/risks_os.json"
    )
    parser.add_argument(
        "--tool_memory", type=str, default="lifelong_library/safety_tools_os.json"
    )
    parser.add_argument("--tarevo_model", type=str, default="deepseek-chat")
    parser.add_argument("--optim_model", type=str, default="deepseek-chat")
    parser.add_argument("--doubt_model", type=str, default="deepseek-chat")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    return args
