def model_config(model):
    if model == "gpt-3.5-turbo":
        config = {
            "model": "gpt-3.5-turbo",
            "api_key": "<YOUR_API_KEY>",
            "price": [
                0.0015,
                0.002,
            ],  # [prompt_price_per_1k, completion_token_price_per_1k]
        }
    else:
        config = {
            # 详见https://gpt.zhizengzeng.com/#/my
            "model": "gpt-4",
            "api_key": "",
            "base_url": "",
            "price": [0.03, 0.06],  # GPT-4 pricing
        }

    return config


def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list
