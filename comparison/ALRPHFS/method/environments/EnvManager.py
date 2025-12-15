import importlib
import sys
import traceback
import os
from copy import deepcopy

# 将environments目录添加到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class EnvManager:
    def __init__(self):
        pass

    def init_env(self, env_name, env_params):
        # 调试信息：打印当前sys.path
        # print(f"Current sys.path: {sys.path[:3]}")
        # print(f"Trying to import environment: {env_name}")
        try:
            """
            使用importlib动态导入与env_name同名的Python模块
            例如：env_name="OS"→ 导入OS.py模块
            """
            env_module = importlib.import_module(env_name)
            # print(f"Successfully imported module: {env_module}")
        except Exception as e:
            print(f"Failed to import environment '{env_name}': {e}")
            traceback.print_exc()
            return None

        try:
            """
            从导入的模块中获取与环境同名的类
            例如：从OS模块中获取OS类
            """
            env = getattr(env_module, env_name)
            # print(f"Successfully got class: {env}")
            return env(parameters=deepcopy(env_params))
        except Exception as e:
            print(f"Failed to instantiate environment '{env_name}': {e}")
            traceback.print_exc()
            return None
