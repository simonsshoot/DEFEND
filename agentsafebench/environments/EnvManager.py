import importlib
import sys
import traceback
import os
from copy import deepcopy

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "environments")
)


class EnvManager:
    def __init__(self):
        pass

    def init_env(self, env_name, env_params):
        # print(sys.path)
        try:
            """
            使用importlib动态导入与env_name同名的Python模块
            例如：env_name="OS"→ 导入OS.py模块
            """
            env_module = importlib.import_module(f"environments.{env_name}")
        except Exception as e:
            print(e)
            traceback.print_exc()
            return None
        # print(env_module)
        """
        从导入的模块中获取与环境同名的类
        例如：从OS模块中获取OS类
        """
        env = getattr(env_module, env_name)

        return env(parameters=deepcopy(env_params))
