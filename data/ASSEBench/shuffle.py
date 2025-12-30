import json
import random
from pathlib import Path

def sample_json_by_label(src_file: str,
                         dst_file: str = "security50.json",
                         label0_cnt: int = 50,
                         label1_cnt: int = 50,
                         seed: int = 42) -> None:
    """
    从原始 JSON 文件中随机抽取指定数量的 label=0 和 label=1 样本，
    合并后写入新的 JSON 文件。

    参数
    ----
    src_file : str
        原始 JSON 文件路径（要求最外层是 list）
    dst_file : str
        输出文件路径，默认为 sampled_50_50.json
    label0_cnt : int
        需要抽取的 label=0 样本数
    label1_cnt : int
        需要抽取的 label=1 样本数
    seed : int
        随机种子，保证可复现
    """
    random.seed(seed)

    # 读取原始数据
    with open(src_file, encoding="utf-8") as f:
        data = json.load(f)

    # 按 label 分组
    label0_pool = [d for d in data if d.get("label") == 0]
    label1_pool = [d for d in data if d.get("label") == 1]

    # 抽样
    sample0 = random.sample(label0_pool, k=min(label0_cnt, len(label0_pool)))
    sample1 = random.sample(label1_pool, k=min(label1_cnt, len(label1_pool)))

    if len(sample0) < label0_cnt:
        print(f"Warning: label=0 只有 {len(sample0)} 条，已全取。")
    if len(sample1) < label1_cnt:
        print(f"Warning: label=1 只有 {len(sample1)} 条，已全取。")

    # 合并 & 写出
    sampled = sample0 + sample1
    random.shuffle(sampled)          # 打乱顺序
    with open(dst_file, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"已写入 {len(sampled)} 条样本到 {Path(dst_file).resolve()}")

# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    sample_json_by_label(r"G:\agent实验系列\DEFEND\data\ASSEBench\dataset\AgentJudge-security.json")   # 改成自己的文件名