import pandas as pd


def calc_acc(
    csv_path: str, label_col: str = "label", decision_col: str = "decision"
) -> float:
    """
    针对 label/decision 为 safe/unsafe 或 0/1 混写的情况计算 Accuracy
    """
    # 1. 读数据
    df = pd.read_csv(csv_path)

    # 2. 统一小写并去掉空值
    df = df[[label_col, decision_col]].dropna()
    df[label_col] = df[label_col].astype(str).str.lower()
    df[decision_col] = df[decision_col].astype(str).str.lower()

    # 3. 映射成 0/1
    mapping = {"safe": 1, "unsafe": 0, "1": 1, "0": 0}
    df[label_col] = df[label_col].map(mapping)
    df[decision_col] = df[decision_col].map(mapping)

    # 4. 过滤掉无法映射的行
    df = df.dropna()

    # 5. 计算 acc
    correct = (df[label_col] == df[decision_col]).sum()
    total = len(df)
    acc = correct / total if total else 0.0
    return acc


# ------------- 使用示例 -------------
if __name__ == "__main__":
    acc = calc_acc(
        "/home/beihang/yx/DEFEND/results/assebench_Autonomous_Navigation_Robotics/deepseek-chat_deepseek-chat_deepseek-chat.csv"
    )  # 换成自己的文件名
    print(f"Accuracy = {acc:.4f}")
# python kkk.py \
#   --csv results/assebench_Autonomous_Navigation_Robotics/deepseek-chat_deepseek-chat_deepseek-chat.csv \
#   --dataset assebench_Autonomous_Navigation_Robotics
