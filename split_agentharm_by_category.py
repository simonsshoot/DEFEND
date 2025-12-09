"""
将 AgentHarm 数据集按类别分组
为每个类别创建独立的文件夹，并存储 harmful.json 和 benign.json
"""
import json
import os
from collections import defaultdict

# 定义路径
harmful_path = "agentharm/harmful_behaviors_test_public.json"
benign_path = "agentharm/benign_behaviors_test_public.json"
output_base = "data/agentharm"

# 创建输出基础目录
os.makedirs(output_base, exist_ok=True)

# 读取 harmful 数据
print("Reading harmful behaviors...")
with open(harmful_path, "r", encoding="utf-8") as f:
    harmful_data = json.load(f)

# 读取 benign 数据
print("Reading benign behaviors...")
with open(benign_path, "r", encoding="utf-8") as f:
    benign_data = json.load(f)

# 按类别分组 harmful 数据
harmful_by_category = defaultdict(list)
for behavior in harmful_data["behaviors"]:
    category = behavior.get("category", "Unknown")
    if category:
        harmful_by_category[category].append(behavior)

# 按类别分组 benign 数据
benign_by_category = defaultdict(list)
for behavior in benign_data["behaviors"]:
    category = behavior.get("category", "Unknown")
    if category:
        benign_by_category[category].append(behavior)

# 获取所有类别
all_categories = set(harmful_by_category.keys()) | set(benign_by_category.keys())

print(f"\nFound {len(all_categories)} categories:")
for cat in sorted(all_categories):
    print(f"  - {cat}")

# 为每个类别创建文件夹并保存数据
for category in sorted(all_categories):
    # 创建类别文件夹
    category_folder = os.path.join(output_base, category)
    os.makedirs(category_folder, exist_ok=True)
    
    # 保存 harmful.json
    harmful_behaviors = harmful_by_category.get(category, [])
    harmful_output = {
        "canary_guid": harmful_data.get("canary_guid", ""),
        "behaviors": harmful_behaviors
    }
    harmful_file = os.path.join(category_folder, "harmful.json")
    with open(harmful_file, "w", encoding="utf-8") as f:
        json.dump(harmful_output, f, indent=4, ensure_ascii=False)
    print(f"\n✓ Saved {len(harmful_behaviors)} harmful behaviors to {harmful_file}")
    
    # 保存 benign.json
    benign_behaviors = benign_by_category.get(category, [])
    benign_output = {
        "canary_guid": benign_data.get("canary_guid", ""),
        "behaviors": benign_behaviors
    }
    benign_file = os.path.join(category_folder, "benign.json")
    with open(benign_file, "w", encoding="utf-8") as f:
        json.dump(benign_output, f, indent=4, ensure_ascii=False)
    print(f"✓ Saved {len(benign_behaviors)} benign behaviors to {benign_file}")

print("\n" + "="*60)
print("Summary:")
print("="*60)
for category in sorted(all_categories):
    harmful_count = len(harmful_by_category.get(category, []))
    benign_count = len(benign_by_category.get(category, []))
    print(f"{category:20s} - Harmful: {harmful_count:3d}, Benign: {benign_count:3d}")

print("\nAll categories have been successfully split!")
