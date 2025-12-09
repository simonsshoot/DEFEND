#!/bin/bash

# 验证 AgentHarm 分类数据集是否正确创建

echo "=============================================="
echo "验证 AgentHarm 分类数据集"
echo "=============================================="
echo ""

categories=("Copyright" "Cybercrime" "Disinformation" "Drugs" "Fraud" "Harassment" "Hate" "Sexual")

for category in "${categories[@]}"; do
    echo "检查类别: $category"
    
    harmful_file="data/agentharm/${category}/harmful.json"
    benign_file="data/agentharm/${category}/benign.json"
    
    if [ -f "$harmful_file" ]; then
        harmful_count=$(python -c "import json; print(len(json.load(open('$harmful_file'))['behaviors']))")
        echo "  ✓ Harmful: $harmful_count 条数据"
    else
        echo "  ✗ Harmful 文件不存在: $harmful_file"
    fi
    
    if [ -f "$benign_file" ]; then
        benign_count=$(python -c "import json; print(len(json.load(open('$benign_file'))['behaviors']))")
        echo "  ✓ Benign: $benign_count 条数据"
    else
        echo "  ✗ Benign 文件不存在: $benign_file"
    fi
    
    echo ""
done

echo "=============================================="
echo "验证完成"
echo "=============================================="
