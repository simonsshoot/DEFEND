# AgentHarm 分类数据集批量运行脚本
# 每个类别包含 harmful 和 benign 两类数据

# AgentHarm 的 8 个类别
cd ..
categories=("Copyright" "Cybercrime" "Disinformation" "Drugs" "Fraud" "Harassment" "Hate" "Sexual")

for category in "${categories[@]}"; do
    echo "处理类别: $category"
    
    # Harmful 数据模拟
    echo "  - 生成 ${category} harmful 模拟数据..."
    nohup python pipeline.py \
        --restart \
        --debug_mode \
        --need_simulate \
        --dataset "agentharm_${category}_harmful" \
        --risk_memory "lifelong_library/agentharm/risk_${category}_harmful.json" \
        --tool_memory "lifelong_library/agentharm/tools_${category}_harmful.json" \
        --debug_file "data/agentharm/${category}/harmful_simulate.jsonl" \
        --debug_doubt_tool_path "debugs/agentharm/${category}_harmful.log" \
        --debug_decision_path "debugs/agentharm/${category}_harmful_decision.log" \
        > "logs/agentharm/${category}_harmful.log" 2>&1 &
    
    # Benign 数据模拟
    echo "  - 生成 ${category} benign 模拟数据..."
    nohup python pipeline.py \
        --restart \
        --debug_mode \
        --need_simulate \
        --dataset "agentharm_${category}_benign" \
        --risk_memory "lifelong_library/agentharm/risk_${category}_benign.json" \
        --tool_memory "lifelong_library/agentharm/tools_${category}_benign.json" \
        --debug_file "data/agentharm/${category}/benign_simulate.jsonl" \
        --debug_doubt_tool_path "debugs/agentharm/${category}_benign.log" \
        --debug_decision_path "debugs/agentharm/${category}_benign_decision.log" \
        > "logs/agentharm/${category}_benign.log" 2>&1 &

    echo ""
done

