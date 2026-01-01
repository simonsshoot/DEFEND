#!/bin/bash
# 文件名: evaluate_rjudge_parallel.sh

# 评估所有 R-Judge 子文件夹（Application, Finance, IoT, Program, Web）的 harmful 和 benign 数据
export CUDA_VISIBLE_DEVICES=1 
cd ..
# 模型配置
SIMULATE_MODEL="deepseek-chat"
TAREVO_MODEL="deepseek-chat"
OPTIM_MODEL="deepseek-chat"
DOUBT_MODEL="deepseek-chat"
SANDBOX_MODEL="deepseek-chat"
SEED=44

# R-Judge 子文件夹列表
# SUBFOLDERS=("Application" "Finance" "IoT" "Program" "Web")
SUBFOLDERS=("Program")
# 数据类型列表
DATA_TYPES=("benign")

echo "========================================"
echo "DEFEND Pipeline R-Judge Batch Evaluation"
echo "========================================"
echo "Subfolders: ${SUBFOLDERS[@]}"
echo "Data Types: ${DATA_TYPES[@]}"
echo "========================================"
echo ""

# 存储所有后台进程的PID
PIDS=()

for subfolder in "${SUBFOLDERS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        echo "========================================="
        echo " Starting R-Judge / $subfolder / $data_type"
        echo "========================================="

        dataset="rjudge_${subfolder}_${data_type}"
        
        risk_memory="lifelong_library/rjudge/risks_rjudge_${subfolder}.json"
        tool_memory="lifelong_library/rjudge/safety_tools_rjudge_${subfolder}.json"

        debug_file="data/R-Judge/${subfolder}/simulate_${data_type}.jsonl"
        fail_tool_debug="debugs/rjudge/fail_tool_rjudge_${subfolder}_${data_type}.json"
        debug_doubt_tool="debugs/rjudge/debug_doubt_tool_rjudge_${subfolder}_${data_type}.log"
        debug_decision="debugs/rjudge/debug_decision_rjudge_${subfolder}_${data_type}.log"
        
        LOG_FILE="logs/rjudge/${subfolder}_${data_type}.log"
        
        # 确保logs目录存在
        mkdir -p logs


        nohup python pipeline.py \
            --dataset "$dataset" \
            --risk_memory "$risk_memory" \
            --tool_memory "$tool_memory" \
            --simulate_model "$SIMULATE_MODEL" \
            --tarevo_model "$TAREVO_MODEL" \
            --optim_model "$OPTIM_MODEL" \
            --doubt_model "$DOUBT_MODEL" \
            --sandbox_model "$SANDBOX_MODEL" \
            --seed "$SEED" \
            --restart \
            --debug_file "$debug_file" \
            --fail_tool_debug "$fail_tool_debug" \
            --debug_doubt_tool_path "$debug_doubt_tool" \
            --debug_decision_path "$debug_decision" > "$LOG_FILE" 2>&1 &
        
        # 获取进程ID
        PID=$!
        PIDS+=("$PID")
        
        echo "✓ Task started with PID: $PID"
        echo "  Log: $LOG_FILE"
        echo "  Dataset: $dataset"
        echo ""
        
    done
done
