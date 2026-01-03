# nohup python pipeline.py --restart --debug_mode --dataset assebench --risk_memory lifelong_library/risks_assebench.json --tool_memory lifelong_library/tools_assebench.json --debug_doubt_tool_path debugs/assebench.log --debug_decision_path debugs/assebench_decision.log --restart --debug_file data/ASSEBench/dataset/simulate_assebench.jsonl > logs/assebench/run_assebench.log 2>&1 &

#!/bin/bash
# 文件名: run_assebench_splits.sh
# 批量运行 ASSEBench split 数据集

export CUDA_VISIBLE_DEVICES=1 
cd ..

# 模型配置
SIMULATE_MODEL="deepseek-chat"
TAREVO_MODEL="deepseek-chat"
OPTIM_MODEL="deepseek-chat"
DOUBT_MODEL="deepseek-chat"
SANDBOX_MODEL="deepseek-chat"
SEED=44

# ASSEBench split 文件列表（不含.json扩展名）
# SPLITS=(
#     "Autonomous_Navigation_Robotics"
#     "ContentCreation_Processing_Communication"
#     "Data_Management"
#     "Evaluation_Diagnosis_Support"
#     "Financial_Operations"
#     "Health_Wellness_Support"
#     "Information_Retrieval_Analysis"
#     "IT System_Network_Operations"
#     "Legal_Compliance_ Audit"
#     "Planning_Scheduling_Optimization"
#     "Security_Operations_System_Misuse"
#     "Task_Automation"
# )
SPLITS=(
    "Information_Retrieval_Analysis"
)
echo "========================================"
echo "DEFEND Pipeline ASSEBench Batch Evaluation"
echo "========================================"
echo "Total splits: ${#SPLITS[@]}"
echo "Splits: ${SPLITS[@]}"
echo "========================================"
echo ""

PIDS=()

for split in "${SPLITS[@]}"; do
    echo "========================================="
    echo " Starting ASSEBench / $split"
    echo "========================================="

    dataset="assebench_${split}"
    
    risk_memory="lifelong_library/assebench/risks_${dataset}.json"
    tool_memory="lifelong_library/assebench/safety_tools_${dataset}.json"

    debug_file="data/ASSEBench/split/simulate_${split}.jsonl"
    fail_tool_debug="debugs/assebench/fail_tool_${dataset}.json"
    debug_doubt_tool="debugs/assebench/debug_doubt_tool_${dataset}.log"
    debug_decision="debugs/assebench/debug_decision_${dataset}.log"
    
    LOG_FILE="logs/assebench/${split}.log"
    
    mkdir -p logs/assebench
    mkdir -p lifelong_library/assebench
    mkdir -p debugs/assebench

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

echo "========================================"
echo "All tasks started"
echo "========================================"
echo "Running PIDs: ${PIDS[@]}"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/assebench/*.log"
echo ""
echo "Check running processes:"
echo "  ps -p ${PIDS[@]}"
echo ""
echo "Wait for all tasks to complete:"
echo "  wait ${PIDS[@]}"
echo "========================================"

