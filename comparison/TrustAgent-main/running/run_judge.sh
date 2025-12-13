#!/bin/bash

AGENT_LLM="deepseek-chat"
SIMULATOR_LLM="deepseek-chat"
SAFETY_CHECKER_LLM="deepseek-chat"
AGENT_TEMP=0.0
SIMULATOR_TYPE="adv_thought"
USE_RETRIEVER="openai"

# R-Judge 子文件夹列表
# SUBFOLDERS=("Application" "Finance" "IoT" "Program" "Web")
SUBFOLDERS=("Program")
# 数据类型列表
DATA_TYPES=("harmful" "benign")

echo "========================================"
echo "TrustAgent R-Judge Batch Evaluation"
echo "========================================"

# 创建必要的目录
mkdir -p results/trajectory/rjudge
mkdir -p results/score/rjudge
mkdir -p log/rjudge

total_processed=0
total_errors=0

for subfolder in "${SUBFOLDERS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        # 获取数据文件中的样本数量
        data_file="../../../data/R-Judge/${subfolder}/${data_type}.json"
        
        if [ ! -f "$data_file" ]; then
            echo "Warning: File not found: $data_file"
            continue
        fi

        sample_count=$(python3 -c "import json; print(len(json.load(open('$data_file'))))")
        echo "Processing $subfolder/$data_type: $sample_count samples"
        
        # 遍历所有样本
        for ((case_idx=0; case_idx<sample_count; case_idx++)); do
            echo "Processing sample $((case_idx + 1))/$sample_count..."
            
            # 串行执行，等待当前任务完成
            python main.py \
                --dataset rjudge \
                --subfolder "$subfolder" \
                --data_type "$data_type" \
                --case_idx "$case_idx" \
                --agent_llm_type "$AGENT_LLM" \
                --simulator_llm_type "$SIMULATOR_LLM" \
                --safety_checker_llm_type "$SAFETY_CHECKER_LLM" \
                --agent_temp "$AGENT_TEMP" \
                --simulator_type "$SIMULATOR_TYPE" \
                --use_retriever "$USE_RETRIEVER" \
                --regulation_prompting \
                --regulation_check > "log/rjudge/${data_type}_${subfolder}_${case_idx}.log" 2>&1
            
            # 检查上一个命令的退出状态
            exit_status=$?
            if [ $exit_status -eq 0 ]; then
                echo "✓ Sample $((case_idx + 1))/$sample_count completed successfully"
                ((total_processed++))
            else
                echo "✗ Sample $((case_idx + 1))/$sample_count failed with exit code: $exit_status"
                ((total_errors++))
                echo "Check log file: log/rjudge/${data_type}_${subfolder}_${case_idx}.log for details"
            fi
        done
    done
done

echo "========================================"
echo "Processing completed!"
echo "Total processed: $total_processed"
echo "Total errors: $total_errors"
echo ""
echo "Results saved in:"
echo "  Trajectories: results/trajectory/"
echo "  Scores: results/score/"
echo "========================================"