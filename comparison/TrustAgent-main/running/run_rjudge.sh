AGENT_LLM="deepseek-chat"
SIMULATOR_LLM="deepseek-chat"
SAFETY_CHECKER_LLM="deepseek-chat"
AGENT_TEMP=0.0
SIMULATOR_TYPE="adv_thought"
USE_RETRIEVER="openai"

# R-Judge 子文件夹列表
# SUBFOLDERS=("Application" "Finance" "IoT" "Program" "Web")
SUBFOLDERS=("Application")
# 数据类型列表
DATA_TYPES=("harmful" "benign")


echo "========================================"
echo "TrustAgent R-Judge Batch Evaluation"
echo "========================================"


mkdir -p results/trajectory/rjudge
mkdir -p results/score/rjudge

for subfolder in "${SUBFOLDERS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        # 获取数据文件中的样本数量
        data_file="../../../data/R-Judge/${subfolder}/${data_type}.json"
        
        if [ ! -f "$data_file" ]; then
            echo "Warning: File not found: $data_file"
            continue
        fi

        sample_count=$(python3 -c "import json; print(len(json.load(open('$data_file'))))")
        
        
        # 遍历所有样本
        for ((case_idx=0; case_idx<sample_count; case_idx++)); do
            echo "Processing sample $((case_idx + 1))/$sample_count..."
            
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
                --regulation_check
            
            if [ $? -eq 0 ]; then
                echo "✓ Successfully processed sample $case_idx"
            else
                echo "✗ Failed to process sample $case_idx"
            fi
        done
        
        echo ""
    done
done

echo "========================================"
echo "Results saved in:"
echo "  Trajectories: results/trajectory/"
echo "  Scores: results/score/"
echo "========================================"

