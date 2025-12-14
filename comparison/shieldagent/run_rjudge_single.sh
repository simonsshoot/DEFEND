#!/bin/bash

# ShieldAgent R-Judge 单个子文件夹评估脚本
# 用法: ./run_rjudge_single.sh <subfolder> <type>
# 示例: ./run_rjudge_single.sh Application harmful
#       ./run_rjudge_single.sh Finance benign

if [ $# -lt 2 ]; then
    echo "Usage: $0 <subfolder> <type>"
    echo "  subfolder: Application, Finance, IoT, Program, Web"
    echo "  type: harmful or benign"
    echo ""
    echo "Example: $0 Application harmful"
    exit 1
fi

SUBFOLDER=$1
TYPE=$2

MODEL_PATH="/home/beihang/yx/models/shieldagent"
OUTPUT_DIR="results"

# 验证 subfolder
if [[ ! "$SUBFOLDER" =~ ^(Application|Finance|IoT|Program|Web)$ ]]; then
    echo "Error: Invalid subfolder '$SUBFOLDER'"
    echo "Valid subfolders: Application, Finance, IoT, Program, Web"
    exit 1
fi

# 验证 type
if [[ ! "$TYPE" =~ ^(harmful|benign)$ ]]; then
    echo "Error: Invalid type '$TYPE'"
    echo "Valid types: harmful, benign"
    exit 1
fi

DATASET="rjudge_$TYPE"

echo "=========================================="
echo "ShieldAgent R-Judge Evaluation"
echo "Subfolder: $SUBFOLDER"
echo "Type: $TYPE"
echo "=========================================="

python evaluate.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --subfolder "$SUBFOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --simulate_data False

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/rjudge/$SUBFOLDER/${DATASET}_results.csv"
echo "=========================================="
