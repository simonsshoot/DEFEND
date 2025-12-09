#!/bin/bash

# AGrail4Agent 评估脚本 - 对比 DEFEND 数据集
# 数据集: AgentHarm, ASB, R-Judge

# 创建日志目录
mkdir -p logs/defend_comparison

echo "开始 AGrail4Agent 评估实验..."
echo "数据集: AgentHarm, ASB, R-Judge"
echo "模型: gpt-4o (deepseek-chat)"
echo ""

# ============================= AgentHarm 数据集 =============================
echo "=========================================="
echo "评估 AgentHarm 数据集"
echo "=========================================="

# AgentHarm Harmful
echo "  - AgentHarm Harmful..."
nohup python evaluate_defend.py \
    --dataset agentharm \
    --data_type harmful \
    --model gpt-4o \
    > logs/defend_comparison/agentharm_harmful.log 2>&1 &

# AgentHarm Benign
echo "  - AgentHarm Benign..."
nohup python evaluate_defend.py \
    --dataset agentharm \
    --data_type benign \
    --model gpt-4o \
    > logs/defend_comparison/agentharm_benign.log 2>&1 &

echo ""

# ============================= ASB 数据集 =============================
echo "=========================================="
echo "评估 AgentSafeBench (ASB) 数据集"
echo "=========================================="

# ASB Harmful
echo "  - ASB Harmful..."
nohup python evaluate_defend.py \
    --dataset asb \
    --data_type harmful \
    --model gpt-4o \
    > logs/defend_comparison/asb_harmful.log 2>&1 &

# ASB Benign
echo "  - ASB Benign..."
nohup python evaluate_defend.py \
    --dataset asb \
    --data_type benign \
    --model gpt-4o \
    > logs/defend_comparison/asb_benign.log 2>&1 &

echo ""

# ============================= R-Judge 数据集 =============================
echo "=========================================="
echo "评估 R-Judge 数据集"
echo "=========================================="

# R-Judge Harmful
echo "  - R-Judge Harmful..."
nohup python evaluate_defend.py \
    --dataset rjudge \
    --data_type harmful \
    --model gpt-4o \
    > logs/defend_comparison/rjudge_harmful.log 2>&1 &

# R-Judge Benign
echo "  - R-Judge Benign..."
nohup python evaluate_defend.py \
    --dataset rjudge \
    --data_type benign \
    --model gpt-4o \
    > logs/defend_comparison/rjudge_benign.log 2>&1 &

echo ""
echo "=========================================="
echo "所有评估任务已提交到后台运行"
echo "=========================================="
echo ""
echo "查看日志:"
echo "  tail -f logs/defend_comparison/agentharm_harmful.log"
echo "  tail -f logs/defend_comparison/agentharm_benign.log"
echo "  tail -f logs/defend_comparison/asb_harmful.log"
echo "  tail -f logs/defend_comparison/asb_benign.log"
echo "  tail -f logs/defend_comparison/rjudge_harmful.log"
echo "  tail -f logs/defend_comparison/rjudge_benign.log"
echo ""
echo "结果将保存在: result/defend_comparison/"
echo ""
