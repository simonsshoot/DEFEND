# 创建必要的目录
mkdir -p result/comparison/agentharm
mkdir -p result/comparison/asb
mkdir -p result/comparison/rjudge
mkdir -p memory
mkdir -p logs

# 模型选择
MODEL="deepseek-chat"


# ==================== AgentHarm 数据集 ====================
echo ""
echo "Evaluating AgentHarm dataset..."
export CUDA_VISIBLE_DEVICES=4,5,6,7
# AgentHarm Harmful
echo "  - Running AgentHarm harmful..."
nohup python evaluate.py \
    --dataset agentharm \
    --data_type harmful \
    --model deepseek-chat \
    --seed 44 \
    > logs/agrail_agentharm_harmful.log 2>&1 &

# AgentHarm Benign
echo "  - Running AgentHarm benign..."
nohup python evaluate.py \
    --dataset agentharm \
    --data_type benign \
    --model $MODEL \
    --seed 44 \
    > logs/agrail_agentharm_benign.log 2>&1 &

# ==================== ASB 数据集 ====================
echo ""
echo "Evaluating ASB dataset..."

# ASB Harmful
echo "  - Running ASB harmful..."
nohup python evaluate.py \
    --dataset asb \
    --data_type harmful \
    --model $MODEL \
    --seed 44 \
    > logs/agrail_asb_harmful.log 2>&1 &

# ASB Benign
echo "  - Running ASB benign..."
nohup python evaluate.py \
    --dataset asb \
    --data_type benign \
    --model $MODEL \
    --seed 44 \
    > logs/agrail_asb_benign.log 2>&1 &

# ==================== R-Judge 数据集 ====================
echo ""
echo "Evaluating R-Judge dataset..."

# R-Judge Harmful
echo "  - Running R-Judge harmful..."
nohup python evaluate.py \
    --dataset rjudge \
    --data_type harmful \
    --model $MODEL \
    --seed 44 \
    > logs/agrail_rjudge_harmful.log 2>&1 &

# R-Judge Benign
echo "  - Running R-Judge benign..."
nohup python evaluate.py \
    --dataset rjudge \
    --data_type benign \
    --model $MODEL \
    --seed 44 \
    > logs/agrail_rjudge_benign.log 2>&1 &

# echo "All jobs completed! Running summary..."
# python summarize_results.py --model $MODEL
