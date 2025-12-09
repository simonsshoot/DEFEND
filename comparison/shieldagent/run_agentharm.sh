# ShieldAgent - AgentHarm 分类数据集评估命令集合
# 每个类别的 harmful 和 benign 数据评估

export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_PATH="/home/beihang/yx/models/shieldagent"

# ==================== Copyright ====================
echo "Copyright - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Copyright \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Copyright_harmful.log 2>&1 &

echo "Copyright - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Copyright \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Copyright_benign.log 2>&1 &


# ==================== Cybercrime ====================
echo "Cybercrime - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Cybercrime \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Cybercrime_harmful.log 2>&1 &

echo "Cybercrime - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Cybercrime \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Cybercrime_benign.log 2>&1 &


# ==================== Disinformation ====================
echo "Disinformation - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Disinformation \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Disinformation_harmful.log 2>&1 &

echo "Disinformation - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Disinformation \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Disinformation_benign.log 2>&1 &


# ==================== Drugs ====================
echo "Drugs - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Drugs \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Drugs_harmful.log 2>&1 &

echo "Drugs - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Drugs \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Drugs_benign.log 2>&1 &


# ==================== Fraud ====================
echo "Fraud - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Fraud \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Fraud_harmful.log 2>&1 &

echo "Fraud - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Fraud \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Fraud_benign.log 2>&1 &


# ==================== Harassment ====================
echo "Harassment - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Harassment \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Harassment_harmful.log 2>&1 &

echo "Harassment - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Harassment \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Harassment_benign.log 2>&1 &


# ==================== Hate ====================
echo "Hate - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Hate \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Hate_harmful.log 2>&1 &

echo "Hate - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Hate \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Hate_benign.log 2>&1 &


# ==================== Sexual ====================
echo "Sexual - Harmful"
nohup python evaluate.py \
    --dataset agentharm \
    --category Sexual \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Sexual_harmful.log 2>&1 &

echo "Sexual - Benign"
nohup python evaluate.py \
    --dataset agentharm_benign \
    --category Sexual \
    --model_path $MODEL_PATH \
    --output_dir results \
    > logs/agentharm/shield_Sexual_benign.log 2>&1 &


echo ""
echo "所有任务已启动！"
echo "查看运行状态: ps aux | grep 'python evaluate.py'"
echo "查看日志: tail -f logs/shieldagent/agentharm/shield_Copyright_harmful.log"
