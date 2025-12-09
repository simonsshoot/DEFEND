export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL="deepseek-chat"
# AgentHarm Harmful
echo "  - Running AgentHarm benign..."
nohup python evaluate.py \
    --dataset agentharm \
    --data_type benign \
    --model deepseek-chat \
    --seed 44 \
    > logs/agrail_agentharm_benign.log 2>&1 &

nohup python evaluate.py --dataset agentharm --data_type benign -model deepseek-chat --seed 44 > logs/agrail_agentharm_benign.log 2>&1 &