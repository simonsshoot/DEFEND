export CUDA_VISIBLE_DEVICES=0,1,2
MODEL="deepseek-chat"

CATEGORY="Drugs"  # Copyright, Cybercrime, Disinformation, Drugs, Fraud, Harassment, Hate, Sexual
DATA_TYPE="harmful"    # harmful 或 benign

mkdir -p logs/agentharm
mkdir -p result/comparison/agentharm/$CATEGORY

nohup python evaluate.py \
    --dataset agentharm \
    --data_type $DATA_TYPE \
    --category $CATEGORY \
    --model $MODEL \
    --seed 44 \
    > logs/agentharm/agrail_${CATEGORY}_${DATA_TYPE}.log 2>&1 &

PID=$!
echo ""
echo "任务已启动 (PID: $PID)"
echo "日志文件: logs/agentharm/agrail_${CATEGORY}_${DATA_TYPE}.log"
echo "结果文件: result/comparison/agentharm/${CATEGORY}/${DATA_TYPE}_${MODEL}.csv"
echo ""
echo "查看日志: tail -f logs/agentharm/agrail_${CATEGORY}_${DATA_TYPE}.log"
echo "查看进程: ps aux | grep $PID"
