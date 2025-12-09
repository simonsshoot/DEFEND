export CUDA_VISIBLE_DEVICES=0,1,2
MODEL="deepseek-chat"
DATA_TYPE="harmful"

# 定义要处理的类别列表
CATEGORIES=("Fraud" "Harassment" "Hate" "Sexual")

mkdir -p logs/agentharm
mkdir -p result/comparison/agentharm

echo "开始并行执行以下类别的评估: ${CATEGORIES[@]}"
echo "======================================"

PIDS=()  # 存储进程ID
LOG_FILES=()  # 存储日志文件路径

# 为每个类别启动一个后台任务
for CATEGORY in "${CATEGORIES[@]}"; do
    # 创建类别特定的结果目录
    mkdir -p "result/comparison/agentharm/$CATEGORY"
    
    LOG_FILE="logs/agentharm/agrail_${CATEGORY}_${DATA_TYPE}.log"
    LOG_FILES+=("$LOG_FILE")
    
    echo "启动类别: $CATEGORY"
    echo "日志文件: $LOG_FILE"
    echo "结果文件: result/comparison/agentharm/${CATEGORY}/${DATA_TYPE}_${MODEL}.csv"
    
    # 启动评估任务
    nohup python evaluate.py \
        --dataset agentharm \
        --data_type $DATA_TYPE \
        --category $CATEGORY \
        --model $MODEL \
        --seed 44 \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=("$PID")
    echo "进程ID: $PID"
    echo "--------------------------------------"
done

echo "======================================"
echo "总进程数: ${#PIDS[@]}"
echo ""
echo "查看所有进程状态:"
for i in "${!PIDS[@]}"; do
    echo "  ${CATEGORIES[$i]}: PID=${PIDS[$i]}"
done
echo ""
echo "查看实时日志:"
for i in "${!LOG_FILES[@]}"; do
    echo "  tail -f ${LOG_FILES[$i]}  # ${CATEGORIES[$i]}"
done
echo ""
echo "使用以下命令查看进程状态:"
echo "  ps aux | grep -E \"$(printf '%s|' "${PIDS[@]}")\""
echo ""
echo "等待所有任务完成:"
echo "  for pid in ${PIDS[@]}; do wait \$pid; done"