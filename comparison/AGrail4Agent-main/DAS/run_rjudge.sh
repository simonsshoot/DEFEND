MODEL="deepseek-chat"
SEED=44
# SUBFOLDERS=("Application" "Finance" "IoT" "Program" "Web")
SUBFOLDERS=("Application" "Program")
DATA_TYPES=("harmful")

echo "========================================"
echo "AGrail4Agent R-Judge Batch Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "========================================"
echo "Starting all evaluations in background..."
echo ""

# 启动所有任务
for subfolder in "${SUBFOLDERS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        LOG_FILE="logs/rjudge/${subfolder}_${data_type}.log"
        echo "Starting: $subfolder/$data_type (log: $LOG_FILE)"
        
        nohup python evaluate.py \
            --dataset rjudge \
            --data_type $data_type \
            --subfolder $subfolder \
            --model $MODEL \
            --seed $SEED > "$LOG_FILE" 2>&1 &
        
        # 记录进程ID
        echo $! >> evaluation_pids.txt
        echo "  PID: $!"
    done
done

echo ""
echo "All tasks started in background."
echo "Check logs: evaluation_*.log"
echo "Process IDs saved in: evaluation_pids.txt"
echo ""
echo "To monitor progress:"
echo "  tail -f evaluation_Application_harmful.log"
echo "  ps aux | grep python"
echo ""
echo "To kill all tasks:"
echo "  cat evaluation_pids.txt | xargs kill -9"