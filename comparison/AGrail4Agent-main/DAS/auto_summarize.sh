#!/bin/bash

# 自动等待评估完成并汇总结果

echo "等待评估任务完成..."
echo ""

# 检查是否有 evaluate_defend.py 进程在运行
while pgrep -f "evaluate_defend.py" > /dev/null; do
    echo -n "."
    sleep 10
done

echo ""
echo "所有评估任务已完成!"
echo ""

# 等待几秒确保文件写入完成
sleep 5

# 运行汇总脚本
echo "开始汇总结果..."
python summarize_defend_results.py

echo ""
echo "=========================================="
echo "评估和汇总完成!"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - result/defend_comparison/summary_gpt-4o.json"
echo "  - result/defend_comparison/comparison_table_gpt-4o.csv"
echo ""
echo "查看汇总表格:"
echo "  cat result/defend_comparison/comparison_table_gpt-4o.csv"
echo ""
