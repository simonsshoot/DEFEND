#!/bin/bash

# AGrail4Agent DEFEND 评估 - 完整运行流程
# 此脚本包含从测试到评估到汇总的完整流程

echo "=========================================="
echo "AGrail4Agent DEFEND 评估 - 完整流程"
echo "=========================================="
echo ""

# 步骤 1: 测试设置
echo "步骤 1/3: 测试环境设置..."
python test_defend_setup.py
if [ $? -ne 0 ]; then
    echo "错误: 环境测试失败，请检查上述错误信息"
    exit 1
fi
echo ""

# 步骤 2: 运行评估
echo "步骤 2/3: 开始评估..."
bash run_defend_evaluation.sh
echo ""

# 等待提示
echo "=========================================="
echo "评估任务已提交到后台运行"
echo "=========================================="
echo ""
echo "监控进度:"
echo "  1. 查看日志: tail -f logs/defend_comparison/agentharm_harmful.log"
echo "  2. 检查进程: ps aux | grep evaluate_defend"
echo "  3. 查看结果: ls -lh result/defend_comparison/*/"
echo ""
echo "等待所有评估完成后，运行以下命令汇总结果:"
echo "  python summarize_defend_results.py"
echo ""
echo "或者运行以下命令自动等待并汇总:"
echo "  bash auto_summarize.sh"
echo ""
