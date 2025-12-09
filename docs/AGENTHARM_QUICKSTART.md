# AgentHarm 分类数据集快速运行指南

## 🎯 一键运行（推荐）

```bash
cd /path/to/DEFEND
bash script/run_agentharm_categories.sh
```

脚本会自动完成：
1. ✅ 生成所有类别的模拟数据
2. ✅ 运行所有类别的安全评估  
3. ✅ 汇总所有结果并生成报告

## 📋 运行命令总结

### 验证数据集
```bash
bash script/verify_agentharm_categories.sh
```

### 完整运行
```bash
bash script/run_agentharm_categories.sh
```

### 手动汇总（如果需要重新生成汇总）
```bash
python summarize_agentharm_categories.py
```

## 📊 查看结果

### 汇总报告位置
- CSV: `results/agentharm_categories_summary.csv`
- JSON: `results/agentharm_categories_summary.json`

### 单个类别结果
```bash
ls results/agentharm_*/deepseek-chat_deepseek-chat_deepseek-chat.csv
```

## 🔍 监控运行状态

### 查看日志
```bash
# 所有评估日志
tail -f logs/run_agentharm_*.log

# 特定类别
tail -f logs/run_agentharm_Copyright_harmful.log
```

### 检查进程
```bash
ps aux | grep pipeline.py
```

## 🎛️ 数据集类别

1. **Copyright** - 版权
2. **Cybercrime** - 网络犯罪
3. **Disinformation** - 虚假信息
4. **Drugs** - 毒品
5. **Fraud** - 欺诈
6. **Harassment** - 骚扰
7. **Hate** - 仇恨
8. **Sexual** - 性相关

每个类别包含 `harmful` 和 `benign` 两种数据。

## ⏱️ 预计运行时间

- **模拟阶段**: 约 2-4 小时（取决于数据量）
- **评估阶段**: 约 3-6 小时（取决于数据量）
- **汇总阶段**: < 1 分钟

总计: **约 5-10 小时**（16 个并行任务）

## 💡 提示

- 运行脚本会在后台执行所有任务
- 可以关闭终端，任务会继续运行
- 使用 `nohup` 确保任务不会因终端关闭而中断
- 所有输出重定向到 `logs/` 目录
