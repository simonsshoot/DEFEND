# AGrail4Agent DEFEND 评估快速参考

## 测试设置
```bash
cd comparison/AGrail4Agent-main/DAS
python test_defend_setup.py
```

## 运行评估

### 方式1: 批量运行所有数据集（推荐）
```bash
bash run_defend_evaluation.sh
```

### 方式2: 单独运行每个数据集

#### AgentHarm
```bash
# Harmful
python evaluate_defend.py --dataset agentharm --data_type harmful --model gpt-4o

# Benign  
python evaluate_defend.py --dataset agentharm --data_type benign --model gpt-4o
```

#### ASB (AgentSafeBench)
```bash
# Harmful
python evaluate_defend.py --dataset asb --data_type harmful --model gpt-4o

# Benign
python evaluate_defend.py --dataset asb --data_type benign --model gpt-4o
```

#### R-Judge
```bash
# Harmful
python evaluate_defend.py --dataset rjudge --data_type harmful --model gpt-4o

# Benign
python evaluate_defend.py --dataset rjudge --data_type benign --model gpt-4o
```

### 方式3: 调试模式（只处理3条数据）
```bash
python evaluate_defend.py --dataset agentharm --data_type harmful --model gpt-4o --debug
```

## 查看运行日志
```bash
# 实时查看日志
tail -f logs/defend_comparison/agentharm_harmful.log
tail -f logs/defend_comparison/asb_benign.log
tail -f logs/defend_comparison/rjudge_harmful.log

# 查看所有日志文件
ls -lh logs/defend_comparison/
```

## 汇总结果
```bash
# 生成汇总报告
python summarize_defend_results.py --model gpt-4o

# 查看汇总结果
cat result/defend_comparison/summary_gpt-4o.json
cat result/defend_comparison/comparison_table_gpt-4o.csv
```

## 查看结果文件
```bash
# 列出所有结果文件
ls -lh result/defend_comparison/*/

# 查看某个结果文件的前几行
head -20 result/defend_comparison/agentharm/harmful_gpt-4o.csv
```

## 监控进度
```bash
# 查看进程
ps aux | grep evaluate_defend

# 查看GPU使用情况（如果使用GPU）
nvidia-smi

# 查看结果文件大小变化
watch -n 5 'ls -lh result/defend_comparison/*/*.csv'
```

## 重新运行（忽略已有结果）
```bash
python evaluate_defend.py --dataset agentharm --data_type harmful --model gpt-4o --restart
```

## 清理结果
```bash
# 删除所有结果（谨慎使用）
rm -rf result/defend_comparison/*
rm -rf logs/defend_comparison/*
rm -rf memory/memory_*_gpt-4o.json
```

## 常见问题

### 1. 数据文件找不到
```bash
# 检查数据文件路径
ls -l ../../../data/agentharm/harmful.json
ls -l ../../../data/ASB/benign.json
ls -l ../../../data/R-Judge/harmful.json
```

### 2. API 调用失败
```bash
# 检查 API 配置
cat api_config.py

# 测试 API
python test_api.py
```

### 3. 内存不足
```bash
# 使用调试模式减少数据量
python evaluate_defend.py --dataset agentharm --data_type harmful --debug

# 或者逐个运行数据集，而不是并行运行
```

### 4. 查看某个数据集的评估进度
```bash
# 统计已完成的行数（假设每处理一条数据就写入CSV）
wc -l result/defend_comparison/agentharm/harmful_gpt-4o.csv
```

## 与 DEFEND 框架对比

### DEFEND 运行命令（参考）
```bash
cd ../../..  # 返回 DEFEND 根目录
python pipeline.py --dataset agentharm --data_type harmful
```

### AGrail 运行命令
```bash
cd comparison/AGrail4Agent-main/DAS
python evaluate_defend.py --dataset agentharm --data_type harmful
```

### 对比结果
```bash
# DEFEND 结果路径
cat results/agentharm/deepseek-chat_deepseek-chat_deepseek-chat.csv

# AGrail 结果路径
cat result/defend_comparison/agentharm/harmful_gpt-4o.csv

# 运行对比脚本（如果有）
python compare_with_defend.py
```

## 完整工作流程

```bash
# 1. 测试设置
python test_defend_setup.py

# 2. 运行评估
bash run_defend_evaluation.sh

# 3. 等待运行完成（监控日志）
tail -f logs/defend_comparison/agentharm_harmful.log

# 4. 汇总结果
python summarize_defend_results.py --model gpt-4o

# 5. 查看结果
cat result/defend_comparison/comparison_table_gpt-4o.csv
```
