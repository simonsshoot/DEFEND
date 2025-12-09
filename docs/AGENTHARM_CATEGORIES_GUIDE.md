# AgentHarm 分类数据集运行指南

本文档说明如何运行 AgentHarm 分类数据集的完整评估流程。

## 📁 数据集结构

AgentHarm 数据集已按照 8 个类别分组：

```
data/agentharm/
├── Copyright/
│   ├── harmful.json          # 版权类有害数据
│   └── benign.json           # 版权类良性数据
├── Cybercrime/
│   ├── harmful.json          # 网络犯罪类有害数据
│   └── benign.json           # 网络犯罪类良性数据
├── Disinformation/
│   ├── harmful.json          # 虚假信息类有害数据
│   └── benign.json           # 虚假信息类良性数据
├── Drugs/
│   ├── harmful.json          # 毒品类有害数据
│   └── benign.json           # 毒品类良性数据
├── Fraud/
│   ├── harmful.json          # 欺诈类有害数据
│   └── benign.json           # 欺诈类良性数据
├── Harassment/
│   ├── harmful.json          # 骚扰类有害数据
│   └── benign.json           # 骚扰类良性数据
├── Hate/
│   ├── harmful.json          # 仇恨类有害数据
│   └── benign.json           # 仇恨类良性数据
└── Sexual/
    ├── harmful.json          # 性相关有害数据
    └── benign.json           # 性相关良性数据
```

## 🚀 运行流程

### 步骤 1: 验证数据集（可选）

```bash
bash script/verify_agentharm_categories.sh
```

这将检查所有类别的数据文件是否存在，并显示每个类别的数据数量。

### 步骤 2: 运行完整评估流程

```bash
cd /path/to/DEFEND
bash script/run_agentharm_categories.sh
```

该脚本会执行以下操作：

1. **生成模拟数据**（步骤 1）
   - 为每个类别的 harmful 和 benign 数据生成代理操作模拟
   - 模拟数据保存在各类别文件夹下：
     - `data/agentharm/<category>/harmful_simulate.jsonl`
     - `data/agentharm/<category>/benign_simulate.jsonl`

2. **运行评估**（步骤 2）
   - 使用模拟数据运行安全评估
   - 结果保存在：`results/agentharm_<category>_<type>/`

3. **等待完成并汇总**（步骤 3-4）
   - 等待所有后台任务完成
   - 自动运行汇总脚本生成统计报告

### 步骤 3: 查看结果

评估完成后，会自动生成汇总报告：

- **CSV 格式**: `results/agentharm_categories_summary.csv`
- **JSON 格式**: `results/agentharm_categories_summary.json`
- **控制台输出**: 详细的统计信息

## 📊 输出说明

### 单个类别结果

每个类别会生成以下结果文件：

```
results/
├── agentharm_Copyright_harmful/
│   └── deepseek-chat_deepseek-chat_deepseek-chat.csv
├── agentharm_Copyright_benign/
│   └── deepseek-chat_deepseek-chat_deepseek-chat.csv
├── agentharm_Cybercrime_harmful/
│   └── deepseek-chat_deepseek-chat_deepseek-chat.csv
...
```

### 汇总统计

汇总脚本会生成：

1. **按类别统计**
   - 每个类别的 harmful 和 benign 数据的准确率
   - Safe/Unsafe/Error 数量分布
   - Precision, Recall, F1 分数（针对 harmful 数据）

2. **整体统计**
   - 所有 harmful 数据的总体表现
   - 所有 benign 数据的总体表现
   - 全部数据的综合准确率

## 📝 日志文件

运行过程中的日志保存在 `logs/` 目录：

- 模拟阶段: `logs/simulate_agentharm_<category>_<type>.log`
- 评估阶段: `logs/run_agentharm_<category>_<type>.log`

## 🔧 手动运行单个类别

如果需要单独运行某个类别，可以使用以下命令：

### 生成模拟数据

```bash
python pipeline.py \
    --restart \
    --debug_mode \
    --need_simulate \
    --dataset "agentharm_Copyright_harmful" \
    --risk_memory "lifelong_library/risks_agentharm_Copyright_harmful.json" \
    --tool_memory "lifelong_library/tools_agentharm_Copyright_harmful.json" \
    --debug_file "data/agentharm/Copyright/harmful_simulate.jsonl"
```

### 运行评估（使用模拟数据）

```bash
python pipeline.py \
    --restart \
    --debug_mode \
    --dataset "agentharm_Copyright_harmful" \
    --risk_memory "lifelong_library/risks_agentharm_Copyright_harmful.json" \
    --tool_memory "lifelong_library/tools_agentharm_Copyright_harmful.json"
```

### 单独运行汇总

```bash
python summarize_agentharm_categories.py
```

## 📈 监控进度

### 查看实时日志

```bash
# 查看所有评估日志
tail -f logs/run_agentharm_*.log

# 查看特定类别
tail -f logs/run_agentharm_Copyright_harmful.log
```

### 检查运行状态

```bash
# 查看正在运行的 pipeline 进程
ps aux | grep pipeline.py

# 使用 watch 实时监控
watch -n 5 'ps aux | grep pipeline.py | wc -l'
```

## ⚠️ 注意事项

1. **资源需求**: 16 个并行任务（8 类别 × 2 类型）会占用大量资源
2. **运行时间**: 完整运行可能需要数小时，取决于数据量和硬件性能
3. **磁盘空间**: 确保有足够空间存储模拟数据和结果
4. **步骤顺序**: 必须先完成步骤 1（模拟）再运行步骤 2（评估）

## 🐛 故障排查

### 模拟数据未生成

检查日志文件：
```bash
cat logs/simulate_agentharm_<category>_<type>.log
```

### 评估失败

确认模拟数据存在：
```bash
ls -lh data/agentharm/*/harmful_simulate.jsonl
ls -lh data/agentharm/*/benign_simulate.jsonl
```

### 汇总结果为空

确认所有评估已完成：
```bash
ls -lh results/agentharm_*/deepseek-chat_deepseek-chat_deepseek-chat.csv
```

## 📞 支持

如有问题，请查看：
- 日志文件: `logs/`
- 调试输出: `debugs/`
- 失败工具记录: `results/fail_tool_debug2.json`
