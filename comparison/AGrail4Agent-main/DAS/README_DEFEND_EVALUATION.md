# AGrail4Agent 评估实验 - DEFEND 数据集对比

本目录包含使用 AGrail4Agent 框架评估 DEFEND 项目数据集的脚本。

## 数据集

评估以下三个数据集：

1. **AgentHarm**: Agent 安全性评估数据集
   - 路径: `DEFEND/data/agentharm/`
   - 包含 harmful 和 benign 两类数据

2. **ASB (AgentSafeBench)**: Agent 安全基准测试
   - 路径: `DEFEND/data/ASB/`
   - 包含 harmful 和 benign 两类数据

3. **R-Judge**: 基于规则的 Agent 评估数据集
   - 路径: `DEFEND/data/R-Judge/`
   - 包含 harmful 和 benign 两类数据

## 文件说明

### 主要脚本

- **`evaluate_defend.py`**: 评估脚本
  - 支持 AgentHarm、ASB、R-Judge 三个数据集
  - 使用 AGrail4Agent 的 guardrail 机制进行安全检查
  - 输出详细的评估结果

- **`run_defend_evaluation.sh`**: 批量运行脚本
  - 自动运行所有数据集和数据类型的评估
  - 后台运行，输出日志到 `logs/defend_comparison/`

- **`summarize_defend_results.py`**: 结果汇总脚本
  - 汇总所有评估结果
  - 计算准确率、精确率、召回率、F1分数等指标
  - 生成对比表格

## 使用方法

### 1. 单个数据集评估

```bash
# 评估 AgentHarm harmful 数据
python evaluate_defend.py --dataset agentharm --data_type harmful --model gpt-4o

# 评估 ASB benign 数据
python evaluate_defend.py --dataset asb --data_type benign --model gpt-4o

# 评估 R-Judge harmful 数据
python evaluate_defend.py --dataset rjudge --data_type harmful --model gpt-4o

# 调试模式（只处理前3条数据）
python evaluate_defend.py --dataset agentharm --data_type harmful --model gpt-4o --debug

# 重新开始评估（忽略已有结果）
python evaluate_defend.py --dataset agentharm --data_type harmful --model gpt-4o --restart
```

### 2. 批量评估所有数据集

```bash
# 运行评估脚本（推荐）
bash run_defend_evaluation.sh

# 查看运行日志
tail -f logs/defend_comparison/agentharm_harmful.log
tail -f logs/defend_comparison/asb_benign.log
tail -f logs/defend_comparison/rjudge_harmful.log
```

### 3. 汇总评估结果

```bash
# 汇总所有评估结果
python summarize_defend_results.py --model gpt-4o
```

## 输出结果

### 评估结果

每个数据集的评估结果保存在 `result/defend_comparison/` 目录下：

```
result/defend_comparison/
├── agentharm/
│   ├── harmful_gpt-4o.csv
│   └── benign_gpt-4o.csv
├── asb/
│   ├── harmful_gpt-4o.csv
│   └── benign_gpt-4o.csv
├── rjudge/
│   ├── harmful_gpt-4o.csv
│   └── benign_gpt-4o.csv
├── summary_gpt-4o.json
└── comparison_table_gpt-4o.csv
```

### CSV 文件格式

每个 CSV 文件包含以下列：

- `index`: 数据索引
- `user_request`: 用户请求
- `agent_actions`: Agent 操作
- `label`: 真实标签 (0=harmful/should block, 1=benign/should pass)
- `decision`: AGrail 判断结果 (safe/unsafe/error)
- `guardrail_output`: Guardrail 输出
- `check_process`: 检查过程
- `code`: 生成的检查代码
- `token_length`: Token 长度

### 汇总结果

- **`summary_gpt-4o.json`**: JSON 格式的详细汇总结果
- **`comparison_table_gpt-4o.csv`**: CSV 格式的对比表格

## 评估指标

脚本会计算以下指标：

1. **Accuracy (准确率)**: (TP + TN) / Total
2. **Precision (精确率)**: TP / (TP + FP)
3. **Recall (召回率)**: TP / (TP + FN)
4. **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
5. **False Positive Rate (误报率)**: FP / Total
6. **False Negative Rate (漏报率)**: FN / Total

其中：
- TP (True Positive): 正确识别为 unsafe (harmful 数据)
- TN (True Negative): 正确识别为 safe (benign 数据)
- FP (False Positive): 错误识别为 unsafe (benign 误判为 harmful)
- FN (False Negative): 错误识别为 safe (harmful 误判为 benign)

## 注意事项

1. **环境要求**:
   - 需要安装 AGrail4Agent 的所有依赖
   - 需要配置 OpenAI API 密钥（或其他模型 API）
   - 需要 Docker（用于 Container 环境）

2. **数据路径**:
   - 脚本假设数据集位于 `../../../data/` 相对路径下
   - 如果数据集路径不同，请修改 `evaluate_defend.py` 中的路径

3. **运行时间**:
   - 每个数据集可能需要较长时间运行
   - 建议使用后台运行（nohup）
   - 可以使用 `--debug` 参数先测试

4. **错误处理**:
   - 脚本会自动跳过出错的项，并标记为 "error"
   - 可以使用 `--restart` 参数重新运行失败的项

## 对比实验

此评估用于与 DEFEND 框架进行对比：

- **DEFEND**: 使用自进化安全工具的防御框架
- **AGrail4Agent**: 使用 guardrail 机制的防御框架

对比维度：
1. 准确率对比
2. 误报率对比（良性数据被误判为恶意）
3. 漏报率对比（恶意数据被误判为良性）
4. 运行效率对比

## 问题排查

### 数据加载失败

```bash
# 检查数据文件是否存在
ls -l ../../../data/agentharm/harmful.json
ls -l ../../../data/ASB/benign.json
ls -l ../../../data/R-Judge/harmful.json
```

### API 调用失败

```bash
# 检查 API 配置
cat api_config.py

# 测试 API 连接
python test_api.py
```

### 内存不足

```bash
# 减少批处理大小或使用更小的数据集
python evaluate_defend.py --dataset agentharm --data_type harmful --debug
```

## 相关文档

- AGrail4Agent: [原项目文档]
- DEFEND: `../../../README.md`
- 数据集说明: `../../../data/README.md`

## 更新日志

- 2025-12-08: 初始版本，支持 AgentHarm、ASB、R-Judge 数据集
