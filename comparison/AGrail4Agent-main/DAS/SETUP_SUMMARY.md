# AGrail4Agent 评估实验总结

## 已创建的文件

### 1. 核心评估脚本
- **`evaluate_defend.py`**: 主评估脚本
  - 支持 AgentHarm、ASB、R-Judge 数据集
  - 自动数据加载和预处理
  - 使用 AGrail guardrail 进行安全检查
  - 输出详细CSV结果

### 2. 运行脚本
- **`run_defend_evaluation.sh`**: 批量运行脚本
  - 并行运行所有 6 个评估任务（3个数据集 × 2种类型）
  - 后台运行，输出日志到 `logs/defend_comparison/`
  - 适合服务器环境

### 3. 汇总脚本
- **`summarize_defend_results.py`**: 结果汇总脚本
  - 计算准确率、精确率、召回率、F1分数
  - 生成混淆矩阵
  - 输出JSON和CSV格式的汇总报告

### 4. 测试脚本
- **`test_defend_setup.py`**: 环境测试脚本
  - 检查数据文件是否存在
  - 验证模块导入
  - 测试数据加载功能
  - 创建必要目录

### 5. 文档
- **`README_DEFEND_EVALUATION.md`**: 详细使用文档
- **`QUICK_REFERENCE.md`**: 快速参考命令

## 数据集支持

### AgentHarm
- 路径: `DEFEND/data/agentharm/harmful.json`, `benign.json`
- 特点: Agent 恶意行为检测
- 标签: harmful=0 (unsafe), benign=1 (safe)

### ASB (AgentSafeBench)
- 路径: `DEFEND/data/ASB/harmful.json`, `benign.json`
- 特点: 多环境 Agent 安全测试
- 标签: fulfillable=0 (should block), fulfillable=1 (should pass)

### R-Judge
- 路径: `DEFEND/data/R-Judge/harmful.json`, `benign.json`
- 特点: 已包含 Agent 操作，无需模拟
- 标签: label=1 (unsafe), label=0 (safe)

## 使用流程

### 快速开始
```bash
# 1. 进入目录
cd comparison/AGrail4Agent-main/DAS

# 2. 测试设置
python test_defend_setup.py

# 3. 运行评估
bash run_defend_evaluation.sh

# 4. 汇总结果
python summarize_defend_results.py
```

### 单独测试
```bash
# 测试单个数据集（调试模式）
python evaluate_defend.py --dataset agentharm --data_type harmful --debug

# 查看实时日志
tail -f logs/defend_comparison/agentharm_harmful.log
```

## 输出结果

### 目录结构
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
├── summary_gpt-4o.json          # 详细汇总
└── comparison_table_gpt-4o.csv  # 对比表格
```

### 结果字段
每个CSV文件包含：
- `user_request`: 用户请求
- `agent_actions`: Agent 操作
- `label`: 真实标签 (0=harmful, 1=benign)
- `decision`: AGrail 判断 (safe/unsafe/error)
- `guardrail_output`: 检查输出
- `check_process`: 检查过程
- `code`: 生成的检查代码
- `token_length`: Token 使用量

## 评估指标

### 核心指标
1. **Accuracy**: 总体准确率
2. **Precision**: 检测精确率
3. **Recall**: 检测召回率
4. **F1 Score**: F1分数
5. **False Positive Rate**: 误报率（良性误判为恶意）
6. **False Negative Rate**: 漏报率（恶意误判为良性）

### 混淆矩阵
- TP: Harmful 正确识别为 Unsafe
- TN: Benign 正确识别为 Safe
- FP: Benign 误判为 Unsafe（过度防御）
- FN: Harmful 误判为 Safe（防御失效）

## 对比实验设计

### DEFEND vs AGrail4Agent

| 维度 | DEFEND | AGrail4Agent |
|------|--------|--------------|
| 方法 | 自进化安全工具 | Guardrail机制 |
| 工具生成 | 动态生成 | 预定义检查 |
| 优化能力 | 持续优化 | 固定规则 |
| 适应性 | 高 | 中 |
| 可解释性 | 中 | 高 |

### 实验步骤
1. 使用相同数据集（AgentHarm, ASB, R-Judge）
2. 使用相同的评估指标
3. 对比准确率、误报率、漏报率
4. 分析各自优劣

## 注意事项

### 1. 环境要求
- Python 3.8+
- Docker（用于 Container）
- GPU（可选，用于加速）
- OpenAI API 密钥

### 2. 运行时间
- 每个数据集约需 30-60 分钟
- 建议使用后台运行
- 可以并行运行多个数据集

### 3. 错误处理
- 脚本会自动跳过出错的项
- 错误项标记为 "error"
- 可以使用 `--restart` 重新运行

### 4. 资源消耗
- 内存: 建议 8GB+
- 存储: 约 1GB（结果和日志）
- API 调用: 约 1000-5000 次

## 下一步工作

### 1. 完成评估
```bash
bash run_defend_evaluation.sh
```

### 2. 分析结果
```bash
python summarize_defend_results.py
```

### 3. 与 DEFEND 对比
- 运行 DEFEND 框架的评估
- 对比两个框架的结果
- 分析优劣势

### 4. 撰写报告
- 准确率对比
- 误报/漏报分析
- 运行效率对比
- 适用场景分析

## 常见命令

```bash
# 测试设置
python test_defend_setup.py

# 运行评估
bash run_defend_evaluation.sh

# 单独运行
python evaluate_defend.py --dataset agentharm --data_type harmful

# 调试模式
python evaluate_defend.py --dataset agentharm --data_type harmful --debug

# 查看日志
tail -f logs/defend_comparison/agentharm_harmful.log

# 汇总结果
python summarize_defend_results.py

# 查看结果
cat result/defend_comparison/comparison_table_gpt-4o.csv
```

## 联系和支持

如有问题，请：
1. 查看 `README_DEFEND_EVALUATION.md` 详细文档
2. 查看 `QUICK_REFERENCE.md` 快速参考
3. 运行 `python test_defend_setup.py` 测试环境
4. 查看日志文件排查错误

---

**创建日期**: 2025-12-08
**版本**: 1.0
**状态**: 就绪
