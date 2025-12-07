cd /home/beihang/yx/DEFEND/comparison/sheildagent

# 评估 AgentHarm Benign 数据集
echo "Evaluating AgentHarm Benign dataset..."
python evaluate.py --datasets agentharm_benign --output_dir results

# 评估 ASB Benign 数据集
echo "Evaluating ASB Benign dataset..."
python evaluate.py --datasets asb_benign --output_dir results

# 评估 R-Judge Harmful 数据集
echo "Evaluating R-Judge Harmful dataset..."
python evaluate.py --datasets rjudge_harmful --output_dir results

# 评估所有数据集
echo "Evaluating all datasets..."
nohup python evaluate.py --datasets agentharm_benign asb_benign rjudge_harmful --output_dir results > logs/shieldagent_eval.log 2>&1 &

echo "Evaluation jobs submitted!"
