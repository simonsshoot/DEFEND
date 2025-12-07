python pipeline.py --restart

nohup python pipeline.py --restart > run.log 2>&1 &

nohup python pipeline.py --restart > logs/run_benign_new.log 2>&1 &
nohup python pipeline.py --restart > logs/run_environment_new.log 2>&1 &
nohup python pipeline.py --restart > logs/run_agentsafebench.log 2>&1 &
python pipeline.py --restart --debug_mode

nohup python pipeline.py --restart --debug_mode --need_simulate > logs/run_agentharm.log 2>&1 &
nohup python pipeline.py --restart > logs/run_agentharm_benign.log 2>&1 &



==============================================================================  
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset agentharm --risk_memory lifelong_library/risks_agentharm_new.json --tool_memory lifelong_library/tools_agentharm_new.json --debug_file debug/agentharm_debug.json > logs/run_agentharm_debug.log 2>&1 &

nohup python pipeline.py --restart --debug_mode --need_simulate --dataset agentharm_benign --risk_memory lifelong_library/risks_agentharm_benign.json --tool_memory lifelong_library/tools_agentharm_benign.json --debug_file results/simulate_agentharm_benign.json > logs/run_agentharm_benign.log 2>&1 &

===========================agent harm良性数据的模拟代理数据生成====================================
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset agentharm_benign --risk_memory lifelong_library/risks_agentharm_benign.json --tool_memory lifelong_library/tools_agentharm_benign.json --debug_file /home/beihang/yx/DEFEND/data/agentharm/benign_simulate.json > logs/run_agentharm_benign.log 2>&1 &

===========================agent harm风险数据的模拟代理数据生成====================================
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset agentharm --risk_memory lifelong_library/risks_agentharm.json --tool_memory lifelong_library/tools_agentharm.json --debug_file /home/beihang/yx/DEFEND/data/agentharm/harmful_simulate.jsonl > logs/run_agentharm_harmful.log 2>&1 &

========================safeOS良性数据的模拟代理数据生成==========================
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset benign --risk_memory lifelong_library/risks_safeOSbenign.json --tool_memory lifelong_library/tools_safeOSbenign.json --debug_file /home/beihang/yx/DEFEND/data/safeOS/benign_simulate.jsonl > logs/run_safeos_benign.log 2>&1 &

===========================agent harm良性数据的数据生成(无模拟)====================================
nohup python pipeline.py --restart --debug_mode --dataset agentharm_benign --risk_memory lifelong_library/risks_agentharm_benign.json --tool_memory lifelong_library/tools_agentharm_benign.json --debug_file /home/beihang/yx/DEFEND/data/agentharm/benign_simulate.json > logs/run_agentharm_benign.log 2>&1 &

=====================agentsafetybench风险数据的模拟代理数据生成==========================
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset agentsafebench --risk_memory lifelong_library/risks_asb_harmful.json --tool_memory lifelong_library/tools_asb_harmful.json --debug_file /home/beihang/yx/DEFEND/data/ASB/harmful_simulate.jsonl > logs/run_asb_harmful.log 2>&1 &

=====================agentsafetybench良性数据的模拟代理数据生成==========================
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset asb_benign --risk_memory lifelong_library/risks_asb_benign.json --tool_memory lifelong_library/tools_asb_benign.json --debug_file /home/beihang/yx/DEFEND/data/ASB/benign_simulate.jsonl > logs/run_asb_benign.log 2>&1 &

=====================agentsafetybench良性数据生成(无模拟)==========================
nohup python pipeline.py --restart --debug_mode --dataset asb_benign --risk_memory lifelong_library/risks_asb_benign.json --tool_memory lifelong_library/tools_asb_benign.json --debug_file /home/beihang/yx/DEFEND/data/ASB/benign_simulate.jsonl --debug_doubt_tool_path debugs/asb_benign.log > logs/run_asb_benign.log 2>&1 &

===========================agent harm良性数据的数据生成(无模拟)=========================
nohup python pipeline.py --restart --debug_mode --dataset agentharm_benign --risk_memory lifelong_library/risks_agentharm_benign.json --tool_memory lifelong_library/tools_agentharm_benign.json --debug_file /home/beihang/yx/DEFEND/data/agentharm/benign_simulate.json --debug_doubt_tool_path debugs/agentharm_benign.log > logs/run_agentharm_benign.log 2>&1 &