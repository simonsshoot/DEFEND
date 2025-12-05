python pipeline.py --restart

nohup python pipeline.py --restart > run.log 2>&1 &

nohup python pipeline.py --restart > logs/run_benign_new.log 2>&1 &
nohup python pipeline.py --restart > logs/run_environment_new.log 2>&1 &
nohup python pipeline.py --restart > logs/run_agentsafebench.log 2>&1 &
python pipeline.py --restart --debug_mode

nohup python pipeline.py --restart --debug_mode --need_simulate > logs/run_agentharm.log 2>&1 &
nohup python pipeline.py --restart > logs/run_agentharm_benign.log 2>&1 &



==============================================================================  
nohup python pipeline.py --restart --debug_mode --need_simulate --dataset agentharm --risk_memory lifelong_library/risks_agentharm_new.json --tool_memory lifelong_library/tools_agentharm_new.json --debug_file debug/agentharm_debug.jsonl > logs/run_agentharm_debug.log 2>&1 &