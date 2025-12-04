python pipeline.py --restart

nohup python pipeline.py --restart > run.log 2>&1 &

nohup python pipeline.py --restart > logs/run_benign_new.log 2>&1 &
nohup python pipeline.py --restart > logs/run_environment_new.log 2>&1 &
nohup python pipeline.py --restart > logs/run_agentsafebench.log 2>&1 &