python pipeline.py --restart

nohup python pipeline.py --restart > run.log 2>&1 &

nohup python pipeline.py --restart > logs/run_benign.log 2>&1 &