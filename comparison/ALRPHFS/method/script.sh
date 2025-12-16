nohup python offline_train_harmful.py > train_harmful.log 2>&1 &

nohup python offline_train_benign.py > train_benign.log 2>&1 &

nohup python eval_rjudge.py > eval_rjudge.log 2>&1 &