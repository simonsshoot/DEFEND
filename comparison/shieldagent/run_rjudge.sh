# export CUDA_VISIBLE_DEVICES=0,1,2
# MODEL_PATH="/data/Content_Moderation/ShieldAgent"
# OUTPUT_DIR=results/rjudge


# # Application
# echo "# Application - Harmful"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Application --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_application_harmful.out 2>&1 &"
# echo ""

# echo "# Application - Benign"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Application --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_application_benign.out 2>&1 &"
# echo ""

# # Finance
# echo "# Finance - Harmful"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Finance --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_finance_harmful.out 2>&1 &"
# echo ""

# echo "# Finance - Benign"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Finance --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_finance_benign.out 2>&1 &"
# echo ""

# # IoT
# echo "# IoT - Harmful"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder IoT --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_iot_harmful.out 2>&1 &"
# echo ""

# echo "# IoT - Benign"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder IoT --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_iot_benign.out 2>&1 &"
# echo ""

# # Program
# echo "# Program - Harmful"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Program --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_program_harmful.out 2>&1 &"
# echo ""

# echo "# Program - Benign"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Program --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_program_benign.out 2>&1 &"
# echo ""

# # Web
# echo "# Web - Harmful"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Web --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_web_harmful.out 2>&1 &"
# echo ""

# echo "# Web - Benign"
# echo "nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Web --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_web_benign.out 2>&1 &"
# echo ""



# Application - Harmful
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Application --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_application_harmful.out 2>&1 &

# # Application - Benign
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Application --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_application_benign.out 2>&1 &

# # Finance - Harmful
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Finance --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_finance_harmful.out 2>&1 &

# # Finance - Benign
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Finance --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_finance_benign.out 2>&1 &

# # IoT - Harmful
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder IoT --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_iot_harmful.out 2>&1 &

# # IoT - Benign
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder IoT --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_iot_benign.out 2>&1 &

# # Program - Harmful
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Program --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_program_harmful.out 2>&1 &

# # Program - Benign
# nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Program --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_program_benign.out 2>&1 &

# Web - Harmful
nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_harmful --subfolder Web --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_web_harmful.out 2>&1 &

# Web - Benign
nohup python evaluate.py --model_path /data/Content_Moderation/ShieldAgent --dataset rjudge_benign --subfolder Web --output_dir results/rjudge --simulate_data False > logs/rjudge/rjudge_web_benign.out 2>&1 &