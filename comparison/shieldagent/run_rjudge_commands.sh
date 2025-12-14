MODEL_PATH="/home/beihang/yx/models/shieldagent"
OUTPUT_DIR="results"


# Application
echo "# Application - Harmful"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_harmful --subfolder Application --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_application_harmful.out 2>&1 &"
echo ""

echo "# Application - Benign"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_benign --subfolder Application --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_application_benign.out 2>&1 &"
echo ""

# Finance
echo "# Finance - Harmful"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_harmful --subfolder Finance --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_finance_harmful.out 2>&1 &"
echo ""

echo "# Finance - Benign"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_benign --subfolder Finance --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_finance_benign.out 2>&1 &"
echo ""

# IoT
echo "# IoT - Harmful"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_harmful --subfolder IoT --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_iot_harmful.out 2>&1 &"
echo ""

echo "# IoT - Benign"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_benign --subfolder IoT --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_iot_benign.out 2>&1 &"
echo ""

# Program
echo "# Program - Harmful"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_harmful --subfolder Program --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_program_harmful.out 2>&1 &"
echo ""

echo "# Program - Benign"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_benign --subfolder Program --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_program_benign.out 2>&1 &"
echo ""

# Web
echo "# Web - Harmful"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_harmful --subfolder Web --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_web_harmful.out 2>&1 &"
echo ""

echo "# Web - Benign"
echo "nohup python evaluate.py --model_path \"$MODEL_PATH\" --dataset rjudge_benign --subfolder Web --output_dir \"$OUTPUT_DIR\" --simulate_data False > nohup_rjudge_web_benign.out 2>&1 &"
echo ""

