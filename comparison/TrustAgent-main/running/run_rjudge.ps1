# TrustAgent R-Judge 子文件夹批量评估脚本 (PowerShell版本)
# 评估所有 R-Judge 子文件夹（Application, Finance, IoT, Program, Web）的 harmful 和 benign 数据

# 模型配置
$AGENT_LLM = "deepseek-chat"
$SIMULATOR_LLM = "deepseek-chat"
$SAFETY_CHECKER_LLM = "deepseek-chat"
$AGENT_TEMP = 0.0
$SIMULATOR_TYPE = "adv_thought"
$USE_RETRIEVER = "openai"

# R-Judge 子文件夹列表
$SUBFOLDERS = @("Application", "Finance", "IoT", "Program", "Web")

# 数据类型列表
$DATA_TYPES = @("harmful", "benign")

# Safety methods (可选)
$SAFETY_METHODS = "--regulation_prompting"  # 可以添加 --regulation_check 或 --regulation_learning

Write-Host "========================================"
Write-Host "TrustAgent R-Judge Batch Evaluation"
Write-Host "========================================"
Write-Host "Agent LLM: $AGENT_LLM"
Write-Host "Simulator LLM: $SIMULATOR_LLM"
Write-Host "Safety Checker LLM: $SAFETY_CHECKER_LLM"
Write-Host "Simulator Type: $SIMULATOR_TYPE"
Write-Host "Subfolders: $($SUBFOLDERS -join ', ')"
Write-Host "Data Types: $($DATA_TYPES -join ', ')"
Write-Host "========================================"
Write-Host ""

# 创建结果目录
New-Item -ItemType Directory -Force -Path "results/trajectory/rjudge" | Out-Null
New-Item -ItemType Directory -Force -Path "results/score/rjudge" | Out-Null

# 用于统计的变量
$global:total_samples = 0
$global:successful_samples = 0
$global:failed_samples = 0

# 遍历所有子文件夹和数据类型
foreach ($subfolder in $SUBFOLDERS) {
    foreach ($data_type in $DATA_TYPES) {
        # 获取数据文件中的样本数量
        $data_file = "..\..\..\data\R-Judge\$subfolder\$data_type.json"
        
        if (-not (Test-Path $data_file)) {
            Write-Host "Warning: File not found: $data_file" -ForegroundColor Yellow
            continue
        }
        
        # 使用 Python 获取样本数量
        $sample_count = python -c "import json; print(len(json.load(open('$data_file'))))"
        
        if (-not $sample_count) {
            Write-Host "Error: Could not determine sample count for $data_file" -ForegroundColor Red
            continue
        }
        
        Write-Host "=========================================" -ForegroundColor Cyan
        Write-Host "Evaluating: R-Judge / $subfolder / $data_type" -ForegroundColor Cyan
        Write-Host "Total samples: $sample_count" -ForegroundColor Cyan
        Write-Host "=========================================" -ForegroundColor Cyan
        
        # 遍历所有样本
        for ($case_idx = 0; $case_idx -lt $sample_count; $case_idx++) {
            $global:total_samples++
            Write-Host "Processing sample $($case_idx + 1)/$sample_count..." -ForegroundColor White
            
            # 构建命令参数
            $pythonArgs = @(
                "main.py",
                "--dataset", "rjudge",
                "--subfolder", $subfolder,
                "--data_type", $data_type,
                "--case_idx", $case_idx,
                "--agent_llm_type", $AGENT_LLM,
                "--simulator_llm_type", $SIMULATOR_LLM,
                "--safety_checker_llm_type", $SAFETY_CHECKER_LLM,
                "--agent_temp", $AGENT_TEMP,
                "--simulator_type", $SIMULATOR_TYPE,
                "--use_retriever", $USE_RETRIEVER
            )
            
            # 添加 safety methods
            if ($SAFETY_METHODS) {
                $pythonArgs += $SAFETY_METHODS.Split(" ")
            }
            
            # 执行 Python 命令
            & python $pythonArgs
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Successfully processed sample $case_idx" -ForegroundColor Green
                $global:successful_samples++
            } else {
                Write-Host "✗ Failed to process sample $case_idx" -ForegroundColor Red
                $global:failed_samples++
            }
        }
        
        Write-Host ""
    }
}

Write-Host "========================================"
Write-Host "All R-Judge evaluations completed!" -ForegroundColor Green
Write-Host "Results saved in:"
Write-Host "  Trajectories: results/trajectory/"
Write-Host "  Scores: results/score/"
Write-Host "========================================"

# 显示总体统计
Write-Host ""
Write-Host "Overall Statistics:" -ForegroundColor Cyan
Write-Host "  Total samples processed: $global:total_samples"
Write-Host "  Successful: $global:successful_samples" -ForegroundColor Green
Write-Host "  Failed: $global:failed_samples" -ForegroundColor Red

# 汇总统计信息
Write-Host ""
Write-Host "Summary of results:" -ForegroundColor Cyan
foreach ($subfolder in $SUBFOLDERS) {
    Write-Host "----------------------------------------"
    Write-Host "Subfolder: $subfolder" -ForegroundColor Yellow
    foreach ($data_type in $DATA_TYPES) {
        $result_dir = "results/score"
        $pattern = "${AGENT_LLM}_rjudge_${subfolder}_${data_type}_*.json"
        
        # 获取匹配的文件
        $files = Get-ChildItem -Path $result_dir -Filter $pattern -ErrorAction SilentlyContinue
        $count = $files.Count
        
        Write-Host "  $data_type`: $count samples evaluated"
        
        # 计算准确率（如果有结果文件）
        if ($count -gt 0) {
            try {
                $correct = 0
                foreach ($file in $files) {
                    $content = Get-Content $file.FullName | ConvertFrom-Json
                    if ($content.rjudge_correct -eq $true) {
                        $correct++
                    }
                }
                
                $accuracy = [math]::Round(($correct / $count) * 100, 2)
                Write-Host "    Accuracy: $correct/$count ($accuracy%)" -ForegroundColor Green
            }
            catch {
                Write-Host "    Could not calculate accuracy: $_" -ForegroundColor Yellow
            }
        }
    }
}
Write-Host "----------------------------------------"

Write-Host ""
Write-Host "Evaluation completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
