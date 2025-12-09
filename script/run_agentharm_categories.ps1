# AgentHarm 分类数据集批量运行脚本 (PowerShell)
# 每个类别包含 harmful 和 benign 两类数据

Write-Host "=============================================="
Write-Host "AgentHarm 分类数据集批量运行"
Write-Host "=============================================="
Write-Host ""

# AgentHarm 的 8 个类别
$categories = @("Copyright", "Cybercrime", "Disinformation", "Drugs", "Fraud", "Harassment", "Hate", "Sexual")

# 创建必要的目录
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "lifelong_library" | Out-Null
New-Item -ItemType Directory -Force -Path "debugs" | Out-Null
New-Item -ItemType Directory -Force -Path "results" | Out-Null

Write-Host "=============================================="
Write-Host "步骤 1: 生成模拟数据（--need_simulate）"
Write-Host "=============================================="
Write-Host ""

foreach ($category in $categories) {
    Write-Host "处理类别: $category"
    
    # Harmful 数据模拟
    Write-Host "  - 生成 ${category} harmful 模拟数据..."
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList @(
        "pipeline.py",
        "--restart",
        "--debug_mode",
        "--need_simulate",
        "--dataset", "agentharm_${category}_harmful",
        "--risk_memory", "lifelong_library/risks_agentharm_${category}_harmful.json",
        "--tool_memory", "lifelong_library/tools_agentharm_${category}_harmful.json",
        "--debug_file", "data/agentharm/${category}/harmful_simulate.jsonl",
        "--debug_doubt_tool_path", "debugs/agentharm_${category}_harmful.log",
        "--debug_decision_path", "debugs/agentharm_${category}_harmful_decision.log"
    ) -RedirectStandardOutput "logs/simulate_agentharm_${category}_harmful.log" -RedirectStandardError "logs/simulate_agentharm_${category}_harmful_error.log"
    
    # Benign 数据模拟
    Write-Host "  - 生成 ${category} benign 模拟数据..."
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList @(
        "pipeline.py",
        "--restart",
        "--debug_mode",
        "--need_simulate",
        "--dataset", "agentharm_${category}_benign",
        "--risk_memory", "lifelong_library/risks_agentharm_${category}_benign.json",
        "--tool_memory", "lifelong_library/tools_agentharm_${category}_benign.json",
        "--debug_file", "data/agentharm/${category}/benign_simulate.jsonl",
        "--debug_doubt_tool_path", "debugs/agentharm_${category}_benign.log",
        "--debug_decision_path", "debugs/agentharm_${category}_benign_decision.log"
    ) -RedirectStandardOutput "logs/simulate_agentharm_${category}_benign.log" -RedirectStandardError "logs/simulate_agentharm_${category}_benign_error.log"
    
    Write-Host ""
}

Write-Host "所有模拟任务已启动！"
Write-Host ""
Write-Host "可以使用以下命令检查进度:"
Write-Host "  Get-Content logs\simulate_agentharm_*.log -Wait -Tail 20"
Write-Host ""
Write-Host "按任意键继续到步骤 2（确保步骤 1 已完成）..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "=============================================="
Write-Host "步骤 2: 运行评估（使用模拟数据）"
Write-Host "=============================================="
Write-Host ""

foreach ($category in $categories) {
    Write-Host "评估类别: $category"
    
    # Harmful 数据评估
    Write-Host "  - 评估 ${category} harmful 数据..."
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList @(
        "pipeline.py",
        "--restart",
        "--debug_mode",
        "--dataset", "agentharm_${category}_harmful",
        "--risk_memory", "lifelong_library/risks_agentharm_${category}_harmful.json",
        "--tool_memory", "lifelong_library/tools_agentharm_${category}_harmful.json",
        "--debug_doubt_tool_path", "debugs/agentharm_${category}_harmful.log",
        "--debug_decision_path", "debugs/agentharm_${category}_harmful_decision.log"
    ) -RedirectStandardOutput "logs/run_agentharm_${category}_harmful.log" -RedirectStandardError "logs/run_agentharm_${category}_harmful_error.log"
    
    # Benign 数据评估
    Write-Host "  - 评估 ${category} benign 数据..."
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList @(
        "pipeline.py",
        "--restart",
        "--debug_mode",
        "--dataset", "agentharm_${category}_benign",
        "--risk_memory", "lifelong_library/risks_agentharm_${category}_benign.json",
        "--tool_memory", "lifelong_library/tools_agentharm_${category}_benign.json",
        "--debug_doubt_tool_path", "debugs/agentharm_${category}_benign.log",
        "--debug_decision_path", "debugs/agentharm_${category}_benign_decision.log"
    ) -RedirectStandardOutput "logs/run_agentharm_${category}_benign.log" -RedirectStandardError "logs/run_agentharm_${category}_benign_error.log"
    
    Write-Host ""
}

Write-Host "所有评估任务已启动！"
Write-Host ""
Write-Host "可以使用以下命令检查进度:"
Write-Host "  Get-Content logs\run_agentharm_*.log -Wait -Tail 20"
Write-Host "  Get-Process python"
Write-Host ""
Write-Host "等待所有任务完成后，运行汇总脚本:"
Write-Host "  python summarize_results.py"
Write-Host ""
