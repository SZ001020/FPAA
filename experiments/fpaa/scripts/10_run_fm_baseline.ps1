param(
    [ValidateSet("MFNet", "SAM_RS")]
    [string]$Backbone = "MFNet",

    [ValidateSet("R2U", "U2R")]
    [string]$Direction = "R2U",

    [int]$Seed = 3407,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ExpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $ExpRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runId = "LoveDA_{0}_{1}_s{2}_{3}" -f $Direction, $Backbone, $Seed, (Get-Date -Format "yyyyMMdd_HHmmss")
$logPath = Join-Path $logDir ("{0}.log" -f $runId)

$env:PYTHONHASHSEED = "$Seed"
$env:EXP_DIRECTION = "$Direction"
$env:EXP_SEED = "$Seed"

if ($Backbone -eq "MFNet") {
    $entry = Join-Path $ProjectRoot "MFNet\train.py"
} else {
    $entry = Join-Path $ProjectRoot "SAM_RS\train.py"
}

Write-Host "Run ID: $runId"
Write-Host "Entry : $entry"
Write-Host "Log   : $logPath"
Write-Host ""
Write-Host "提示: 该仓库训练入口多为脚本内配置，请先在对应 utils/train 文件中确认 DATASET 与数据根目录设置。"
Write-Host "开始执行..."

& $PythonExe $entry *>&1 | Tee-Object -FilePath $logPath

Write-Host "Finished: $runId"
