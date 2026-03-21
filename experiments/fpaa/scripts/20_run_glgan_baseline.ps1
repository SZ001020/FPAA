param(
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

$runId = "LoveDA_{0}_GLGAN_s{1}_{2}" -f $Direction, $Seed, (Get-Date -Format "yyyyMMdd_HHmmss")
$logPath = Join-Path $logDir ("{0}.log" -f $runId)

$env:PYTHONHASHSEED = "$Seed"
$env:EXP_DIRECTION = "$Direction"
$env:EXP_SEED = "$Seed"

if ($Direction -eq "R2U") {
    $entry = Join-Path $ProjectRoot "GLGAN\GLGAN_LoveDA_R2U.py"
} else {
    $entry = Join-Path $ProjectRoot "GLGAN\GLGAN_LoveDA_U2R.py"
}

Write-Host "Run ID: $runId"
Write-Host "Entry : $entry"
Write-Host "Log   : $logPath"
Write-Host ""
Write-Host "提示: 请先确认 GLGAN 脚本中的数据目录与预训练权重路径。"
Write-Host "开始执行..."

& $PythonExe $entry *>&1 | Tee-Object -FilePath $logPath

Write-Host "Finished: $runId"
