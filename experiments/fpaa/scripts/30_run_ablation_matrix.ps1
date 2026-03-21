param(
    [string]$MatrixCsv = (Join-Path $PSScriptRoot "..\configs\ablation_matrix.csv"),
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $MatrixCsv)) {
    throw "Matrix file not found: $MatrixCsv"
}

$rows = Import-Csv $MatrixCsv
if (-not $rows -or $rows.Count -eq 0) {
    throw "Matrix file is empty: $MatrixCsv"
}

Write-Host "Loaded matrix rows: $($rows.Count)"

foreach ($row in $rows) {
    $runName = $row.run_name
    $backbone = $row.backbone
    $direction = $row.direction
    $seed = [int]$row.seed
    $quality = $row.quality_mode
    $liteDa = $row.use_lite_da

    Write-Host "----------------------------------------"
    Write-Host "Run: $runName"
    Write-Host "backbone=$backbone direction=$direction seed=$seed quality_mode=$quality use_lite_da=$liteDa"

    $env:FPAA_QUALITY_MODE = "$quality"
    $env:FPAA_USE_LITE_DA = "$liteDa"

    if ($backbone -eq "GLGAN") {
        & (Join-Path $PSScriptRoot "20_run_glgan_baseline.ps1") -Direction $direction -Seed $seed -PythonExe $PythonExe
    } else {
        & (Join-Path $PSScriptRoot "10_run_fm_baseline.ps1") -Backbone $backbone -Direction $direction -Seed $seed -PythonExe $PythonExe
    }
}

Write-Host "----------------------------------------"
Write-Host "All matrix runs finished."
