param(
    [string]$DatasetRoot = "E:/Alluserdata/RSS/autodl-tmp/dataset"
)

$ErrorActionPreference = "Stop"

function Get-ImageCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    return (Get-ChildItem -Path $Path -Recurse -File -Include *.png,*.jpg,*.jpeg,*.tif,*.tiff | Measure-Object).Count
}

$targets = @(
    "LoveDA/Train",
    "LoveDA/Val",
    "LoveDA/Test",
    "Hunan_Dataset/train",
    "Hunan_Dataset/val",
    "Hunan_Dataset/test",
    "Potsdam",
    "Vaihingen"
)

Write-Host "Dataset root: $DatasetRoot"
Write-Host "----------------------------------------"
foreach ($t in $targets) {
    $p = Join-Path $DatasetRoot $t
    $exists = Test-Path $p
    $count = Get-ImageCount -Path $p
    Write-Host ("{0,-24} exists={1,-5} images={2}" -f $t, $exists, $count)
}

Write-Host "----------------------------------------"
Write-Host "Data check finished."
