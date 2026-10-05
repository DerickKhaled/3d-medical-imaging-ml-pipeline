# Runs the complete pipeline on Windows (the Makefile equivalent for PowerShell).
#
#   .\scripts\run_pipeline.ps1                  # full run, including ~55 min of CPU training
#   .\scripts\run_pipeline.ps1 -Experiment EXP-001 -SkipTraining   # reuse an existing experiment
#
# Stops at the first failing step.

param(
    [string]$Experiment = "EXP-001",
    [string]$Version = "v1.0",
    [string]$Scan = "data/Task04_Hippocampus/imagesTr/hippocampus_017.nii.gz",  # test split
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

function Step([string]$title, [string[]]$arguments) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "step failed: $title" }
}

Step "Download dataset" @("-m", "src.data.download", "--out", "data")
Step "Build manifest" @("-m", "src.data.manifest", "--config", "configs/data.yaml")
if (-not $SkipTraining) {
    Step "Train" @("-m", "src.training.train", "--config", "configs/train.yaml")
}
Step "Register" @("-m", "src.registry.model_registry", "register",
    "--experiment", $Experiment, "--version", $Version)
Step "Evaluate on test split" @("-m", "src.evaluation.evaluate",
    "--model-version", $Version, "--split", "test")
Step "Promote to validated" @("-m", "src.registry.model_registry", "promote", "--version", $Version,
    "--to", "validated", "--reason", "test-split evaluation meets the release gate")
Step "Promote to production" @("-m", "src.registry.model_registry", "promote", "--version", $Version,
    "--to", "production", "--reason", "approved for the demo release")
Step "Registry" @("-m", "src.registry.model_registry", "list")

Write-Host "`n=== Inference ===" -ForegroundColor Cyan
$output = & $python -m src.inference.predict --input $Scan --model-version production
if ($LASTEXITCODE -ne 0) { throw "step failed: Inference" }
$output | Write-Host
$case = (($output | Select-String "^inference_id: ").Line -split ": ")[1]

Step "3D meshes" @("-m", "src.reconstruction.mesh", "--inference-id", $case)
Step "Slice figure" @("-m", "src.visualization.slices", "--case", $case, "--out", "docs/images/slices.png")
Step "Viewer screenshot" @("-m", "src.visualization.viewer", "--case", $case,
    "--screenshot", "docs/images/viewer.png")
Step "Lineage trace" @("-m", "src.lineage.trace", "--inference-id", $case)

Write-Host "`nDone. Open the interactive viewer with:" -ForegroundColor Green
Write-Host "  python -m src.visualization.viewer --case $case"
