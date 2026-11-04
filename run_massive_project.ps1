$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = "G:\CubeSat\.venv\Scripts\python.exe"

& $python .\src\materialize_manifest.py `
  --manifest .\data\aurora_100k_resigned_samples.jsonl `
  --output .\data\massive_10000_rgb_spec.npz `
  --limit 10000 `
  --cache .\.remote_cache

$env:MASSIVE_DATASET = Join-Path $PSScriptRoot "data\massive_10000_rgb_spec.npz"
& $python .\src\train_massive_spectral.py

$env:SPECTRAL_MODEL_INPUT = Join-Path $PSScriptRoot "artifacts\spectral_massive_10000.pt"
& $python .\run_project.py
