# AURORA-JALADRISHTI

This is the public `main` branch for the NASA Space Apps AURORA/Jaladrishti
drought-monitoring system.

The project combines:

1. `SpectralNetV10` RGB-to-six-band spectral reconstruction.
2. Physics-guided temporal drought prediction.
3. AURORA resource-aware mission planning and belief updates.

## Important data distinction

`data/aurora_100k_resigned_samples.jsonl` contains **29,487 metadata rows**:
scene IDs, coordinates, dates, bands, and patch IDs. It does not contain RGB
pixels or spectral targets. Metadata cannot be used as tensors for training.
The package therefore does not make a false 29K training claim.

The verified local training dataset is
`data/dataset_seq_forecast_production_real.npz`, containing 2,552 three-date
sequences. The model is the paper architecture, `SpectralNetV10`, followed by
the physics-guided temporal drought model.

## Run everything

From PowerShell:

```powershell
cd G:\CubeSat\FINAL_PROJECT_NASA
G:\CubeSat\.venv\Scripts\python.exe .\run_project.py
```

The runner trains, evaluates end-to-end inference, writes validation metrics,
and creates all outputs under `artifacts\`.

To change training epochs:

```powershell
$env:SPECTRAL_EPOCHS="4"
$env:DROUGHT_EPOCHS="3"
G:\CubeSat\.venv\Scripts\python.exe .\run_project.py
```

## Artifacts

- `artifacts\spectral_realworld_production.pt`: trained SpectralNetV10.
- `artifacts\drought_realworld_production_model.pt`: trained drought model.
- `artifacts\drought_realworld_results.json`: validation metrics.
- `artifacts\end_to_end_test_results.json`: reconstruction and classification test.
- `artifacts\inference_validation_results.json`: regional inference sanity check.
- `artifacts\FINAL_PROJECT_NASA_report.json`: provenance and final summary.
- `artifacts\integrated_system_results.json`: drought outputs passed through
  the AURORA mission planner and belief update.
- The complete research manuscript is maintained locally but intentionally
  excluded from this public GitHub branch.

## Massive-data mode

The massive-data workflow is explicit and two-stage:

```powershell
cd G:\CubeSat\FINAL_PROJECT_NASA
G:\CubeSat\.venv\Scripts\python.exe .\src\materialize_manifest.py `
  --manifest .\data\aurora_100k_resigned_samples.jsonl `
  --output .\data\massive_10000_rgb_spec.npz `
  --limit 10000 `
  --cache .\.remote_cache
$env:MASSIVE_DATASET="G:\CubeSat\FINAL_PROJECT_NASA\data\massive_10000_rgb_spec.npz"
G:\CubeSat\.venv\Scripts\python.exe .\src\train_massive_spectral.py
```

This requests up to 10,000 candidates, but only patches that are successfully
retrieved, quality-checked, and persisted are trainable. In the verified run,
978 patches were valid; the other candidates were rejected by scene
availability, authorization, boundary, or pixel-quality checks. It does not
create drought labels from metadata. Drought training and testing remain on
paired temporal data, followed by the mission-planner integration.

The same workflow can be run end to end with:

```powershell
.\run_massive_project.ps1
```

The remote materialization is intentionally resumeless and may take hours.
Keep the process attached until `massive_10000_rgb_spec.npz` is written.
The output filename is historical; inspect its stored array shape and the
massive metrics report for the actual valid count.

## Requirements

Install with:

```powershell
G:\CubeSat\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

The package requires Python, PyTorch, NumPy, SciPy, Matplotlib, and the
geospatial/STAC packages listed in `requirements.txt`.

Large local datasets and checkpoints are intentionally excluded from GitHub
by `.gitignore`: the tensor cache is over 1 GB and GitHub rejects files of
that size. Place the local files back under `data\` and `artifacts\` before
running.

The best verified production run is recorded in
`artifacts\production_run_summary.json`. The remote massive-pretraining
checkpoint is retained locally as an ablation, but it is not the production
model because it reduced downstream performance.

## How to obtain a genuine 29K training run

The metadata manifest must be joined to downloaded and validated Sentinel-2
windows. The remote trainer can do that only while signed Planetary Computer
URLs are valid and network access is available. The resulting RGB/target arrays
must be persisted locally before training. Until that is done, report the
29,487 rows as metadata candidates, not as 29,487 training samples.
