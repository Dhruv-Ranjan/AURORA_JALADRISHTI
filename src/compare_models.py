#!/usr/bin/env python3
"""Compare the paper baseline and the newly trained spectral model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from real_data_pipeline import (
    compute_reconstruction_metrics,
    load_real_dataset,
    load_spectral_model,
    run_spectral_reconstruction,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset_seq_forecast_production_real.npz"
BASELINE = ROOT / "artifacts" / "spectral_v10_1_balanced.pt"
NEW_MODEL = ROOT / "artifacts" / "spectral_realworld_production.pt"


def evaluate(path: Path, bundle) -> dict[str, float]:
    model = load_spectral_model(path)
    prediction = run_spectral_reconstruction(model, bundle.rgb, batch_size=64)
    return compute_reconstruction_metrics(prediction, bundle.spec)


def main() -> None:
    bundle = load_real_dataset(DATASET)
    baseline = evaluate(BASELINE, bundle)
    new = evaluate(NEW_MODEL, bundle)
    result = {
        "dataset": DATASET.name,
        "samples": len(bundle.y_hard),
        "baseline_checkpoint": BASELINE.name,
        "new_checkpoint": NEW_MODEL.name,
        "baseline": baseline,
        "new": new,
        "delta_new_minus_baseline": {
            key: new[key] - baseline[key] for key in ("mse", "mae", "rmse")
        },
    }
    output = ROOT / "artifacts" / "model_comparison.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
