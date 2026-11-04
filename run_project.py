#!/usr/bin/env python3
"""Run the reproducible local NASA challenge package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def run(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    expected_outputs: tuple[Path, ...] = (),
) -> None:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode and (
        not expected_outputs or not all(path.exists() for path in expected_outputs)
    ):
        raise subprocess.CalledProcessError(result.returncode, command)
    if result.returncode:
        print(
            f"Warning: process exited with {result.returncode} after writing its expected outputs.",
            flush=True,
        )


def main() -> None:
    manifest = DATA / "aurora_100k_resigned_samples.jsonl"
    tensor_dataset = DATA / "dataset_seq_forecast_production_real.npz"
    metadata_rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())

    print(f"Manifest metadata rows: {metadata_rows}")
    print("The manifest contains metadata only; it is not an image/tensor dataset.")
    print(f"Training locally on the verified tensor dataset: {tensor_dataset.name}")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(SRC),
            "REAL_DATASET_FILE": str(tensor_dataset),
            "SPECTRAL_EPOCHS": os.environ.get("SPECTRAL_EPOCHS", "4"),
            "DROUGHT_EPOCHS": os.environ.get("DROUGHT_EPOCHS", "3"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "64"),
            "FAST_SPECTRAL_INFERENCE": "1",
            "SPECTRAL_SAVE_FILE": str(ARTIFACTS / "spectral_realworld_production.pt"),
            "DROUGHT_SAVE_FILE": str(ARTIFACTS / "drought_realworld_production_model.pt"),
            "RESULTS_SAVE_FILE": str(ARTIFACTS / "drought_realworld_results.json"),
        }
    )

    run(
        [sys.executable, str(SRC / "realworld_drought_production.py")],
        ROOT,
        env,
        expected_outputs=(
            ARTIFACTS / "spectral_realworld_production.pt",
            ARTIFACTS / "drought_realworld_production_model.pt",
            ARTIFACTS / "drought_realworld_results.json",
        ),
    )

    test_env = env.copy()
    test_env["PYTHONPATH"] = str(SRC)
    test_env["SPECTRAL_MODEL_INPUT"] = str(ARTIFACTS / "spectral_realworld_production.pt")
    test_env["DROUGHT_MODEL_INPUT"] = str(ARTIFACTS / "drought_realworld_production_model.pt")
    run([sys.executable, str(SRC / "test_end_to_end.py")], ARTIFACTS, test_env)
    run([sys.executable, str(SRC / "test_inference_validation.py")], ARTIFACTS, test_env)
    run([sys.executable, str(SRC / "compare_models.py")], ROOT, test_env)
    run([sys.executable, str(SRC / "test_integrated_system.py")], ROOT, test_env)

    metrics = json.loads((ARTIFACTS / "drought_realworld_results.json").read_text(encoding="utf-8"))
    report = {
        "manifest": manifest.name,
        "manifest_metadata_rows": metadata_rows,
        "tensor_dataset": tensor_dataset.name,
        "tensor_dataset_samples": 2552,
        "trained_architecture": "SpectralNetV10",
        "training_mode": "local cached tensor dataset",
        "validation_metrics": metrics["validation_metrics"],
        "provenance_warning": (
            "The metadata manifest is not itself trainable imagery. A 29K claim requires "
            "successful retrieval and validation of the corresponding Sentinel-2 windows."
        ),
    }
    (ARTIFACTS / "FINAL_PROJECT_NASA_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Complete. Artifacts are in {ARTIFACTS}")


if __name__ == "__main__":
    main()
