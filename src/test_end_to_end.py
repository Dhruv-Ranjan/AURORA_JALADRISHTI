#!/usr/bin/env python3
"""
End-to-end validation on cached real satellite sequences.

This script does not fabricate satellite patches. It loads the cached
Sentinel-2 sequence dataset, reconstructs spectra from RGB with the trained
spectral model, and evaluates the drought model on those reconstructed
sequences.
"""

import json
import torch

from real_data_pipeline import (
    DEVICE,
    compute_reconstruction_metrics,
    evaluate_predictions,
    load_drought_model,
    load_real_dataset,
    load_spectral_model,
    predict_drought_probabilities,
    run_spectral_reconstruction,
)


print("=" * 80)
print("REAL-DATA END-TO-END VALIDATION")
print("=" * 80)
print(f"Device: {DEVICE}")

bundle = load_real_dataset()
print(f"Dataset: {bundle.dataset_path.name}")
print(f"Samples: {len(bundle.y_hard)}")
print(f"Regions: {len(set(bundle.region_name.tolist()))}")

spectral_model = load_spectral_model()
drought_model = load_drought_model()
checkpoint = torch.load("drought_realworld_production_model.pt", map_location="cpu")
threshold = float(checkpoint.get("best_threshold", 0.5)) if isinstance(checkpoint, dict) else 0.5

pred_spec = run_spectral_reconstruction(spectral_model, bundle.rgb, batch_size=8)
reconstruction = compute_reconstruction_metrics(pred_spec, bundle.spec)
probs, confs = predict_drought_probabilities(drought_model, pred_spec, batch_size=8)
classification = evaluate_predictions(probs, bundle.y_hard.numpy(), threshold=threshold)

print("\nReconstruction metrics")
print(f"  RMSE: {reconstruction['rmse']:.5f}")
print(f"  MAE:  {reconstruction['mae']:.5f}")
print(f"  MSE:  {reconstruction['mse']:.5f}")

print("\nDrought classification metrics")
print(f"  Accuracy:  {classification['accuracy']:.4f}")
print(f"  Precision: {classification['precision']:.4f}")
print(f"  Recall:    {classification['recall']:.4f}")
print(f"  F1:        {classification['f1']:.4f}")
print(f"  Mean prob: {classification['mean_probability']:.4f}")
print(f"  Threshold: {threshold:.2f}")

results = {
    "dataset": bundle.dataset_path.name,
    "device": str(DEVICE),
    "reconstruction": reconstruction,
    "classification": classification,
    "threshold": threshold,
    "note": "Evaluation uses cached real Sentinel-2 sequences. Targets may be proxy-derived where TerraClimate labels are unavailable.",
}

with open("end_to_end_test_results.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print("\nSaved: end_to_end_test_results.json")
