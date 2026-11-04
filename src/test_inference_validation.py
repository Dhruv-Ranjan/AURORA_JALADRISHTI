#!/usr/bin/env python3
"""
Inference sanity-check on real cached locations.

Reports the most drought-like and least drought-like cached regions under the
current spectral+drought model pair, using actual dataset coordinates.
"""

import json

from real_data_pipeline import (
    DEVICE,
    load_drought_model,
    load_real_dataset,
    load_spectral_model,
    predict_drought_probabilities,
    region_summary,
    run_spectral_reconstruction,
)


print("=" * 80)
print("REAL-LOCATION INFERENCE VALIDATION")
print("=" * 80)
print(f"Device: {DEVICE}")

bundle = load_real_dataset()
spectral_model = load_spectral_model()
drought_model = load_drought_model()

pred_spec = run_spectral_reconstruction(spectral_model, bundle.rgb, batch_size=8)
probs, confs = predict_drought_probabilities(drought_model, pred_spec, batch_size=8)

all_regions = sorted(set(bundle.region_name.tolist()))
summaries = region_summary(bundle, probs, confs, all_regions)
lowest = summaries[:5]
highest = summaries[-5:]

print("\nLowest predicted drought regions")
for item in lowest:
    print(
        f"  {item['region']}: prob={item['mean_probability']:.3f} "
        f"target_soft={item['mean_soft_label']:.3f} samples={item['samples']}"
    )

print("\nHighest predicted drought regions")
for item in reversed(highest):
    print(
        f"  {item['region']}: prob={item['mean_probability']:.3f} "
        f"target_soft={item['mean_soft_label']:.3f} samples={item['samples']}"
    )

results = {
    "dataset": bundle.dataset_path.name,
    "device": str(DEVICE),
    "lowest_predicted_regions": lowest,
    "highest_predicted_regions": list(reversed(highest)),
    "note": "Results are ranked on actual cached regions and should be interpreted alongside sample count and proxy-vs-real label availability.",
}

with open("inference_validation_results.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print("\nSaved: inference_validation_results.json")
