#!/usr/bin/env python3
"""Exercise drought outputs through the AURORA mission planner."""

from __future__ import annotations

import json
from pathlib import Path

from integrated_mission import plan_mission, update_belief_after_observation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "artifacts" / "inference_validation_results.json").read_text())
    regions = results["lowest_predicted_regions"] + results["highest_predicted_regions"]
    rows = [
        {
            "region": item["region"],
            "drought_index": float(item["mean_probability"]) * 100.0,
            "confidence": max(0.0, min(1.0, 1.0 - abs(float(item["mean_probability"]) - float(item["mean_soft_label"])))),
            "lat": item.get("mean_lat", 0.0),
            "lon": item.get("mean_lon", 0.0),
        }
        for item in regions
    ]
    plan = plan_mission(rows)
    if plan["selection"] is None:
        raise RuntimeError("Mission planner selected no feasible observation.")
    selected = next(row for row in rows if row["region"] == plan["selection"]["region"])
    posterior = update_belief_after_observation(
        selected,
        observed_drought_index=selected["drought_index"] * 0.9,
    )
    output = {
        "input_regions": len(rows),
        "selected_region": plan["selection"]["region"],
        "selected_utility": plan["selection"]["utility"],
        "posterior_update": posterior,
        "plan": plan,
    }
    (root / "artifacts" / "integrated_system_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
