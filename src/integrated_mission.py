"""Bridge the deployed drought estimates into a resource-aware EO mission plan.

This module deliberately uses the existing region forecast rows as the
science state. It does not pretend that a planned CubeSat observation is a
measured Sentinel-2 scene: the returned plan is a reproducible planning
simulation whose observation updates the belief state and can be replanned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MissionResources:
    battery_wh: float = 55.0
    storage_mb: float = 0.0
    storage_capacity_mb: float = 96.0
    minimum_battery_wh: float = 11.0
    observation_energy_wh: float = 1.5
    compute_energy_wh: float = 0.35
    product_size_mb: float = 8.0


@dataclass(frozen=True)
class ObservationCandidate:
    region: str
    lat: float
    lon: float
    access_window_minutes: int
    cloud_probability: float
    energy_wh: float
    data_mb: float
    information_gain: float
    utility: float
    feasible: bool
    rejection_reason: Optional[str] = None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _uncertainty(row: Dict[str, Any]) -> float:
    """Estimate normalized belief uncertainty from the existing model outputs.

    Confidence is explicitly uncalibrated in the UI, so it is treated as a
    planning signal rather than a probability of correctness.
    """

    confidence = max(0.0, min(1.0, _number(row.get("confidence"))))
    sparse = row.get("scoring_mode") == "sparse_vegetation_fallback"
    uncertainty = 1.0 - confidence
    return min(1.0, uncertainty + (0.15 if sparse else 0.0))


def _candidate(row: Dict[str, Any], resources: MissionResources) -> ObservationCandidate:
    risk = max(0.0, min(1.0, _number(row.get("drought_index")) / 100.0))
    uncertainty = _uncertainty(row)
    cloud = max(0.0, min(1.0, 0.18 + 0.55 * (1.0 - _number(row.get("confidence")))))
    # Value is highest where risk is material and the current state is uncertain.
    information_gain = risk * uncertainty * (1.0 - cloud)
    lat = _number(row.get("lat"))
    lon = _number(row.get("lon"))
    access_window = max(5, int(round(42.0 - min(32.0, abs(lat) * 0.12))))
    energy = resources.observation_energy_wh + resources.compute_energy_wh
    feasible = True
    reason: Optional[str] = None
    if resources.battery_wh - energy < resources.minimum_battery_wh:
        feasible = False
        reason = "battery reserve"
    elif resources.storage_mb + resources.product_size_mb > resources.storage_capacity_mb:
        feasible = False
        reason = "storage capacity"
    utility = information_gain / max(energy, 1e-9) if feasible else 0.0
    return ObservationCandidate(
        region=str(row.get("region") or "Unknown region"),
        lat=lat,
        lon=lon,
        access_window_minutes=access_window,
        cloud_probability=cloud,
        energy_wh=energy,
        data_mb=resources.product_size_mb,
        information_gain=information_gain,
        utility=utility,
        feasible=feasible,
        rejection_reason=reason,
    )


def plan_mission(
    rows: Iterable[Dict[str, Any]],
    resources: MissionResources | None = None,
    requested_region: str | None = None,
) -> Dict[str, Any]:
    """Select the next observation and return its auditable decision record."""

    resources = resources or MissionResources()
    row_list = list(rows)
    if requested_region:
        query = requested_region.casefold()
        row_list = [
            row for row in row_list
            if query in str(row.get("region", "")).casefold()
            or query in str(row.get("country", "")).casefold()
        ]
    candidates = [_candidate(row, resources) for row in row_list]
    ranked = sorted(candidates, key=lambda item: item.utility, reverse=True)
    selected = next((item for item in ranked if item.feasible), None)
    updated = asdict(resources)
    if selected:
        updated["battery_wh"] = round(resources.battery_wh - selected.energy_wh, 4)
        updated["storage_mb"] = round(resources.storage_mb + selected.data_mb, 4)
    return {
        "schema_version": "aurora-mission-plan.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "science_state": "deployed drought forecast rows",
        "selection": asdict(selected) if selected else None,
        "resources_before": asdict(resources),
        "resources_after": updated,
        "candidates": [asdict(candidate) for candidate in ranked],
        "replan_after_observation": bool(selected),
        "limitations": [
            "Access windows are planning estimates, not TLE/SGP4 ephemerides.",
            "Cloud probability is a forecast heuristic derived from available confidence.",
            "The observation update is a mission simulation; it does not create new satellite data.",
        ],
    }


def update_belief_after_observation(
    row: Dict[str, Any],
    observed_drought_index: float,
    prior_uncertainty: float | None = None,
) -> Dict[str, float]:
    """Perform one explicit belief update so the next plan can be recomputed."""

    prior = _uncertainty(row) if prior_uncertainty is None else max(0.0, min(1.0, prior_uncertainty))
    observed = max(0.0, min(100.0, observed_drought_index))
    prior_risk = max(0.0, min(100.0, _number(row.get("drought_index"))))
    innovation = abs(observed - prior_risk) / 100.0
    posterior_uncertainty = max(0.05, prior * (0.55 + 0.25 * innovation))
    posterior_risk = prior_risk + 0.35 * (observed - prior_risk)
    return {
        "posterior_drought_index": round(posterior_risk, 4),
        "posterior_uncertainty": round(posterior_uncertainty, 4),
        "innovation": round(innovation, 4),
    }
