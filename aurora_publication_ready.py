#!/usr/bin/env python3
"""AURORA: reproducible, scenario-based adaptive EO scheduling experiments.

This is a research simulator, not flight software and not a claim of mission
qualification.  It is deliberately separate from ``auroraV4-10.py`` so the
original results remain reproducible.  This version is designed to make a
defensible *algorithmic* paper possible by providing:

* independent campaign priorities (no priority is derived from latent truth),
* a forecast-only cloud input (the scheduler never reads current cloud truth),
* a genuinely random policy and stronger myopic/resource-aware baselines,
* matched stochastic environments and common random numbers per opportunity,
* multiple nominal and resource-stress scenarios,
* resource constraints that can reject decisions,
* all-seed ablations, paired bootstrap confidence intervals, run identifiers,
  calibration diagnostics, and machine-readable outputs.

Run a fast smoke experiment with ``py -3 aurora_publication_ready.py --quick``.
Run the full experiment with ``py -3 aurora_publication_ready.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Physics import (
    C as PHYSICS,
    eci_to_ecef,
    eclipse_fraction,
    ecef_to_lat_lon,
    geodetic_to_ecef,
    ground_station_position,
    orbital_elements_to_state,
    rk4_step,
    satellite_elevation,
    solar_power,
    sun_direction_eci,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters shared by every scenario.

    All values are explicitly written to ``experiment_config.json``.  A paper
    should report these values and retain the JSON with the raw outcomes.
    """

    master_seed: int = 20260910
    monte_carlo_runs: int = 24
    duration_days: float = 3.0
    decision_step_s: float = 600.0
    grid_lat: int = 12
    grid_lon: int = 24
    fov_off_nadir_deg: float = 34.0
    observation_noise: float = 0.045
    success_probability_clear: float = 0.985
    cloud_measurement_multiplier: float = 1.60
    cloud_forecast_skill: float = 0.67
    cloud_persistence: float = 0.80
    cloud_base_probability: float = 0.24
    process_noise: float = 0.0030
    process_variance: float = 0.00020
    min_variance: float = 1e-5
    max_variance: float = 0.50
    battery_capacity_wh: float = 55.0
    initial_soc: float = 0.80
    minimum_soc: float = 0.20
    storage_capacity_mb: float = 96.0
    image_data_mb: float = 8.0
    observation_energy_wh: float = 1.5
    compute_energy_wh: float = 0.35
    downlink_rate_mb_s: float = 0.10
    base_load_w: float = 7.0
    downlink_power_w: float = 18.0
    bootstrap_samples: int = 4000
    output_directory: str = "aurora_publication_results"


@dataclass(frozen=True)
class Scenario:
    """Pre-registered stress regimes; no policy-specific tuning is allowed."""

    name: str
    description: str
    battery_scale: float = 1.0
    storage_scale: float = 1.0
    energy_scale: float = 1.0
    data_scale: float = 1.0
    downlink_scale: float = 1.0
    cloud_probability_scale: float = 1.0


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("nominal", "Balanced power, storage and cloud regime."),
    Scenario(
        "battery_stress",
        "Reduced battery and higher imaging energy; power feasibility must matter.",
        battery_scale=0.55,
        energy_scale=1.45,
    ),
    Scenario(
        "storage_stress",
        "Small recorder, larger products and restricted downlink throughput.",
        storage_scale=0.30,
        data_scale=1.55,
        downlink_scale=0.35,
    ),
    Scenario(
        "cloud_stress",
        "Frequent, persistent cloud with imperfect forecast information.",
        cloud_probability_scale=1.75,
    ),
)


@dataclass(frozen=True)
class Target:
    target_id: int
    lat: float
    lon: float
    grid_i: int
    grid_j: int
    campaign_priority: float


@dataclass(frozen=True)
class Opportunity:
    step: int
    time_s: float
    target_id: int
    off_nadir_deg: float
    eclipse: float
    station_elevation_deg: float
    next_access_s: float
    energy_wh: float
    data_mb: float


@dataclass
class MissionState:
    battery_wh: float
    storage_mb: float
    attempts: int = 0
    successful_observations: int = 0
    failures: int = 0
    generated_mb: float = 0.0
    downlinked_mb: float = 0.0
    information_gain: float = 0.0
    energy_rejections: int = 0
    storage_rejections: int = 0
    no_action_decisions: int = 0


def stable_rng(*parts: int) -> np.random.Generator:
    """Stable RNG stream, independent of policy execution order.

    This prevents one policy's different action count from changing the future
    environment or measurement noise seen by another policy.
    """

    return np.random.default_rng(np.random.SeedSequence([int(x) & 0xFFFFFFFF for x in parts]))


def wrap_lon(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n == 0:
        raise ValueError("Cannot normalize a zero vector")
    return x / n


def grid_laplacian(x: np.ndarray) -> np.ndarray:
    """Periodic longitude, reflecting latitude boundary; no polar wraparound."""

    north = np.vstack((x[0:1], x[:-1]))
    south = np.vstack((x[1:], x[-1:]))
    east = np.roll(x, 1, axis=1)
    west = np.roll(x, -1, axis=1)
    return north + south + east + west - 4.0 * x


class SyntheticEnvironment:
    """Pre-generated latent field, clouds and imperfect exogenous forecasts.

    The planner sees ``cloud_forecast`` only.  It cannot see ``cloud_truth``
    until an image is executed, which avoids the original truth-information
    leak.  This remains a synthetic testbed; replacement with an archived
    geophysical product is a required external-validation step.
    """

    def __init__(self, cfg: ExperimentConfig, scenario: Scenario, seed: int):
        self.cfg = cfg
        self.scenario = scenario
        self.seed = seed
        self.steps = int(round(cfg.duration_days * 86400.0 / cfg.decision_step_s)) + 1
        self.lats = np.linspace(-55.0, 55.0, cfg.grid_lat)
        self.lons = np.linspace(-180.0, 180.0, cfg.grid_lon, endpoint=False)
        lat2, lon2 = np.meshgrid(self.lats, self.lons, indexing="ij")
        self.lat2 = lat2
        self.lon2 = lon2
        rng = stable_rng(cfg.master_seed, seed, 101)
        baseline = 0.24 + 0.055 * np.cos(np.radians(lat2))
        # Latent physical pattern is fixed independently of campaign priorities.
        source = (
            0.18 * np.exp(-((lat2 - 18.0) / 17.0) ** 2 - ((lon2 - 72.0) / 27.0) ** 2)
            + 0.13 * np.exp(-((lat2 + 21.0) / 20.0) ** 2 - ((lon2 - 130.0) / 32.0) ** 2)
            + 0.10 * np.exp(-((lat2 - 35.0) / 22.0) ** 2 - ((lon2 + 100.0) / 35.0) ** 2)
        )
        truth = np.empty((self.steps, cfg.grid_lat, cfg.grid_lon), dtype=float)
        cloud = np.empty_like(truth)
        forecast = np.empty_like(truth)
        truth[0] = np.clip(baseline + source + rng.normal(0.0, 0.018, baseline.shape), 0.0, 1.0)
        p_cloud = min(0.90, cfg.cloud_base_probability * scenario.cloud_probability_scale)
        cloud[0] = (rng.random(baseline.shape) < p_cloud).astype(float)
        forecast[0] = np.clip(
            p_cloud + rng.normal(0.0, 0.12, baseline.shape), 0.02, 0.98
        )
        for k in range(1, self.steps):
            transport = 0.025 * (np.roll(truth[k - 1], 1 if k % 2 else -1, axis=1) - truth[k - 1])
            diffusion = 0.040 * grid_laplacian(truth[k - 1])
            seasonal = 0.002 * math.sin(2 * math.pi * k / 144.0)
            truth[k] = np.clip(
                truth[k - 1] + transport + diffusion + 0.0026 * source + seasonal
                + rng.normal(0.0, cfg.process_noise, baseline.shape),
                0.0,
                1.0,
            )
            new_cloud = (rng.random(baseline.shape) < p_cloud).astype(float)
            retain = rng.random(baseline.shape) < cfg.cloud_persistence
            cloud[k] = np.where(retain, cloud[k - 1], new_cloud)
            # Exogenous nowcast based on the previous cloud map; intentionally imperfect.
            forecast[k] = np.clip(
                (1.0 - cfg.cloud_forecast_skill) * p_cloud
                + cfg.cloud_forecast_skill * cloud[k - 1]
                + rng.normal(0.0, 0.16, baseline.shape),
                0.02,
                0.98,
            )
        self.truth = truth
        self.cloud_truth = cloud
        self.cloud_forecast = forecast
        self.prior_mean = baseline


class BeliefState:
    """Diagonal Gaussian estimator with deliberately imperfect dynamics."""

    def __init__(self, env: SyntheticEnvironment):
        self.cfg = env.cfg
        self.mean = env.prior_mean.copy()
        self.variance = np.full_like(self.mean, 0.10, dtype=float)
        self.age = np.zeros_like(self.mean, dtype=float)

    def predict(self, step: int) -> None:
        transport = 0.018 * (np.roll(self.mean, -1 if step % 2 else 1, axis=1) - self.mean)
        diffusion = 0.026 * grid_laplacian(self.mean)
        self.mean = np.clip(self.mean + transport + diffusion, 0.0, 1.0)
        self.variance = np.clip(
            self.variance + self.cfg.process_variance,
            self.cfg.min_variance,
            self.cfg.max_variance,
        )
        self.age += 1.0

    def expected_reduction(self, i: int, j: int, cloud_probability: float) -> float:
        p = float(np.clip(cloud_probability, 0.0, 1.0))
        r = self.cfg.observation_noise ** 2 * (1.0 + self.cfg.cloud_measurement_multiplier * p) ** 2
        v = float(self.variance[i, j])
        return v * v / (v + r)

    def observe(self, i: int, j: int, measurement: float, measurement_variance: float) -> float:
        prior = float(self.variance[i, j])
        r = max(float(measurement_variance), 1e-9)
        gain = prior * prior / (prior + r)
        k = prior / (prior + r)
        self.mean[i, j] = np.clip(self.mean[i, j] + k * (measurement - self.mean[i, j]), 0.0, 1.0)
        self.variance[i, j] = max(self.cfg.min_variance, (1.0 - k) * prior)
        self.age[i, j] = 0.0
        return gain


def build_targets(cfg: ExperimentConfig) -> List[Target]:
    """Every grid cell is a possible campaign; campaign value is exogenous."""

    rng = stable_rng(cfg.master_seed, 202)
    lats = np.linspace(-55.0, 55.0, cfg.grid_lat)
    lons = np.linspace(-180.0, 180.0, cfg.grid_lon, endpoint=False)
    targets: List[Target] = []
    tid = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Campaign priorities are sampled before any science realization.
            priority = float(rng.uniform(0.25, 1.0))
            targets.append(Target(tid, float(lat), float(lon), i, j, priority))
            tid += 1
    return targets


def make_orbit_states(cfg: ExperimentConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    r, v = orbital_elements_to_state()
    states = [(r.copy(), v.copy())]
    for _ in range(1, int(round(cfg.duration_days * 86400.0 / cfg.decision_step_s)) + 1):
        r, v = rk4_step(r, v, cfg.decision_step_s)
        states.append((r.copy(), v.copy()))
    return states


def opportunity_geometry(
    cfg: ExperimentConfig, scenario: Scenario, targets: Sequence[Target]
) -> Tuple[Dict[int, List[Opportunity]], List[Tuple[float, float, float]]]:
    """Build target windows from line-of-sight geometry, once per experiment."""

    states = make_orbit_states(cfg)
    target_ecef = {t.target_id: geodetic_to_ecef(t.lat, t.lon) for t in targets}
    raw: Dict[int, List[Opportunity]] = defaultdict(list)
    common: List[Tuple[float, float, float]] = []  # eclipse, elevation, solar W
    station = ground_station_position()
    for step, (r, _) in enumerate(states):
        time_s = step * cfg.decision_step_s
        ecef = eci_to_ecef(r, time_s)
        eclipse = float(eclipse_fraction(r, sun_direction_eci(time_s)))
        elevation = float(satellite_elevation(ecef, station))
        # A sun-tracking array is an explicit modelling assumption here.  The
        # power model remains replaceable with a mission-specific attitude plan.
        p_solar, _, _ = solar_power(r, time_s, sun_direction_eci(time_s))
        common.append((eclipse, elevation, float(p_solar)))
        earth_rotation = np.array(
            [
                [math.cos(PHYSICS.OMEGA_EARTH * time_s), -math.sin(PHYSICS.OMEGA_EARTH * time_s), 0.0],
                [math.sin(PHYSICS.OMEGA_EARTH * time_s), math.cos(PHYSICS.OMEGA_EARTH * time_s), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        for target in targets:
            target_eci = earth_rotation @ target_ecef[target.target_id]
            los = target_eci - r
            # Target must be above the local horizon and within sensor look limit.
            if float(np.dot(los, target_eci)) <= 0.0:
                continue
            off_nadir = math.degrees(math.acos(np.clip(float(np.dot(unit(los), -unit(r))), -1.0, 1.0)))
            if off_nadir > cfg.fov_off_nadir_deg:
                continue
            energy = cfg.observation_energy_wh * scenario.energy_scale * (1.0 + 0.010 * off_nadir + 0.28 * eclipse)
            data = cfg.image_data_mb * scenario.data_scale * (1.0 + 0.006 * off_nadir)
            raw[target.target_id].append(
                Opportunity(step, time_s, target.target_id, off_nadir, eclipse, elevation, math.inf, energy, data)
            )
    # Attach next target-specific access for future-observability scoring.
    by_step: Dict[int, List[Opportunity]] = defaultdict(list)
    for tid, ops in raw.items():
        for idx, op in enumerate(ops):
            next_t = ops[idx + 1].time_s if idx + 1 < len(ops) else math.inf
            by_step[op.step].append(replace(op, next_access_s=next_t))
    return by_step, common


class Planner:
    name = "base"

    def choose(
        self,
        candidates: Sequence[Opportunity],
        targets: Dict[int, Target],
        belief: BeliefState,
        cloud_forecast: np.ndarray,
        state: MissionState,
        cfg: ExperimentConfig,
        scenario: Scenario,
        random_seed: int,
    ) -> Opportunity | None:
        raise NotImplementedError

    @staticmethod
    def feasible(candidates: Sequence[Opportunity], state: MissionState, cfg: ExperimentConfig, scenario: Scenario) -> List[Opportunity]:
        cap = cfg.storage_capacity_mb * scenario.storage_scale
        reserve = cfg.battery_capacity_wh * scenario.battery_scale * cfg.minimum_soc
        result = []
        for op in candidates:
            if state.battery_wh - op.energy_wh < reserve:
                state.energy_rejections += 1
                continue
            if state.storage_mb + op.data_mb > cap:
                state.storage_rejections += 1
                continue
            result.append(op)
        return result


class UniformRandomPlanner(Planner):
    name = "uniform_random"

    def choose(self, candidates, targets, belief, cloud_forecast, state, cfg, scenario, random_seed):
        feasible = self.feasible(candidates, state, cfg, scenario)
        if not feasible:
            return None
        # A separate stream makes this a true random selection policy.
        return feasible[int(stable_rng(cfg.master_seed, random_seed, 301, candidates[0].step).integers(len(feasible)))]


class StaticPriorityPlanner(Planner):
    name = "static_priority"

    def choose(self, candidates, targets, belief, cloud_forecast, state, cfg, scenario, random_seed):
        feasible = self.feasible(candidates, state, cfg, scenario)
        return max(feasible, key=lambda op: targets[op.target_id].campaign_priority, default=None)


class MyopicInformationPlanner(Planner):
    name = "myopic_information"

    def choose(self, candidates, targets, belief, cloud_forecast, state, cfg, scenario, random_seed):
        feasible = self.feasible(candidates, state, cfg, scenario)
        return max(
            feasible,
            key=lambda op: belief.expected_reduction(
                targets[op.target_id].grid_i,
                targets[op.target_id].grid_j,
                cloud_forecast[targets[op.target_id].grid_i, targets[op.target_id].grid_j],
            ),
            default=None,
        )


class ResourceAwareMyopicPlanner(Planner):
    name = "resource_aware_myopic"

    def choose(self, candidates, targets, belief, cloud_forecast, state, cfg, scenario, random_seed):
        feasible = self.feasible(candidates, state, cfg, scenario)
        cap = cfg.storage_capacity_mb * scenario.storage_scale
        reserve = cfg.battery_capacity_wh * scenario.battery_scale * cfg.minimum_soc
        def score(op: Opportunity) -> float:
            target = targets[op.target_id]
            gain = belief.expected_reduction(target.grid_i, target.grid_j, cloud_forecast[target.grid_i, target.grid_j])
            cost = op.energy_wh / max(state.battery_wh - reserve, 1e-6) + op.data_mb / max(cap - state.storage_mb, 1e-6)
            return gain * target.campaign_priority / (1.0 + cost)
        return max(feasible, key=score, default=None)


class DynamicInformationUtilityPlanner(Planner):
    """Proposed policy with individually switchable terms for all-seed ablation."""

    name = "dynamic_information_utility"

    def __init__(self, aging=True, future_access=True, forecast=True, resources=True):
        self.aging = aging
        self.future_access = future_access
        self.forecast = forecast
        self.resources = resources

    def choose(self, candidates, targets, belief, cloud_forecast, state, cfg, scenario, random_seed):
        feasible = self.feasible(candidates, state, cfg, scenario)
        cap = cfg.storage_capacity_mb * scenario.storage_scale
        reserve = cfg.battery_capacity_wh * scenario.battery_scale * cfg.minimum_soc
        def score(op: Opportunity) -> float:
            target = targets[op.target_id]
            i, j = target.grid_i, target.grid_j
            cloud_p = float(cloud_forecast[i, j])
            gain = belief.expected_reduction(i, j, cloud_p)
            age_term = 1.0 + 0.020 * float(belief.age[i, j]) if self.aging else 1.0
            # Forecast relevance uses estimator uncertainty and forecast cloud, never cloud truth.
            forecast_term = 1.0 + 0.40 * min(1.0, float(belief.variance[i, j]) / 0.10) * (1.0 - cloud_p) if self.forecast else 1.0
            if self.future_access:
                gap_hours = 8.0 if not math.isfinite(op.next_access_s) else min(8.0, (op.next_access_s - op.time_s) / 3600.0)
                future_term = 1.0 + 0.12 * gap_hours
            else:
                future_term = 1.0
            resource_term = 1.0
            if self.resources:
                energy_pressure = op.energy_wh / max(state.battery_wh - reserve, 1e-6)
                storage_pressure = op.data_mb / max(cap - state.storage_mb, 1e-6)
                resource_term = 1.0 / (1.0 + 1.2 * energy_pressure + 0.8 * storage_pressure)
            return gain * target.campaign_priority * age_term * forecast_term * future_term * resource_term
        return max(feasible, key=score, default=None)


def resource_housekeeping(
    state: MissionState, common: Tuple[float, float, float], cfg: ExperimentConfig, scenario: Scenario
) -> None:
    """Advance power and downlink before a decision; retain traceable assumptions."""

    _, elevation, solar_w = common
    dt_h = cfg.decision_step_s / 3600.0
    capacity = cfg.battery_capacity_wh * scenario.battery_scale
    reserve = capacity * cfg.minimum_soc
    load = cfg.base_load_w
    if elevation >= PHYSICS.minimum_elevation_deg and state.storage_mb > 0:
        transferable = cfg.downlink_rate_mb_s * scenario.downlink_scale * cfg.decision_step_s
        down = min(state.storage_mb, transferable)
        down_energy = cfg.downlink_power_w * dt_h
        if state.battery_wh - down_energy >= reserve:
            state.storage_mb -= down
            state.downlinked_mb += down
            state.battery_wh -= down_energy
            load += cfg.downlink_power_w
    net = (solar_w - load) * dt_h
    if net >= 0:
        net *= PHYSICS.charge_efficiency
    else:
        net /= PHYSICS.discharge_efficiency
    state.battery_wh = float(np.clip(state.battery_wh + net, reserve, capacity))


def event_uniform(master_seed: int, run_seed: int, step: int, target_id: int, stream: int) -> float:
    return float(stable_rng(master_seed, run_seed, step, target_id, stream).random())


def simulate(
    planner: Planner,
    cfg: ExperimentConfig,
    scenario: Scenario,
    run_id: int,
    seed: int,
    targets: Sequence[Target],
    opportunities: Dict[int, List[Opportunity]],
    common_geometry: Sequence[Tuple[float, float, float]],
) -> Tuple[Dict[str, float | int | str], List[Dict[str, float | int | str]]]:
    env = SyntheticEnvironment(cfg, scenario, seed)
    belief = BeliefState(env)
    target_map = {x.target_id: x for x in targets}
    capacity = cfg.battery_capacity_wh * scenario.battery_scale
    state = MissionState(battery_wh=capacity * cfg.initial_soc, storage_mb=0.0)
    selection_log: List[Dict[str, float | int | str]] = []
    for step in range(env.steps):
        if step > 0:
            belief.predict(step)
        resource_housekeeping(state, common_geometry[step], cfg, scenario)
        op = planner.choose(
            opportunities.get(step, []), target_map, belief, env.cloud_forecast[step], state,
            cfg, scenario, seed,
        )
        if op is None:
            state.no_action_decisions += 1
            continue
        state.attempts += 1
        target = target_map[op.target_id]
        i, j = target.grid_i, target.grid_j
        cloud_truth = float(env.cloud_truth[step, i, j])
        cloud_forecast = float(env.cloud_forecast[step, i, j])
        state.battery_wh -= op.energy_wh
        success_probability = max(0.50, cfg.success_probability_clear - 0.16 * cloud_truth)
        success = event_uniform(cfg.master_seed, seed, step, op.target_id, 401) < success_probability
        gain = 0.0
        if success:
            measurement_std = cfg.observation_noise * (1.0 + cfg.cloud_measurement_multiplier * cloud_truth)
            noise = float(stable_rng(cfg.master_seed, seed, step, op.target_id, 402).normal(0.0, measurement_std))
            measurement = float(np.clip(env.truth[step, i, j] + noise, 0.0, 1.0))
            gain = belief.observe(i, j, measurement, measurement_std ** 2)
            state.successful_observations += 1
            state.information_gain += gain
            state.generated_mb += op.data_mb
            state.storage_mb += op.data_mb
        else:
            state.failures += 1
        selection_log.append({
            "scenario": scenario.name, "run_id": run_id, "seed": seed, "planner": planner.name,
            "step": step, "time_s": op.time_s, "target_id": op.target_id,
            "campaign_priority": target.campaign_priority, "off_nadir_deg": op.off_nadir_deg,
            "cloud_forecast": cloud_forecast, "cloud_truth": cloud_truth, "success": int(success),
            "information_gain": gain, "battery_wh_after": state.battery_wh, "storage_mb_after": state.storage_mb,
        })
    err = belief.mean - env.truth[-1]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mean_variance = float(np.mean(belief.variance))
    normalized_sq_error = float(np.mean(err ** 2 / np.maximum(belief.variance, cfg.min_variance)))
    coverage95 = float(np.mean(np.abs(err) <= 1.96 * np.sqrt(belief.variance)))
    row: Dict[str, float | int | str] = {
        "scenario": scenario.name, "run_id": run_id, "seed": seed, "planner": planner.name,
        "rmse": rmse, "mae": mae, "mean_variance": mean_variance,
        "normalized_squared_error": normalized_sq_error, "coverage95": coverage95,
        "information_gain": state.information_gain, "attempts": state.attempts,
        "successful_observations": state.successful_observations, "failures": state.failures,
        "generated_mb": state.generated_mb, "downlinked_mb": state.downlinked_mb,
        "final_storage_mb": state.storage_mb, "final_battery_wh": state.battery_wh,
        "energy_rejections": state.energy_rejections, "storage_rejections": state.storage_rejections,
        "no_action_decisions": state.no_action_decisions,
    }
    return row, selection_log


def mean_ci(values: Sequence[float], rng: np.random.Generator, samples: int) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean, mean
    idx = rng.integers(0, len(x), size=(samples, len(x)))
    boot = np.mean(x[idx], axis=1)
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize(rows: Sequence[Dict[str, float | int | str]], cfg: ExperimentConfig) -> List[Dict[str, float | int | str]]:
    groups: Dict[Tuple[str, str], List[Dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["planner"]))].append(row)
    metrics = [
        "rmse", "mae", "mean_variance", "normalized_squared_error", "coverage95",
        "information_gain", "successful_observations", "attempts", "failures", "generated_mb",
        "downlinked_mb", "final_storage_mb", "final_battery_wh", "energy_rejections",
        "storage_rejections", "no_action_decisions",
    ]
    summary = []
    for number, ((scenario, planner), group) in enumerate(sorted(groups.items())):
        out: Dict[str, float | int | str] = {"scenario": scenario, "planner": planner, "n": len(group)}
        for metric in metrics:
            values = [float(x[metric]) for x in group]
            mean, lo, hi = mean_ci(values, stable_rng(cfg.master_seed, 501, number, len(metric)), cfg.bootstrap_samples)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci95_low"] = lo
            out[f"{metric}_ci95_high"] = hi
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def paired_comparisons(
    rows: Sequence[Dict[str, float | int | str]], cfg: ExperimentConfig, proposed: str = "dynamic_information_utility"
) -> List[Dict[str, float | int | str]]:
    lookup = {(str(r["scenario"]), int(r["run_id"]), str(r["planner"])): r for r in rows}
    planners = sorted({str(r["planner"]) for r in rows if str(r["planner"]) != proposed})
    scenarios = sorted({str(r["scenario"]) for r in rows})
    out = []
    for scenario in scenarios:
        for baseline in planners:
            keys = sorted(k for k in lookup if k[0] == scenario and k[2] == proposed)
            for metric, lower_is_better in (("rmse", True), ("mae", True), ("information_gain", False)):
                diff = []
                for key in keys:
                    base_key = (key[0], key[1], baseline)
                    if base_key not in lookup:
                        continue
                    a = float(lookup[key][metric])
                    b = float(lookup[base_key][metric])
                    diff.append((b - a) if lower_is_better else (a - b))
                mean, lo, hi = mean_ci(diff, stable_rng(cfg.master_seed, 701, len(out)), cfg.bootstrap_samples)
                out.append({
                    "scenario": scenario, "proposed": proposed, "baseline": baseline, "metric": metric,
                    "direction": "positive favors proposed", "n_pairs": len(diff), "mean_difference": mean,
                    "ci95_low": lo, "ci95_high": hi,
                    "win_rate": float(np.mean(np.asarray(diff) > 0.0)),
                })
    return out


def write_csv(rows: Sequence[Dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary: Sequence[Dict[str, float | int | str]], output: Path) -> None:
    for scenario in sorted({str(x["scenario"]) for x in summary}):
        group = [x for x in summary if str(x["scenario"]) == scenario]
        for metric, label, filename in (
            ("rmse", "Final field RMSE (95% bootstrap CI)", "rmse"),
            ("information_gain", "Cumulative information gain (95% bootstrap CI)", "information_gain"),
            ("successful_observations", "Successful observations (95% bootstrap CI)", "observations"),
        ):
            names = [str(x["planner"]) for x in group]
            values = np.array([float(x[f"{metric}_mean"]) for x in group])
            low = values - np.array([float(x[f"{metric}_ci95_low"]) for x in group])
            high = np.array([float(x[f"{metric}_ci95_high"]) for x in group]) - values
            plt.figure(figsize=(10, 5))
            plt.bar(names, values, yerr=np.vstack((low, high)), capsize=4)
            plt.ylabel(label)
            plt.title(f"AURORA adaptive scheduling: {scenario}")
            plt.xticks(rotation=22, ha="right")
            plt.tight_layout()
            plt.savefig(output / f"{scenario}_{filename}.png", dpi=180)
            plt.close()


def validate_study(rows: Sequence[Dict[str, float | int | str]], scenarios: Sequence[Scenario], planners: Sequence[Planner]) -> List[str]:
    """Integrity checks, intentionally separate from claims of superiority."""

    problems: List[str] = []
    if len({str(x["scenario"]) for x in rows}) != len(scenarios):
        problems.append("Missing scenario outcomes")
    if len({str(x["planner"]) for x in rows}) != len(planners):
        problems.append("Missing planner outcomes")
    numeric = [v for row in rows for v in row.values() if isinstance(v, (float, int))]
    if not all(math.isfinite(float(v)) for v in numeric):
        problems.append("Non-finite numeric output")
    keys = [(x["scenario"], x["run_id"], x["planner"]) for x in rows]
    if len(keys) != len(set(keys)):
        problems.append("Duplicate scenario/run/planner outcome")
    resource_blocks = sum(int(x["energy_rejections"]) + int(x["storage_rejections"]) for x in rows)
    if resource_blocks == 0:
        problems.append("No resource constraints ever bound; stress design is ineffective")
    if not any(str(x["planner"]) == "uniform_random" for x in rows):
        problems.append("True random baseline absent")
    return problems


def write_report(
    output: Path,
    cfg: ExperimentConfig,
    scenarios: Sequence[Scenario],
    summary: Sequence[Dict[str, float | int | str]],
    comparisons: Sequence[Dict[str, float | int | str]],
    validation: Sequence[str],
) -> None:
    lines = [
        "# AURORA publication-oriented experiment report",
        "",
        "## Scope",
        "",
        "This is a synthetic, algorithmic evaluation. It does not establish flight readiness, operational performance, or publication acceptance. Claims must be restricted to the scenarios and assumptions recorded in experiment_config.json.",
        "",
        "## Pre-registered comparison design",
        "",
        f"- Monte-Carlo runs per scenario: {cfg.monte_carlo_runs}",
        f"- Duration per run: {cfg.duration_days} days", 
        f"- Decision interval: {cfg.decision_step_s} s",
        "- Matched environments: same scenario/run seed for every planner.",
        "- Common random numbers: success/noise stream is indexed by run, time, and target, not policy execution order.",
        "- Scheduler input: imperfect cloud forecast only; latent current cloud remains hidden until execution.",
        "- Primary metric: final field RMSE. Secondary metrics: MAE, information gain, uncertainty calibration, observation efficiency, and resource rejections.",
        "",
        "## Scenarios",
        "",
    ]
    lines += [f"- **{s.name}** — {s.description}" for s in scenarios]
    lines += ["", "## Per-scenario summary", ""]
    for row in summary:
        lines.append(
            f"- {row['scenario']} / {row['planner']}: RMSE {float(row['rmse_mean']):.6f} "
            f"[{float(row['rmse_ci95_low']):.6f}, {float(row['rmse_ci95_high']):.6f}], "
            f"information gain {float(row['information_gain_mean']):.4f}, "
            f"successful observations {float(row['successful_observations_mean']):.2f}, "
            f"energy/storage rejections {float(row['energy_rejections_mean']):.1f}/"
            f"{float(row['storage_rejections_mean']):.1f}."
        )
    lines += ["", "## Paired DIU comparisons", "", "Positive differences favor DIU. A CI excluding zero is evidence within this synthetic experiment, not universal superiority.", ""]
    for row in comparisons:
        lines.append(
            f"- {row['scenario']} / {row['metric']} vs {row['baseline']}: "
            f"{float(row['mean_difference']):.6f} [{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}], "
            f"win rate {float(row['win_rate']):.2%}, n={row['n_pairs']}."
        )
    lines += ["", "## Integrity checks", ""]
    lines += ["- PASS: " + x for x in (["All configured integrity checks passed."] if not validation else [])]
    lines += ["- FAIL: " + x for x in validation]
    lines += [
        "",
        "## Required external-validation path",
        "",
        "1. Replace synthetic truth and cloud forecasts with archived products and a time-respecting train/tune/test split.",
        "2. Validate the geometry, instrument, slew, power, thermal and downlink models against mission-specific sources.",
        "3. Compare against current optimization/MPC/RL baselines and report runtime, complexity, and parameter sensitivity.",
        "4. Perform a literature review before asserting novelty; resource-aware EO scheduling and value-based replanning are established areas.",
    ]
    (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(cfg: ExperimentConfig, quick: bool = False) -> Path:
    scenarios = SCENARIOS[:2] if quick else SCENARIOS
    runs = 4 if quick else cfg.monte_carlo_runs
    if quick:
        cfg = replace(cfg, monte_carlo_runs=runs, bootstrap_samples=600, output_directory="aurora_publication_smoke_results")
    output = Path(cfg.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    targets = build_targets(cfg)
    opportunities, common_geometry = opportunity_geometry(cfg, scenarios[0], targets)
    planners: List[Planner] = [
        UniformRandomPlanner(), StaticPriorityPlanner(), MyopicInformationPlanner(),
        ResourceAwareMyopicPlanner(), DynamicInformationUtilityPlanner(),
    ]
    all_rows: List[Dict[str, float | int | str]] = []
    all_logs: List[Dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(scenarios):
        # Geometry does not depend on the science realization; resource costs do.
        opportunities, common_geometry = opportunity_geometry(cfg, scenario, targets)
        for run_id in range(runs):
            seed = cfg.master_seed + 10000 * scenario_index + run_id
            for planner in planners:
                row, log = simulate(planner, cfg, scenario, run_id, seed, targets, opportunities, common_geometry)
                all_rows.append(row)
                all_logs.extend(log)
    # Every ablation is evaluated across every scenario and seed, never one cherry-picked run.
    ablation_rows: List[Dict[str, float | int | str]] = []
    ablations: List[Tuple[str, Planner]] = [
        ("diu_full", DynamicInformationUtilityPlanner()),
        ("diu_no_aging", DynamicInformationUtilityPlanner(aging=False)),
        ("diu_no_future_access", DynamicInformationUtilityPlanner(future_access=False)),
        ("diu_no_forecast", DynamicInformationUtilityPlanner(forecast=False)),
        ("diu_no_resources", DynamicInformationUtilityPlanner(resources=False)),
    ]
    for scenario_index, scenario in enumerate(scenarios):
        opportunities, common_geometry = opportunity_geometry(cfg, scenario, targets)
        for run_id in range(runs):
            seed = cfg.master_seed + 10000 * scenario_index + run_id
            for name, planner in ablations:
                planner.name = name
                row, _ = simulate(planner, cfg, scenario, run_id, seed, targets, opportunities, common_geometry)
                ablation_rows.append(row)
    summary = summarize(all_rows, cfg)
    comparisons = paired_comparisons(all_rows, cfg)
    ablation_summary = summarize(ablation_rows, cfg)
    validation = validate_study(all_rows, scenarios, planners)
    write_csv(all_rows, output / "raw_results.csv")
    write_csv(summary, output / "summary.csv")
    write_csv(comparisons, output / "paired_comparisons.csv")
    write_csv(ablation_rows, output / "ablation_raw_results.csv")
    write_csv(ablation_summary, output / "ablation_summary.csv")
    write_csv(all_logs, output / "decision_log.csv")
    (output / "experiment_config.json").write_text(
        json.dumps({"config": asdict(cfg), "scenarios": [asdict(s) for s in scenarios]}, indent=2), encoding="utf-8"
    )
    make_plots(summary, output)
    write_report(output, cfg, scenarios, summary, comparisons, validation)
    if validation:
        raise RuntimeError("Study integrity validation failed: " + "; ".join(validation))
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AURORA publication-oriented experiments")
    parser.add_argument("--quick", action="store_true", help="small smoke experiment; not a scientific result")
    args = parser.parse_args()
    out = run_experiment(ExperimentConfig(), quick=args.quick)
    print(f"AURORA experiment complete: {out}")


if __name__ == "__main__":
    main()
