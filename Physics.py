
#!/usr/bin/env python3
"""
AURORA V1-V3
Physics foundation for an autonomous Earth-observation satellite simulator.

V1: two-body orbital dynamics + RK4
V2: Earth rotation, geodetic ground track, Sun geometry, eclipse,
    ground-station access
V3: attitude-independent power budget, battery, thermal, storage, downlink,
    simple observation opportunities and validation

This is intentionally a foundation, not the research planner.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Config:
    # Earth / Sun
    MU_EARTH: float = 3.986004418e14
    R_EARTH: float = 6378.137e3
    OMEGA_EARTH: float = 7.2921150e-5
    SOLAR_CONSTANT: float = 1361.0
    AU: float = 149597870700.0
    SIGMA: float = 5.670374419e-8

    # Orbit
    altitude: float = 500e3
    eccentricity: float = 0.001
    inclination_deg: float = 97.4
    raan_deg: float = 0.0
    argument_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    # Simulation
    simulation_days: float = 1.0
    dt: float = 10.0
    output_directory: str = "aurora_results"

    # Electrical
    solar_area_m2: float = 0.040
    solar_efficiency: float = 0.29
    solar_degradation: float = 0.05
    panel_absorptivity: float = 0.85

    battery_capacity_Wh: float = 55.0
    initial_soc: float = 0.75
    minimum_soc: float = 0.15
    maximum_soc: float = 1.0
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.94

    base_power_W: float = 4.0
    payload_power_W: float = 8.0
    compute_power_W: float = 3.0
    downlink_power_W: float = 18.0

    # Data system
    image_size_MB: float = 8.0
    image_cooldown_s: float = 600.0
    storage_capacity_MB: float = 512.0
    downlink_rate_Mbps: float = 2.0

    # Ground station
    ground_station_lat_deg: float = 13.0827
    ground_station_lon_deg: float = 77.5877
    minimum_elevation_deg: float = 10.0

    # Thermal
    thermal_capacity_J_K: float = 9000.0
    initial_temperature_K: float = 293.15
    radiator_area_m2: float = 0.060
    radiator_emissivity: float = 0.82
    radiator_absorptivity: float = 0.15
    radiator_to_internal_fraction: float = 0.75
    earth_IR_W_m2: float = 237.0
    earth_albedo: float = 0.30

    # Validation
    energy_relative_tolerance: float = 1e-5
    angular_momentum_relative_tolerance: float = 1e-5


C = Config()


# ----------------------------- math helpers -----------------------------

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 0:
        raise ValueError("Cannot normalize a zero vector.")
    return np.asarray(v, dtype=float) / n


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# --------------------------- orbital mechanics --------------------------

def orbital_elements_to_state():
    a = C.R_EARTH + C.altitude
    e = C.eccentricity
    inc = math.radians(C.inclination_deg)
    raan = math.radians(C.raan_deg)
    argp = math.radians(C.argument_perigee_deg)
    nu = math.radians(C.true_anomaly_deg)

    p = a * (1 - e * e)
    rmag = p / (1 + e * math.cos(nu))

    r_pqw = np.array([rmag * math.cos(nu), rmag * math.sin(nu), 0.0])
    v_pqw = math.sqrt(C.MU_EARTH / p) * np.array(
        [-math.sin(nu), e + math.cos(nu), 0.0]
    )

    Q = rot_z(raan) @ rot_x(inc) @ rot_z(argp)
    return Q @ r_pqw, Q @ v_pqw


def acceleration(r):
    return -C.MU_EARTH * r / norm(r) ** 3


def derivative(state):
    return np.concatenate((state[3:], acceleration(state[:3])))


def rk4_step(r, v, dt):
    y = np.concatenate((r, v))
    k1 = derivative(y)
    k2 = derivative(y + 0.5 * dt * k1)
    k3 = derivative(y + 0.5 * dt * k2)
    k4 = derivative(y + dt * k3)
    y2 = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y2[:3], y2[3:]


def specific_energy(r, v):
    return 0.5 * np.dot(v, v) - C.MU_EARTH / norm(r)


def specific_angular_momentum(r, v):
    return np.cross(r, v)


def semi_major_axis():
    return C.R_EARTH + C.altitude


def orbital_period():
    return 2 * math.pi * math.sqrt(semi_major_axis() ** 3 / C.MU_EARTH)


# ----------------------------- Earth geometry ----------------------------

def eci_to_ecef(r_eci, t):
    return rot_z(-C.OMEGA_EARTH * t) @ r_eci


def ecef_to_lat_lon(r):
    x, y, z = r
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.hypot(x, y))
    return math.degrees(lat), math.degrees(lon)


def geodetic_to_ecef(lat_deg, lon_deg):
    # Spherical Earth is deliberately used consistently with the ground-track model.
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return C.R_EARTH * np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ])


def ground_station_position():
    return geodetic_to_ecef(C.ground_station_lat_deg, C.ground_station_lon_deg)


# ------------------------------- Sun model -------------------------------

def sun_direction_eci(t):
    """Low-order solar direction, adequate for V1-V3 power/eclipse studies."""
    d = t / 86400.0

    L = math.radians((280.460 + 0.9856474 * d) % 360.0)
    g = math.radians((357.528 + 0.9856003 * d) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2*g)
    eps = math.radians(23.439 - 0.0000004 * d)

    return unit(np.array([
        math.cos(lam),
        math.cos(eps) * math.sin(lam),
        math.sin(eps) * math.sin(lam),
    ]))


# ------------------------------- Eclipse --------------------------------

def eclipse_fraction(r_sc, sun_hat):
    """
    Finite-Sun-radius cylindrical/umbra approximation.

    Returns 0 sunlight, 1 full umbra.
    Penumbra is not yet resolved; V4+ can replace this with a conical model.
    """
    projection = float(np.dot(r_sc, sun_hat))
    if projection >= 0:
        return 0.0

    perpendicular = r_sc - projection * sun_hat
    return 1.0 if norm(perpendicular) <= C.R_EARTH else 0.0


# ------------------------- Ground station access --------------------------

def satellite_elevation(sat_ecef, station_ecef):
    los = sat_ecef - station_ecef
    return math.degrees(math.asin(np.clip(np.dot(unit(los), unit(station_ecef)), -1, 1)))


# ------------------------------- Battery ---------------------------------

class Battery:
    def __init__(self):
        self.energy_Wh = C.battery_capacity_Wh * C.initial_soc

    @property
    def soc(self):
        return self.energy_Wh / C.battery_capacity_Wh

    def update(self, net_power_W, dt):
        if net_power_W >= 0:
            delta_Wh = net_power_W * dt / 3600.0 * C.charge_efficiency
        else:
            delta_Wh = net_power_W * dt / 3600.0 / C.discharge_efficiency

        self.energy_Wh += delta_Wh
        lo = C.battery_capacity_Wh * C.minimum_soc
        hi = C.battery_capacity_Wh * C.maximum_soc
        self.energy_Wh = float(np.clip(self.energy_Wh, lo, hi))


# ------------------------------- Power -----------------------------------

def solar_power(r_sc, t, panel_normal_eci):
    sun = sun_direction_eci(t)
    eclipse = eclipse_fraction(r_sc, sun)
    incidence = max(0.0, float(np.dot(unit(panel_normal_eci), sun)))

    p = (
        C.SOLAR_CONSTANT
        * C.solar_area_m2
        * C.solar_efficiency
        * (1 - C.solar_degradation)
        * incidence
        * (1 - eclipse)
    )
    return p, eclipse, incidence


# ------------------------------- Thermal ---------------------------------

def thermal_step(T, internal_power_W, r_sc, t, eclipse, dt):
    """
    Lumped-node thermal model.

    Internal electrical dissipation is treated as heat.
    Solar/albedo/IR heating uses the radiator's nadir-facing surface.
    """
    nadir_hat = -unit(r_sc)
    sun_hat = sun_direction_eci(t)

    solar_absorbed = (
        C.SOLAR_CONSTANT
        * C.radiator_area_m2
        * C.radiator_absorptivity
        * max(0.0, float(np.dot(nadir_hat, sun_hat)))
        * (1 - eclipse)
    )

    # Earth IR is visible to a nadir-facing radiator. A simple sphere-view factor
    # approximation is used rather than pretending the full hemisphere is Earth.
    earth_view_factor = 0.5 * (
        1.0 - math.sqrt(max(0.0, 1.0 - (C.R_EARTH / norm(r_sc)) ** 2))
    )
    earth_ir_absorbed = (
        C.earth_IR_W_m2
        * C.radiator_area_m2
        * C.radiator_absorptivity
        * earth_view_factor
    )

    # Very conservative first-order albedo approximation.
    albedo_absorbed = (
        C.SOLAR_CONSTANT
        * C.earth_albedo
        * C.radiator_area_m2
        * C.radiator_absorptivity
        * max(0.0, float(np.dot(nadir_hat, sun_hat)))
        * (1 - eclipse)
        * earth_view_factor
    )

    internal_heat = internal_power_W * C.radiator_to_internal_fraction

    radiated = (
        C.radiator_emissivity
        * C.SIGMA
        * C.radiator_area_m2
        * T**4
    )

    net = internal_heat + solar_absorbed + earth_ir_absorbed + albedo_absorbed - radiated
    return T + net * dt / C.thermal_capacity_J_K


# ------------------------------- Targets ---------------------------------

@dataclass
class Target:
    name: str
    lat: float
    lon: float
    weight: float


TARGETS = [
    Target("Karnataka", 15.3173, 75.7139, 1.00),
    Target("Rajasthan", 27.0238, 74.2179, 0.95),
    Target("Bihar", 25.0961, 85.3131, 0.90),
    Target("Tamil_Nadu", 11.1271, 78.6569, 0.90),
]


def target_distance_deg(lat, lon, target):
    dlat = lat - target.lat
    dlon = (lon - target.lon + 180) % 360 - 180
    return math.hypot(dlat, dlon)


# --------------------------- Mission simulation --------------------------

def run_simulation():
    r, v = orbital_elements_to_state()
    e0 = specific_energy(r, v)
    h0 = specific_angular_momentum(r, v)

    battery = Battery()
    T = C.initial_temperature_K
    storage = 0.0
    station = ground_station_position()

    observations = []
    records = []
    last_image = -1e30

    steps = int(round(C.simulation_days * 86400.0 / C.dt))

    for k in range(steps + 1):
        t = k * C.dt

        if k > 0:
            r, v = rk4_step(r, v, C.dt)

        r_ecef = eci_to_ecef(r, t)
        lat, lon = ecef_to_lat_lon(r_ecef)
        altitude_km = (norm(r) - C.R_EARTH) / 1000.0

        sun = sun_direction_eci(t)

        # Foundation attitude: fixed inertial panel normal.
        # V4 introduces an explicit attitude/pointing state.
        panel_normal = np.array([1.0, 0.0, 0.0])
        p_solar, eclipse, incidence = solar_power(r, t, panel_normal)

        elevation = satellite_elevation(r_ecef, station)
        visible = elevation >= C.minimum_elevation_deg

        imaging = False
        selected = ""

        if (
            t - last_image >= C.image_cooldown_s
            and battery.soc > 0.30
            and storage + C.image_size_MB <= C.storage_capacity_MB
        ):
            distances = [target_distance_deg(lat, lon, x) for x in TARGETS]
            idx = int(np.argmin(distances))

            if distances[idx] <= 5.0:
                imaging = True
                selected = TARGETS[idx].name
                last_image = t
                storage += C.image_size_MB
                observations.append((t, selected, C.image_size_MB))

        downlink = False
        downlinked = 0.0

        if visible and storage > 0 and battery.soc > 0.25:
            rate_MB_s = C.downlink_rate_Mbps / 8.0
            capacity = rate_MB_s * C.dt
            downlinked = min(storage, capacity)
            storage -= downlinked
            downlink = downlinked > 0

        load = C.base_power_W
        if imaging:
            load += C.payload_power_W + C.compute_power_W
        if downlink:
            load += C.downlink_power_W

        net_power = p_solar - load
        battery.update(net_power, C.dt)

        T = thermal_step(T, load, r, t, eclipse, C.dt)

        records.append({
            "time_s": t,
            "x_m": r[0], "y_m": r[1], "z_m": r[2],
            "vx_m_s": v[0], "vy_m_s": v[1], "vz_m_s": v[2],
            "altitude_km": altitude_km,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "sun_x": sun[0], "sun_y": sun[1], "sun_z": sun[2],
            "sun_incidence": incidence,
            "eclipse": eclipse,
            "solar_power_W": p_solar,
            "load_W": load,
            "net_power_W": net_power,
            "battery_Wh": battery.energy_Wh,
            "soc": battery.soc,
            "temperature_K": T,
            "storage_MB": storage,
            "ground_station_visible": int(visible),
            "elevation_deg": elevation,
            "imaging": int(imaging),
            "downlink_MB": downlinked,
        })

    ef = specific_energy(r, v)
    hf = specific_angular_momentum(r, v)

    return {
        "records": records,
        "observations": observations,
        "initial_energy": e0,
        "final_energy": ef,
        "energy_error": abs((ef - e0) / e0),
        "initial_h": h0,
        "final_h": hf,
        "angular_momentum_error": norm(hf - h0) / norm(h0),
        "final_battery_Wh": battery.energy_Wh,
        "final_soc": battery.soc,
        "final_temperature_K": T,
        "final_storage_MB": storage,
    }


# ------------------------------- Validation ------------------------------

def validate(results):
    rec = results["records"]
    alt = np.array([x["altitude_km"] for x in rec])
    soc = np.array([x["soc"] for x in rec])
    temp = np.array([x["temperature_K"] for x in rec])

    # Since nu=0 at initialization, radius is perigee.
    a = semi_major_axis()
    expected_r0 = a * (1 - C.eccentricity)
    actual_r0 = norm(orbital_elements_to_state()[0])

    # For an ellipse: r_a-r_p = 2ae. Convert metres to km.
    expected_alt_span_km = 2 * a * C.eccentricity / 1000.0

    checks = [
        (
            "Initial radius matches orbital elements",
            abs(actual_r0 - expected_r0) < 1e-3,
            abs(actual_r0 - expected_r0),
            "m",
        ),
        (
            "Specific orbital energy conservation",
            results["energy_error"] < C.energy_relative_tolerance,
            results["energy_error"],
            f"< {C.energy_relative_tolerance:g}",
        ),
        (
            "Specific angular momentum conservation",
            results["angular_momentum_error"] < C.angular_momentum_relative_tolerance,
            results["angular_momentum_error"],
            f"< {C.angular_momentum_relative_tolerance:g}",
        ),
        (
            "Altitude variation matches eccentricity",
            abs(np.ptp(alt) - expected_alt_span_km) < 0.05,
            np.ptp(alt),
            f"expected ≈ {expected_alt_span_km:.3f} km",
        ),
        (
            "Battery stays inside configured limits",
            soc.min() >= C.minimum_soc - 1e-12 and soc.max() <= C.maximum_soc + 1e-12,
            f"{soc.min():.4f} -> {soc.max():.4f}",
            f"{C.minimum_soc:.2f} -> {C.maximum_soc:.2f}",
        ),
        (
            "All numeric states finite",
            all(np.isfinite(v) for row in rec for v in row.values() if isinstance(v, (int, float))),
            "finite",
            "all finite",
        ),
        (
            "Temperature remains physical",
            np.all(temp > 100.0) and np.all(temp < 1000.0),
            f"{temp.min()-273.15:.2f} -> {temp.max()-273.15:.2f} °C",
            "100 K < T < 1000 K",
        ),
    ]
    return checks


# ------------------------------- Unit tests -------------------------------

def run_unit_tests():
    print("Running unit tests...")

    r, v = orbital_elements_to_state()
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))

    # Correct eccentric-orbit expectation: true anomaly 0 = perigee.
    a = semi_major_axis()
    assert abs(norm(r) - a * (1 - C.eccentricity)) < 1e-3

    r2, v2 = rk4_step(r, v, 1.0)
    assert np.all(np.isfinite(r2))
    assert np.all(np.isfinite(v2))

    e0 = specific_energy(r, v)
    e1 = specific_energy(r2, v2)
    assert abs((e1 - e0) / e0) < 1e-10

    h0 = specific_angular_momentum(r, v)
    h1 = specific_angular_momentum(r2, v2)
    assert norm(h1 - h0) / norm(h0) < 1e-10

    sun = sun_direction_eci(0.0)
    assert abs(norm(sun) - 1.0) < 1e-12

    station = ground_station_position()
    assert abs(norm(station) - C.R_EARTH) < 1e-6

    b = Battery()
    initial = b.energy_Wh
    b.update(10.0, 60.0)
    assert b.energy_Wh > initial
    b.update(-10.0, 60.0)
    assert b.energy_Wh < initial + 1.0

    p, eclipse, incidence = solar_power(r, 0.0, np.array([1.0, 0.0, 0.0]))
    assert p >= 0.0
    assert eclipse in (0.0, 1.0)
    assert 0.0 <= incidence <= 1.0

    # Eclipse geometry sanity.
    sun_hat = np.array([1.0, 0.0, 0.0])
    assert eclipse_fraction(np.array([-(C.R_EARTH + 500e3), 0, 0.0]), sun_hat) == 1.0
    assert eclipse_fraction(np.array([-(C.R_EARTH + 500e3), C.R_EARTH + 1.0, 0.0]), sun_hat) == 0.0

    print("UNIT TESTS: PASS")


# ------------------------------- Outputs ---------------------------------

def save_csv(records, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def plot(x, y, xlabel, ylabel, title, path, hline=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    if hline is not None:
        plt.axhline(hline, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_plots(results, out):
    r = results["records"]
    t = np.array([x["time_s"] for x in r]) / 3600.0

    plot(t, [x["altitude_km"] for x in r], "Time [h]", "Altitude [km]",
         "AURORA V1 — Orbital Altitude", out / "altitude.png")

    plt.figure(figsize=(10, 5))
    plt.plot([x["longitude_deg"] for x in r], [x["latitude_deg"] for x in r])
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("AURORA V2 — Ground Track")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "ground_track.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, [x["solar_power_W"] for x in r], label="Solar generation")
    plt.plot(t, [x["load_W"] for x in r], label="Spacecraft load")
    plt.xlabel("Time [h]")
    plt.ylabel("Power [W]")
    plt.title("AURORA V3 — Electrical Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "power.png", dpi=160)
    plt.close()

    plot(t, np.array([x["soc"] for x in r]) * 100,
         "Time [h]", "Battery SOC [%]", "AURORA V3 — Battery", out / "battery.png")

    plot(t, np.array([x["temperature_K"] for x in r]) - 273.15,
         "Time [h]", "Temperature [°C]", "AURORA V3 — Thermal State",
         out / "temperature.png")

    plot(t, [x["storage_MB"] for x in r],
         "Time [h]", "Stored data [MB]", "AURORA V3 — Onboard Storage",
         out / "storage.png")

    plot(t, [x["elevation_deg"] for x in r],
         "Time [h]", "Elevation [deg]", "AURORA V2 — Ground Station Access",
         out / "ground_station.png", hline=C.minimum_elevation_deg)

    plot(t, [x["eclipse"] for x in r],
         "Time [h]", "Umbra [0/1]", "AURORA V2 — Eclipse State",
         out / "eclipse.png")


def write_summary(results, checks, out):
    r = results["records"]
    min_soc = min(x["soc"] for x in r)
    min_T = min(x["temperature_K"] for x in r) - 273.15
    max_T = max(x["temperature_K"] for x in r) - 273.15
    generated = sum(x[2] for x in results["observations"])
    downlinked = sum(x["downlink_MB"] for x in r)

    lines = [
        "AURORA V1-V3 PHYSICS FOUNDATION",
        "=" * 72,
        "",
        "ORBIT",
        f"Semimajor axis: {semi_major_axis()/1000:.3f} km",
        f"Reference altitude: {C.altitude/1000:.3f} km",
        f"Eccentricity: {C.eccentricity:.6f}",
        f"Inclination: {C.inclination_deg:.3f} deg",
        f"Period: {orbital_period()/60:.3f} min",
        "",
        "MISSION",
        f"Duration: {C.simulation_days:.3f} days",
        f"Observations: {len(results['observations'])}",
        f"Data generated: {generated:.2f} MB",
        f"Data downlinked: {downlinked:.2f} MB",
        f"Final storage: {results['final_storage_MB']:.2f} MB",
        "",
        "POWER",
        f"Final battery: {results['final_battery_Wh']:.3f} Wh",
        f"Final SOC: {results['final_soc']*100:.2f}%",
        f"Minimum SOC: {min_soc*100:.2f}%",
        "",
        "THERMAL",
        f"Temperature range: {min_T:.2f} to {max_T:.2f} deg C",
        "",
        "ORBITAL NUMERICS",
        f"Relative energy error: {results['energy_error']:.6e}",
        f"Relative angular-momentum error: {results['angular_momentum_error']:.6e}",
        "",
        "VALIDATION",
        "-" * 72,
    ]
    for name, ok, value, limit in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"      value = {value}")
        lines.append(f"      limit = {limit}")

    lines += [
        "",
        "MODEL SCOPE",
        "-" * 72,
        "V1: two-body gravity + RK4.",
        "V2: Earth rotation, spherical ground track, solar geometry, umbra, station access.",
        "V3: solar power, battery, lumped thermal model, storage and downlink.",
        "",
        "NEXT RESEARCH VERSIONS",
        "V4: evolving spatiotemporal science field + cloud/weather uncertainty.",
        "V5: measurement model + Bayesian/state uncertainty.",
        "V6: baseline planners.",
        "V7+: dynamic information utility, coupled resource opportunity cost, Monte Carlo.",
    ]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 72)
    print("AURORA V1-V3 — PHYSICS FOUNDATION")
    print("=" * 72)

    out = Path(C.output_directory)
    out.mkdir(parents=True, exist_ok=True)

    run_unit_tests()

    print("\nRunning spacecraft simulation...")
    results = run_simulation()

    print("\nRunning physics validation...")
    checks = validate(results)

    for name, ok, value, limit in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"      value = {value}")
        print(f"      limit = {limit}")

    if not all(x[1] for x in checks):
        raise RuntimeError("Physics validation failed. Inspect the failed checks above.")

    save_csv(results["records"], out / "mission_results.csv")
    generate_plots(results, out)
    write_summary(results, checks, out)

    print("\n" + "=" * 72)
    print("MISSION RESULTS")
    print("=" * 72)
    print(f"Observations: {len(results['observations'])}")
    print(f"Final battery: {results['final_battery_Wh']:.3f} Wh")
    print(f"Final SOC: {results['final_soc']*100:.2f}%")
    print(f"Final temperature: {results['final_temperature_K']-273.15:.2f} °C")
    print(f"Final storage: {results['final_storage_MB']:.2f} MB")
    print(f"\nOutputs: {out.resolve()}")
    print("=" * 72)
    print("AURORA V1-V3 COMPLETE")


if __name__ == "__main__":
    main()
