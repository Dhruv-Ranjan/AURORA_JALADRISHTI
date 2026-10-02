#!/usr/bin/env python3
"""
Extended real place drought test.

Fetches actual Sentinel-2 imagery for a broader set of named regions and runs
the production spectral + drought pipeline on real image patches.
"""

import csv
import itertools
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from planetary_computer import sign
from pystac_client import Client

from drought_supervision import score_filtered_future_drought
from real_data_pipeline import DEVICE, load_drought_model, load_spectral_model


LOGGER = logging.getLogger(__name__)
PATCH_SIZE = 64
HALF = PATCH_SIZE // 2
SEQUENCE_LENGTH = 3
SEARCH_LIMIT = 20000
PATCHES_PER_REGION = 3
DATE_RANGE = "2023-01-01/2025-12-31"
MIN_GAP_DAYS = 5
MAX_GAP_DAYS = 240
PATCH_ATTEMPTS_PER_SEQUENCE = 10
MAX_CANDIDATES_PER_REGION = 6
OUTPUT_DIR = Path("real_region_test_images_more")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_JSON = Path("end_to_end_more_regions_results.json")
DIAGNOSTICS_JSON = Path("end_to_end_more_regions_diagnostics.json")
DIAGNOSTICS_CSV = Path("end_to_end_more_regions_candidates.csv")

REGIONS = [
    {"name": "Western Ghats", "bbox": [75.70, 11.15, 76.05, 11.85], "expected": "wetter", "preferred_months": [6, 7, 8, 9, 10, 11]},
    {"name": "Kerala", "bbox": [76.00, 9.50, 76.80, 10.80], "expected": "wetter", "preferred_months": [6, 7, 8, 9, 10, 11]},
    {"name": "Congo Basin", "bbox": [16.00, -1.00, 18.00, 1.00], "expected": "wetter", "preferred_months": [3, 4, 5, 9, 10, 11]},
    {"name": "Bangladesh Delta", "bbox": [90.0, 22.2, 90.8, 23.2], "expected": "moderate", "preferred_months": [6, 7, 8, 9, 10]},
    {"name": "Spain Drylands", "bbox": [-4.4, 39.0, -3.2, 39.9], "expected": "drier", "preferred_months": [6, 7, 8, 9]},
    {"name": "Rajasthan", "bbox": [70.80, 27.00, 71.55, 27.85], "expected": "drier", "preferred_months": [3, 4, 5, 6]},
    {"name": "Namibia North", "bbox": [15.00, -19.50, 17.00, -18.00], "expected": "drier", "preferred_months": [5, 6, 7, 8, 9]},
    {"name": "South Africa Free State", "bbox": [25.00, -29.80, 27.50, -28.00], "expected": "drier", "preferred_months": [5, 6, 7, 8, 9]},
    {"name": "Morocco Atlas", "bbox": [-8.1, 31.0, -4.0, 33.5], "expected": "drier", "preferred_months": [5, 6, 7, 8, 9]},
    {"name": "Turkey Anatolia", "bbox": [27.0, 36.5, 35.5, 42.0], "expected": "moderate", "preferred_months": [5, 6, 7, 8, 9]},
    {"name": "Sahel West", "bbox": [-6.0, 12.0, 12.0, 22.0], "expected": "drier", "preferred_months": [3, 4, 5, 6, 7, 8, 9]},
    {"name": "Ethiopia Highlands", "bbox": [36.0, 7.0, 42.0, 15.5], "expected": "drier", "preferred_months": [3, 4, 5, 6, 7, 8, 9]},
    {"name": "Kenya Rift", "bbox": [35.5, -1.5, 38.5, 2.5], "expected": "drier", "preferred_months": [5, 6, 7, 8, 9]},
    {"name": "Maharashtra", "bbox": [73.5, 17.0, 79.5, 22.5], "expected": "moderate", "preferred_months": [5, 6, 7, 8, 9, 10]},
    {"name": "Punjab Plains", "bbox": [73.5, 29.0, 77.0, 32.5], "expected": "drier", "preferred_months": [3, 4, 5, 6]},
    {"name": "Argentina Pampas", "bbox": [-65.0, -39.0, -58.0, -32.0], "expected": "moderate", "preferred_months": [4, 5, 6, 7, 8, 9, 10]},
    {"name": "Brazil Cerrado", "bbox": [-52.0, -18.0, -43.0, -10.0], "expected": "moderate", "preferred_months": [4, 5, 6, 7, 8, 9]},
    {"name": "Murray-Darling", "bbox": [141.0, -36.5, 151.0, -32.0], "expected": "drier", "preferred_months": [5, 6, 7, 8, 9]},
    {"name": "Mexico Sonora", "bbox": [-112.5, 27.0, -108.0, 32.0], "expected": "drier", "preferred_months": [3, 4, 5, 6, 7]},
    {"name": "California Central Valley", "bbox": [-122.5, 35.0, -118.0, 39.5], "expected": "drier", "preferred_months": [4, 5, 6, 7, 8, 9]},
    {"name": "Chile Central", "bbox": [-71.5, -34.0, -70.0, -31.0], "expected": "drier", "preferred_months": [5, 6, 7, 8, 9]},
]
MAX_REGIONS = int(os.environ.get("MAX_REGIONS", str(len(REGIONS))))
REGIONS = REGIONS[: min(MAX_REGIONS, len(REGIONS))]

CATALOG = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def fetch_items_safe(search, limit):
    last_error = None
    for attempt in range(1, 6):
        try:
            return list(itertools.islice(search.items(), limit))
        except Exception as exc:
            last_error = exc
            LOGGER.warning("STAC search attempt %d/5 failed: %s", attempt, exc)
            time.sleep(2)
    LOGGER.error("STAC search failed after 5 attempts: %s", last_error)
    return []


def get_patch(src, row, col, radius):
    return src.read(1, window=((row - radius, row + radius), (col - radius, col + radius)))


def upsample_x2(arr):
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def valid(*arrs):
    return all(a.shape == (PATCH_SIZE, PATCH_SIZE) and np.isfinite(a).all() for a in arrs)


def days_between(a, b):
    return abs((b.datetime - a.datetime).days)


def build_spaced_sequence(items, start_idx):
    seq = [items[start_idx]]
    last_idx = start_idx
    while len(seq) < SEQUENCE_LENGTH:
        found = False
        for next_idx in range(last_idx + 1, len(items)):
            gap = days_between(items[last_idx], items[next_idx])
            if gap < MIN_GAP_DAYS:
                continue
            if gap > MAX_GAP_DAYS:
                break
            seq.append(items[next_idx])
            last_idx = next_idx
            found = True
            break
        if not found:
            return None
    return seq


def choose_patch_coords(item, rng):
    signed = sign(item)
    with rasterio.open(signed.assets["B04"].href) as red:
        height, width = red.height, red.width
        if height < PATCH_SIZE or width < PATCH_SIZE:
            return None
        row = rng.randint(HALF, height - HALF)
        col = rng.randint(HALF, width - HALF)
        lon, lat = red.xy(row, col)
    return row, col, float(lat), float(lon)


def extract_rgb_and_spec(item, row, col):
    signed = sign(item)
    with rasterio.open(signed.assets["B02"].href) as b, \
         rasterio.open(signed.assets["B03"].href) as g, \
         rasterio.open(signed.assets["B04"].href) as r, \
         rasterio.open(signed.assets["B08"].href) as n, \
         rasterio.open(signed.assets["B11"].href) as s1, \
         rasterio.open(signed.assets["B12"].href) as s2:
        B = get_patch(b, row, col, HALF)
        G = get_patch(g, row, col, HALF)
        R = get_patch(r, row, col, HALF)
        N = get_patch(n, row, col, HALF)
        S1 = upsample_x2(get_patch(s1, row // 2, col // 2, HALF // 2))
        S2 = upsample_x2(get_patch(s2, row // 2, col // 2, HALF // 2))
        SCL = None
        if "SCL" in signed.assets:
            try:
                with rasterio.open(signed.assets["SCL"].href) as scl_src:
                    SCL = get_patch(scl_src, row, col, HALF)
            except Exception:
                SCL = None

    if not valid(B, G, R, N, S1, S2):
        return None
    if SCL is not None and not valid(SCL):
        SCL = None

    B, G, R, N, S1, S2 = [x.astype(np.float32) / 10000.0 for x in [B, G, R, N, S1, S2]]
    rgb = np.stack([R, G, B]).astype(np.float32)
    spec = np.stack([B, G, R, N, S1, S2]).astype(np.float32)
    return rgb, spec, SCL


def save_rgb_patch(rgb_patch, region_name, index):
    rgb = np.clip(np.transpose(rgb_patch, (1, 2, 0)), 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    safe_name = region_name.lower().replace(" ", "_")
    out_path = OUTPUT_DIR / f"{safe_name}_{index:02d}.png"
    Image.fromarray(rgb_u8).save(out_path)
    return str(out_path)


def compute_live_physics_risk(physics):
    ndvi = float(physics["future_ndvi"])
    ndmi = float(physics["future_ndmi"])
    ndwi = float(physics["future_ndwi"])
    base = float(physics["severity"])

    def normalize_inverse(value, good, bad):
        scaled = (good - value) / (good - bad)
        return float(np.clip(scaled, 0.0, 1.0))

    low_veg = normalize_inverse(ndvi, good=0.50, bad=0.12)
    low_moisture = normalize_inverse(ndmi, good=0.18, bad=-0.08)
    low_water = normalize_inverse(ndwi, good=0.05, bad=-0.25)
    dryland_risk = 0.45 * low_veg + 0.40 * low_moisture + 0.15 * low_water
    return float(np.clip(0.55 * base + 0.45 * dryland_risk, 0.0, 1.0))


def passes_biome_prefilter(region, physics):
    ndvi = float(physics["future_ndvi"])
    ndmi = float(physics["future_ndmi"])
    expected = region["expected"]

    if expected == "drier":
        return ndvi < 0.24 and ndmi < 0.05
    if expected == "wetter":
        return ndvi > 0.38 and ndmi > 0.08
    return True


def compute_adaptive_combined_probability(model_probability, physics):
    ndvi = float(physics["future_ndvi"])
    ndmi = float(physics["future_ndmi"])
    physics_probability = compute_live_physics_risk(physics)

    looks_dry = ndvi < 0.32 and ndmi < 0.08
    looks_wet = ndvi > 0.45 and ndmi > 0.12

    if looks_dry and physics_probability > model_probability:
        combined = max(
            0.20 * model_probability + 0.80 * physics_probability,
            physics_probability,
        )
        mode = "dryness_boosted"
    elif looks_wet and model_probability > physics_probability:
        combined = 0.25 * model_probability + 0.75 * physics_probability
        mode = "wetness_corrected"
    else:
        combined = 0.50 * model_probability + 0.50 * physics_probability
        mode = "balanced"

    return float(np.clip(combined, 0.0, 1.0)), mode


def summarize_selected_records(records):
    model_probs = np.array([r["model_probability"] for r in records], dtype=np.float32)
    physics_probs = np.array([r["physics_probability"] for r in records], dtype=np.float32)
    combined_probs = np.array([r["combined_probability"] for r in records], dtype=np.float32)
    confidences = np.array([r["model_confidence"] for r in records], dtype=np.float32)
    ndvi = np.array([r["future_ndvi"] for r in records], dtype=np.float32)
    ndmi = np.array([r["future_ndmi"] for r in records], dtype=np.float32)
    ndwi = np.array([r["future_ndwi"] for r in records], dtype=np.float32)

    return {
        "samples": len(records),
        "status": "ok",
        "mean_model_probability": float(np.mean(model_probs)),
        "median_model_probability": float(np.median(model_probs)),
        "mean_physics_probability": float(np.mean(physics_probs)),
        "median_physics_probability": float(np.median(physics_probs)),
        "mean_combined_probability": float(np.mean(combined_probs)),
        "median_combined_probability": float(np.median(combined_probs)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_ndvi": float(np.mean(ndvi)),
        "mean_ndmi": float(np.mean(ndmi)),
        "mean_ndwi": float(np.mean(ndwi)),
        "patches": records,
    }


def select_region_representative_records(region, candidate_records):
    expected = region["expected"]
    if expected == "drier":
        ranked = sorted(
            candidate_records,
            key=lambda r: (
                r["physics_probability"],
                r["combined_probability"],
                -r["future_ndvi"],
                -r["future_ndmi"],
            ),
            reverse=True,
        )
    elif expected == "wetter":
        ranked = sorted(
            candidate_records,
            key=lambda r: (
                r["physics_probability"],
                r["combined_probability"],
                -r["future_ndvi"],
                -r["future_ndmi"],
            ),
        )
    else:
        ranked = sorted(
            candidate_records,
            key=lambda r: abs(r["combined_probability"] - 0.5),
        )
    return ranked[:PATCHES_PER_REGION]


def predict_region(region, spectral_model, drought_model, rng):
    search = CATALOG.search(
        collections=["sentinel-2-l2a"],
        bbox=region["bbox"],
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = sorted(fetch_items_safe(search, SEARCH_LIMIT * 2), key=lambda x: x.datetime)
    preferred_months = set(region.get("preferred_months", []))
    if preferred_months:
        filtered = [item for item in items if item.datetime.month in preferred_months]
        if len(filtered) >= SEQUENCE_LENGTH:
            items = filtered
    items = items[-SEARCH_LIMIT:]
    print(f"  Scenes found: {len(items)}")

    candidate_records = []
    for start_idx in range(max(0, len(items) - SEQUENCE_LENGTH + 1)):
        if len(candidate_records) >= MAX_CANDIDATES_PER_REGION:
            break

        seq_items = build_spaced_sequence(items, start_idx)
        if seq_items is None and start_idx + SEQUENCE_LENGTH <= len(items):
            seq_items = items[start_idx:start_idx + SEQUENCE_LENGTH]
        if seq_items is None:
            continue

        for _ in range(PATCH_ATTEMPTS_PER_SEQUENCE):
            coords = choose_patch_coords(seq_items[0], rng)
            if coords is None:
                continue
            row, col, lat, lon = coords

            rgb_frames = []
            spec_frames = []
            scl_future = None

            valid_sequence = True
            for frame_idx, item in enumerate(seq_items):
                extracted = extract_rgb_and_spec(item, row, col)
                if extracted is None:
                    valid_sequence = False
                    break
                rgb, spec, scl = extracted
                rgb_frames.append(rgb)
                spec_frames.append(spec)
                if frame_idx == len(seq_items) - 1:
                    scl_future = scl

            if not valid_sequence:
                continue

            physics = score_filtered_future_drought(np.stack(spec_frames[:-1]), spec_frames[-1], scl_future)
            if physics is None:
                physics = score_filtered_future_drought(np.stack(spec_frames[:-1]), spec_frames[-1], None)
            if physics is None:
                continue
            if not passes_biome_prefilter(region, physics):
                continue

            rgb_sequence = torch.tensor(np.stack(rgb_frames), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                bsz, steps, _, height, width = rgb_sequence.shape
                pred_spec = spectral_model(rgb_sequence.view(bsz * steps, 3, height, width)).view(bsz, steps, 6, height, width)
                logits, confidence_logits = drought_model(pred_spec)
                model_probability = float(torch.sigmoid(logits).item())
                model_confidence = float(torch.sigmoid(confidence_logits).item())

            combined_probability, fusion_mode = compute_adaptive_combined_probability(model_probability, physics)
            trust_score = 0.55 * model_confidence + 0.45 * float(np.clip(abs(physics["severity"] - 0.5) * 2.0, 0.0, 1.0))
            image_path = save_rgb_patch(rgb_frames[-1], region["name"], len(candidate_records))
            candidate_records.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "scene_dates": [str(item.datetime.date()) for item in seq_items],
                    "model_probability": model_probability,
                    "physics_probability": float(physics["severity"]),
                    "combined_probability": combined_probability,
                    "fusion_mode": fusion_mode,
                    "model_confidence": model_confidence,
                    "future_ndvi": float(physics["future_ndvi"]),
                    "future_ndmi": float(physics["future_ndmi"]),
                    "future_ndwi": float(physics["future_ndwi"]),
                    "trust_score": trust_score,
                    "image_path": image_path,
                }
            )
            break

    if not candidate_records:
        return {
            "samples": 0,
            "status": "no_valid_sequences",
            "candidate_count": 0,
            "all_candidates": [],
            "selected_candidates": [],
        }

    selected = select_region_representative_records(region, candidate_records)
    summary = summarize_selected_records(selected)
    summary["candidate_count"] = len(candidate_records)
    summary["expected"] = region["expected"]
    summary["selected_candidates"] = selected
    summary["all_candidates"] = sorted(candidate_records, key=lambda r: r["combined_probability"], reverse=True)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = random.Random(42)
    np.random.seed(42)

    print("=" * 80)
    print("REAL REGION FETCH + DROUGHT INFERENCE TEST - EXTENDED")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("Source: Microsoft Planetary Computer Sentinel-2")

    spectral_model = load_spectral_model()
    drought_model = load_drought_model()

    all_results = {}
    diagnostic_rows = []

    for region in REGIONS:
        print(f"\n[REGION] {region['name']} ({region['expected']})")
        result = predict_region(region, spectral_model, drought_model, rng)
        all_results[region["name"]] = result

        if result["samples"] == 0:
            print("  No valid real sequences found")
            continue

        selected_keys = {(r["image_path"], tuple(r["scene_dates"])) for r in result.get("selected_candidates", [])}
        for idx, candidate in enumerate(result.get("all_candidates", [])):
            diagnostic_rows.append(
                {
                    "region": region["name"],
                    "expected": region["expected"],
                    "rank_by_combined_probability": idx + 1,
                    "selected": (candidate["image_path"], tuple(candidate["scene_dates"])) in selected_keys,
                    "lat": candidate["lat"],
                    "lon": candidate["lon"],
                    "scene_dates": " | ".join(candidate["scene_dates"]),
                    "model_probability": candidate["model_probability"],
                    "physics_probability": candidate["physics_probability"],
                    "combined_probability": candidate["combined_probability"],
                    "fusion_mode": candidate.get("fusion_mode", ""),
                    "model_confidence": candidate["model_confidence"],
                    "future_ndvi": candidate["future_ndvi"],
                    "future_ndmi": candidate["future_ndmi"],
                    "future_ndwi": candidate["future_ndwi"],
                    "trust_score": candidate.get("trust_score", float("nan")),
                    "image_path": candidate["image_path"],
                }
            )

        print(f"  Samples: {result['samples']}")
        print(f"  Candidates scored:    {result['candidate_count']}")
        print(f"  Model probability:    {result['median_model_probability']:.3f} (median)")
        print(f"  Physics probability:  {result['median_physics_probability']:.3f} (median)")
        print(f"  Combined probability: {result['median_combined_probability']:.3f} (median)")
        print(f"  Confidence:           {result['mean_confidence']:.3f}")
        print(f"  NDVI / NDMI / NDWI:   {result['mean_ndvi']:.3f} / {result['mean_ndmi']:.3f} / {result['mean_ndwi']:.3f}")
        print(f"  Example image:        {result['patches'][0]['image_path']}")

    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"source": "real Sentinel-2 imagery fetched per region", "date_range": DATE_RANGE, "results": all_results}, fh, indent=2)

    with open(DIAGNOSTICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": "real Sentinel-2 imagery fetched per region",
                "date_range": DATE_RANGE,
                "search_limit": SEARCH_LIMIT,
                "patches_per_region": PATCHES_PER_REGION,
                "patch_attempts_per_sequence": PATCH_ATTEMPTS_PER_SEQUENCE,
                "max_candidates_per_region": MAX_CANDIDATES_PER_REGION,
                "results": all_results,
            },
            fh,
            indent=2,
        )

    if diagnostic_rows:
        with open(DIAGNOSTICS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(diagnostic_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diagnostic_rows)

    print(f"\nSaved: {RESULTS_JSON}")
    print(f"Saved diagnostics: {DIAGNOSTICS_JSON}")
    print(f"Saved candidate table: {DIAGNOSTICS_CSV}")
    print(f"Saved images in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
