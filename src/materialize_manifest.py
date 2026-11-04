#!/usr/bin/env python3
"""Materialize validated RGB/target patches from an existing STAC manifest."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    release = Path(__file__).resolve().parents[2] / "AURORA_SR_RELEASE"
    sys.path.insert(0, str(release))
    from remote_training.config import RemoteTrainingConfig
    from remote_training.dataset import REQUIRED_BANDS, RemoteSample, RemoteWindowReader

    import planetary_computer
    import pystac_client

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break

    ids = sorted({row["item_id"].split("#", 1)[0] for row in rows})
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items = {}
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        search = catalog.search(collections=["sentinel-2-l2a"], ids=batch)
        for item in search.items():
            items[item.id] = planetary_computer.sign(item).to_dict()
        print(f"metadata={min(start + len(batch), len(ids))}/{len(ids)}", flush=True)

    config = RemoteTrainingConfig(
        target_valid_samples=args.limit,
        cache_directory=args.cache,
        cache_size_gb=20,
        manifest_path=args.output.with_suffix(".jsonl"),
        patches_per_item=1,
    )
    def process(row):
        item_id = row["item_id"].split("#", 1)[0]
        payload = items.get(item_id)
        if payload is None:
            return None, None, row, "missing_item"
        sample = RemoteSample(
            item_id=row["item_id"],
            region=row["region"],
            acquired=row["acquired"],
            longitude=float(row["longitude"]),
            latitude=float(row["latitude"]),
            asset_hrefs={},
            item_payload=payload,
        )
        try:
            rgb, target = RemoteWindowReader(config).read(sample)
        except Exception as exc:
            return None, None, row, f"{type(exc).__name__}: {exc}"
        return rgb, target, row, None

    rgbs, targets, metadata = [], [], []
    rejected = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, (rgb, target, row, error) in enumerate(executor.map(process, rows), start=1):
            if error is not None:
                rejected += 1
                if rejected <= 20:
                    print(f"rejected {index}/{len(rows)}: {error}", flush=True)
                continue
            rgbs.append(rgb)
            targets.append(target)
            metadata.append(row)
            if len(rgbs) % 100 == 0:
                print(f"validated={len(rgbs)} rejected={rejected}", flush=True)

    if not rgbs:
        raise RuntimeError("No valid patches were materialized.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rgb=np.stack(rgbs),
        spec=np.stack(targets),
        metadata=np.asarray(metadata, dtype=object),
    )
    print(f"saved={args.output} valid={len(rgbs)} rejected={rejected}", flush=True)


if __name__ == "__main__":
    main()
