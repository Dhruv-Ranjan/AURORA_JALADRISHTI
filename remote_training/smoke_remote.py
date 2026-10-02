from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RemoteTrainingConfig
from .dataset import discover_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover remote metadata without downloading imagery.")
    parser.add_argument("--regions-json", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=5)
    args = parser.parse_args()
    regions = tuple(json.loads(args.regions_json.read_text(encoding="utf-8")))
    config = RemoteTrainingConfig(regions=regions, max_candidates=args.max_candidates)
    samples = discover_samples(config)
    print(json.dumps({"metadata_samples": len(samples), "remote_read": False}, indent=2))


if __name__ == "__main__":
    main()
