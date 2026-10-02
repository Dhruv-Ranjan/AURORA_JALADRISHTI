from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable


def geographic_temporal_split(
    records: Iterable[dict[str, object]],
    validation_regions: set[str],
    test_regions: set[str],
    temporal_cutoff: datetime | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Split by region first, then by time; never split a scene's patch rows."""

    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        region = str(record["region"])
        acquired = record.get("acquired")
        if region in test_regions:
            bucket = "test"
        elif region in validation_regions:
            bucket = "validation"
        elif temporal_cutoff and acquired:
            date = acquired if isinstance(acquired, datetime) else datetime.fromisoformat(str(acquired))
            bucket = "test" if date >= temporal_cutoff else "train"
        else:
            bucket = "train"
        result[bucket].append(record)
    return {key: list(value) for key, value in result.items()}
