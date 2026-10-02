from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .dataset import RemoteSample


def diversify_samples(samples: list[RemoteSample], seed: int = 42) -> list[RemoteSample]:
    """Interleave region and acquisition-month buckets."""

    buckets: dict[tuple[str, int], list[RemoteSample]] = defaultdict(list)
    for sample in samples:
        try:
            month = datetime.fromisoformat(sample.acquired.replace("Z", "+00:00")).month
        except ValueError:
            month = 0
        buckets[(sample.region, month)].append(sample)
    ordered: list[RemoteSample] = []
    while buckets:
        for key in sorted(tuple(buckets)):
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop(0))
            if not bucket:
                buckets.pop(key, None)
    return ordered
