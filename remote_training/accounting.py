from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass
class SampleAccounting:
    """Counts candidate outcomes without conflating candidates and samples."""

    candidates_examined: int = 0
    valid_samples: int = 0
    cloud_rejected: int = 0
    quality_rejected: int = 0
    missing_bands: int = 0
    invalid_pixels: int = 0
    label_unavailable: int = 0
    network_failures: int = 0
    other_failures: int = 0
    replacement_candidates_requested: int = 0
    duplicate_rejected: int = 0
    geographic_balance_rejections: int = 0
    temporal_balance_rejections: int = 0

    def record(self, reason: str | None) -> None:
        self.candidates_examined += 1
        if reason is None:
            self.valid_samples += 1
            return
        field = {
            "cloud": "cloud_rejected",
            "quality": "quality_rejected",
            "missing_bands": "missing_bands",
            "invalid_pixels": "invalid_pixels",
            "label": "label_unavailable",
            "network": "network_failures",
            "duplicate": "duplicate_rejected",
            "geographic_balance": "geographic_balance_rejections",
            "temporal_balance": "temporal_balance_rejections",
        }.get(reason, "other_failures")
        setattr(self, field, getattr(self, field) + 1)

    def merge(self, other: "SampleAccounting") -> None:
        for key in asdict(other):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def as_dict(self) -> Mapping[str, int]:
        return asdict(self)
