from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from remote_training.accounting import SampleAccounting
from remote_training.cache import BoundedFileCache
from remote_training.config import RemoteTrainingConfig
from remote_training.splits import geographic_temporal_split


def test_accounting_distinguishes_valid_and_rejected_candidates():
    counts = SampleAccounting()
    counts.record(None)
    counts.record("cloud")
    counts.record("invalid_pixels")
    assert counts.valid_samples == 1
    assert counts.candidates_examined == 3
    assert counts.cloud_rejected == 1
    assert counts.invalid_pixels == 1


def test_cache_evicts_oldest_file_at_limit():
    with tempfile.TemporaryDirectory() as directory:
        cache = BoundedFileCache(Path(directory), limit_bytes=4)
        cache.put("first", b"1234")
        cache.put("second", b"5678")
        assert cache.usage_bytes() <= 4
        assert cache.get("second") == b"5678"


def test_split_keeps_regions_in_one_partition():
    records = [
        {"region": "validation-region", "acquired": "2023-01-01"},
        {"region": "test-region", "acquired": "2024-01-01"},
        {"region": "train-region", "acquired": datetime(2023, 1, 1)},
    ]
    split = geographic_temporal_split(
        records,
        validation_regions={"validation-region"},
        test_regions={"test-region"},
        temporal_cutoff=datetime(2023, 6, 1),
    )
    assert len(split["validation"]) == 1
    assert len(split["test"]) == 1
    assert len(split["train"]) == 1


def test_config_rejects_unbounded_invalid_values():
    config = RemoteTrainingConfig(target_valid_samples=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "target_valid_samples" in str(exc)
    else:
        raise AssertionError("invalid configuration was accepted")
