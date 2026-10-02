from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RemoteTrainingConfig:
    """Configuration for bounded, windowed Planetary Computer training."""

    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection: str = "sentinel-2-l2a"
    date_range: str = "2023-01-01/2025-12-31"
    cloud_cover_max: float = 30.0
    patch_size: int = 64
    sequence_length: int = 3
    min_gap_days: int = 5
    max_gap_days: int = 240
    target_valid_samples: int = 1000
    max_candidates: int = 0
    patches_per_item: int = 8
    max_retries: int = 4
    request_timeout_seconds: float = 60.0
    retry_backoff_seconds: float = 1.0
    cache_enabled: bool = True
    cache_size_gb: float = 2.0
    cache_directory: Path = Path(".aurora_remote_cache")
    manifest_path: Path = Path("remote_samples.jsonl")
    checkpoint_path: Path = Path("remote_spectral_checkpoint.pt")
    output_directory: Path = Path("remote_training_results")
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    num_workers: int = 0
    mixed_precision: bool = True
    cuda_device: str = "cuda:0"
    seed: int = 42
    regions: tuple[dict[str, object], ...] = field(default_factory=tuple)

    @property
    def cache_limit_bytes(self) -> int:
        return max(0, int(self.cache_size_gb * 1024**3))

    def validate(self) -> None:
        if self.patch_size <= 0 or self.patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.target_valid_samples <= 0:
            raise ValueError("target_valid_samples must be positive")
        if self.cache_size_gb < 0:
            raise ValueError("cache_size_gb must not be negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.patches_per_item <= 0:
            raise ValueError("patches_per_item must be positive")
