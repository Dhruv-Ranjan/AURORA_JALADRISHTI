from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import RemoteTrainingConfig
from .dataset import RemoteSequenceDataset, discover_samples
from .sampler import diversify_samples

LOGGER = logging.getLogger("aurora.remote_training")


def load_factory(spec: str) -> Callable[[], torch.nn.Module]:
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    return factory


class TorchIterableDataset(IterableDataset):
    def __init__(self, source: RemoteSequenceDataset):
        self.source = source

    def __iter__(self):
        yield from self.source


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(rows: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.stack([row["rgb"] for row in rows])),
        torch.from_numpy(np.stack([row["target"] for row in rows])),
    )


def train(
    config: RemoteTrainingConfig,
    model_factory: Callable[[], torch.nn.Module],
    initial_checkpoint: Path | None = None,
) -> dict[str, object]:
    config.validate()
    _seed(config.seed)
    device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOGGER.warning("CUDA is unavailable; this run is CPU-bound and should be treated as a smoke test.")
    else:
        LOGGER.info("CUDA available: %s (%s)", torch.cuda.is_available(), torch.cuda.get_device_name(device))

    samples = diversify_samples(discover_samples(config), config.seed)
    source = RemoteSequenceDataset(config, samples)
    loader = DataLoader(
        TorchIterableDataset(source),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    start_epoch = 0
    if initial_checkpoint is not None:
        initial = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        state = initial.get("model_state_dict", initial.get("model", initial))
        model.load_state_dict(state)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if config.checkpoint_path.exists():
        checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.monotonic()
    steps = 0
    total_loss = 0.0
    for epoch in range(start_epoch, config.epochs):
        model.train()
        for rgb, target in loader:
            rgb, target = rgb.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                prediction = model(rgb)
                loss = torch.nn.functional.l1_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            steps += 1
        torch.save(
            {
                "epoch": epoch,
                "step": steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
                "accounting": source.accounting.as_dict(),
            },
            config.checkpoint_path,
        )
        if source.accounting.valid_samples >= config.target_valid_samples:
            break

    elapsed = max(time.monotonic() - started, 1e-9)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    report = {
        "config": asdict(config),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "steps": steps,
        "mean_training_loss": total_loss / max(steps, 1),
        "elapsed_seconds": elapsed,
        "samples_per_second": source.accounting.valid_samples / elapsed,
        "cache_bytes": source.reader.cache.usage_bytes(),
        "peak_cache_bytes": source.reader.cache.peak_bytes,
        "disk_free_bytes": shutil.disk_usage(config.output_directory).free,
        "accounting": source.accounting.as_dict(),
    }
    (config.output_directory / "training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    if source.exhausted:
        LOGGER.warning(
            "Candidate pool exhausted at %d valid samples; requested %d.",
            source.accounting.valid_samples,
            config.target_valid_samples,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an existing AURORA model from remote windows.")
    parser.add_argument("--model-factory", required=True, help="module:function returning the existing model")
    parser.add_argument("--target-valid-samples", type=int, default=1000)
    parser.add_argument("--regions-json", type=Path, required=True)
    parser.add_argument("--cache-size-gb", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patches-per-item", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=Path("remote_samples.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("remote_spectral_checkpoint.pt"))
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("remote_training_results"))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    regions = tuple(json.loads(args.regions_json.read_text(encoding="utf-8")))
    config = RemoteTrainingConfig(
        target_valid_samples=args.target_valid_samples,
        cache_size_gb=args.cache_size_gb,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patches_per_item=args.patches_per_item,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_directory=args.output_dir,
        epochs=args.epochs,
        regions=regions,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(config, load_factory(args.model_factory), args.initial_checkpoint)


if __name__ == "__main__":
    main()
