#!/usr/bin/env python3
"""Train SpectralNetV10 on a materialized remote patch dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_defs import SpectralNetV10


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = Path(os.environ["MASSIVE_DATASET"])
    output = root / "artifacts" / "spectral_massive_10000.pt"
    report_path = root / "artifacts" / "spectral_massive_10000_metrics.json"
    data = np.load(data_path, allow_pickle=True)
    rgb = torch.from_numpy(data["rgb"]).float()
    spec = torch.from_numpy(data["spec"]).float()
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(rgb), generator=generator)
    split = max(1, int(len(rgb) * 0.9))
    train_idx, val_idx = permutation[:split], permutation[split:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralNetV10().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(rgb[train_idx], spec[train_idx]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(rgb[val_idx], spec[val_idx]), batch_size=32)
    epochs = int(os.environ.get("MASSIVE_EPOCHS", "4"))
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        errors = []
        with torch.no_grad():
            for xb, yb in val_loader:
                errors.append((model(xb.to(device)).cpu() - yb).numpy())
        error = np.concatenate(errors)
        metrics = {
            "epoch": epoch + 1,
            "mse": float(np.mean(error**2)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }
        history.append(metrics)
        print(metrics, flush=True)
    torch.save({"model_state_dict": model.state_dict(), "dataset": data_path.name, "history": history}, output)
    report_path.write_text(json.dumps({"dataset": data_path.name, "samples": len(rgb), "history": history}, indent=2))
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
