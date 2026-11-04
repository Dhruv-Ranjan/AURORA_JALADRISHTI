from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from drought_model import (
    PhysicsTemporalDroughtModel,
    build_drought_physics_features,
    build_sequence_physics_vector,
)
from model_defs import (
    SpectralNetV10,
    binary_classification_metrics_with_threshold,
    load_sequence_dataset_extended,
    predict_spectral_with_tta,
)


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DEFAULT_DATASET_CANDIDATES = [
    BASE_DIR / "dataset_seq_forecast_merged_real.npz",
    BASE_DIR / "dataset_seq_forecast_production_real.npz",
    BASE_DIR / "dataset_seq_forecast_v4.npz",
    BASE_DIR / "dataset_seq_forecast_v2.npz",
    BASE_DIR / "dataset_seq_forecast_v1.npz",
]
DEFAULT_SPECTRAL_MODEL = Path(
    os.environ.get("SPECTRAL_MODEL_INPUT", ARTIFACT_DIR / "spectral_v10_1_balanced.pt")
)
DEFAULT_DROUGHT_MODEL = Path(
    os.environ.get("DROUGHT_MODEL_INPUT", ARTIFACT_DIR / "drought_realworld_production_model.pt")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAST_SPECTRAL_INFERENCE = True


@dataclass
class RealDatasetBundle:
    dataset_path: Path
    rgb: torch.Tensor
    spec: torch.Tensor
    y_hard: torch.Tensor
    y_soft: torch.Tensor
    y_conf: torch.Tensor
    region_name: np.ndarray
    label_source: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


class RealWorldDroughtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lstm = nn.LSTM(
            128,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )
        self.drought_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch_size, steps, channels, height, width = x.shape
        flat = x.reshape(batch_size * steps, channels, height, width)
        spatial_features = self.spatial(flat).reshape(batch_size, steps, 128)
        temporal_features, _ = self.lstm(spatial_features)
        context = temporal_features[:, -1, :]
        drought_logits = self.drought_head(context).squeeze(-1)
        confidence_logits = self.confidence_head(context).squeeze(-1)
        return drought_logits, confidence_logits


class PhysicsDroughtWrapper(nn.Module):
    def __init__(self, in_channels: int = 18, physics_dim: int = 33):
        super().__init__()
        self.model = PhysicsTemporalDroughtModel(
            in_channels=in_channels,
            physics_dim=physics_dim,
            hidden_dim=192,
            dropout=0.30,
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(physics_dim),
            nn.Linear(physics_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        physics_frames = torch.stack([build_drought_physics_features(x[:, t]) for t in range(x.shape[1])], dim=1)
        physics_vector, physics_prior = build_sequence_physics_vector(physics_frames)
        logits = self.model(physics_frames, physics_vector)
        confidence_logits = self.confidence_head(physics_vector).squeeze(-1) + torch.logit(
            physics_prior.clamp(1e-4, 1 - 1e-4)
        )
        return logits, confidence_logits


def choose_dataset_path(preferred: Path | None = None) -> Path:
    if preferred is None and os.environ.get("REAL_DATASET_FILE"):
        preferred = Path(os.environ["REAL_DATASET_FILE"])
    candidates = [preferred] if preferred is not None else []
    candidates.extend(DEFAULT_DATASET_CANDIDATES)
    for path in candidates:
        if path is not None and Path(path).exists():
            return Path(path)
    raise FileNotFoundError("No cached real sequence dataset found.")


def available_dataset_inputs() -> list[Path]:
    candidates = sorted(
        path
        for path in BASE_DIR.glob("dataset_seq_forecast*.npz")
        if path.name != "dataset_seq_forecast_merged_real.npz"
    )
    return candidates


def _coerce_soft_targets(y_hard: torch.Tensor, raw_soft: torch.Tensor | None, raw_conf: torch.Tensor | None):
    if raw_soft is None:
        y_soft = y_hard.clone()
    else:
        y_soft = raw_soft.clone()
        invalid = ~torch.isfinite(y_soft)
        y_soft[invalid] = y_hard[invalid]

    if raw_conf is None:
        y_conf = torch.ones_like(y_hard)
    else:
        y_conf = raw_conf.clone()
        invalid = ~torch.isfinite(y_conf)
        y_conf[invalid] = 0.35

    return y_soft.float(), torch.clamp(y_conf.float(), 0.0, 1.0)


def load_real_dataset(dataset_path: Path | None = None) -> RealDatasetBundle:
    dataset_path = choose_dataset_path(dataset_path)
    payload = load_sequence_dataset_extended(dataset_path)
    y_hard = payload["y"].float()
    y_soft, y_conf = _coerce_soft_targets(y_hard, payload.get("y_soft"), payload.get("y_confidence"))
    n_samples = len(y_hard)
    region_name = np.asarray(payload.get("region_name", np.array(["Unknown"] * n_samples, dtype=object)))
    label_source = np.asarray(payload.get("label_source", np.array(["unknown"] * n_samples, dtype=object)))
    lat = np.asarray(payload.get("lat", np.full(n_samples, np.nan, dtype=np.float32)), dtype=np.float32)
    lon = np.asarray(payload.get("lon", np.full(n_samples, np.nan, dtype=np.float32)), dtype=np.float32)
    return RealDatasetBundle(
        dataset_path=dataset_path,
        rgb=payload["rgb"].float(),
        spec=payload["spec"].float(),
        y_hard=y_hard,
        y_soft=y_soft,
        y_conf=y_conf,
        region_name=region_name,
        label_source=label_source,
        lat=lat,
        lon=lon,
    )


def load_spectral_model(model_path: Path | None = None, device: torch.device = DEVICE) -> SpectralNetV10:
    model_path = Path(model_path or DEFAULT_SPECTRAL_MODEL)
    model = SpectralNetV10().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    return model


def load_drought_model(model_path: Path | None = None, device: torch.device = DEVICE) -> RealWorldDroughtNet:
    model_path = Path(model_path or DEFAULT_DROUGHT_MODEL)
    state = torch.load(model_path, map_location=device)
    architecture = state.get("architecture") if isinstance(state, dict) else None
    if architecture == "physics_temporal_v1":
        model = PhysicsDroughtWrapper().to(device)
    else:
        model = RealWorldDroughtNet().to(device)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    return model


def run_spectral_reconstruction(
    spectral_model: SpectralNetV10,
    rgb_sequence: torch.Tensor,
    batch_size: int = 8,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(rgb_sequence), batch_size=batch_size, shuffle=False)
    outputs = []
    spectral_model.eval()
    with torch.no_grad():
        for (rgb_batch,) in loader:
            rgb_batch = rgb_batch.to(device)
            bsz, steps, _, height, width = rgb_batch.shape
            flat = rgb_batch.view(bsz * steps, 3, height, width)
            pred = (spectral_model(flat) if FAST_SPECTRAL_INFERENCE else predict_spectral_with_tta(spectral_model, flat)).view(
                bsz, steps, 6, height, width
            )
            outputs.append(pred.cpu())
    return torch.cat(outputs, dim=0)


def compute_reconstruction_metrics(pred_spec: torch.Tensor, true_spec: torch.Tensor) -> dict:
    diff = pred_spec - true_spec
    mse = float(torch.mean(diff ** 2).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def predict_drought_probabilities(
    drought_model: RealWorldDroughtNet,
    spec_sequence: torch.Tensor,
    batch_size: int = 8,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(spec_sequence), batch_size=batch_size, shuffle=False)
    probs, confs = [], []
    drought_model.eval()
    with torch.no_grad():
        for (spec_batch,) in loader:
            spec_batch = spec_batch.to(device)
            logits, conf_logits = drought_model(spec_batch)
            probs.append(torch.sigmoid(logits).cpu())
            confs.append(torch.sigmoid(conf_logits).cpu())
    return torch.cat(probs).numpy(), torch.cat(confs).numpy()


def evaluate_predictions(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    metrics = binary_classification_metrics_with_threshold(probs, labels, threshold=threshold)
    metrics["mean_probability"] = float(np.mean(probs))
    return metrics


def region_summary(
    bundle: RealDatasetBundle,
    probs: np.ndarray,
    confs: np.ndarray,
    regions: Iterable[str],
) -> list[dict]:
    summaries = []
    for region in regions:
        mask = bundle.region_name == region
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        summaries.append(
            {
                "region": region,
                "samples": int(len(idx)),
                "mean_probability": float(np.mean(probs[idx])),
                "mean_confidence": float(np.mean(confs[idx])),
                "mean_hard_label": float(torch.mean(bundle.y_hard[idx]).item()),
                "mean_soft_label": float(torch.mean(bundle.y_soft[idx]).item()),
                "mean_lat": float(np.nanmean(bundle.lat[idx])) if np.isfinite(bundle.lat[idx]).any() else float("nan"),
                "mean_lon": float(np.nanmean(bundle.lon[idx])) if np.isfinite(bundle.lon[idx]).any() else float("nan"),
            }
        )
    summaries.sort(key=lambda item: item["mean_probability"])
    return summaries
