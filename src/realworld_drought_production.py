#!/usr/bin/env python3
"""
Real-data drought production trainer.

This version is intentionally strict about data provenance:
- trains from cached real Sentinel-2 sequence data already stored locally
- optionally fine-tunes the spectral reconstruction model on RGB -> spectral
- trains the drought model on reconstructed spectral sequences
- reports actual metrics instead of claiming "perfect" predictions
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from drought_model import build_drought_physics_features, build_sequence_physics_vector
from model_defs import binary_classification_metrics_with_threshold, compute_auc, find_best_threshold, predict_spectral_with_tta
from real_data_pipeline import (
    BASE_DIR,
    DEFAULT_DROUGHT_MODEL,
    DEFAULT_SPECTRAL_MODEL,
    DEVICE,
    PhysicsDroughtWrapper,
    choose_dataset_path,
    compute_reconstruction_metrics,
    load_real_dataset,
    load_spectral_model,
)
import torch.nn.functional as F


SPECTRAL_EPOCHS = int(os.environ.get("SPECTRAL_EPOCHS", "4"))
DROUGHT_EPOCHS = int(os.environ.get("DROUGHT_EPOCHS", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "3e-4"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.2"))
SEED = int(os.environ.get("SEED", "42"))
REGION_AWARE_SPLIT = os.environ.get("REGION_AWARE_SPLIT", "0") == "1"
REGION_HOLDOUT_SPLIT = os.environ.get("REGION_HOLDOUT_SPLIT", "0") == "1"
FAST_SPECTRAL_INFERENCE = os.environ.get("FAST_SPECTRAL_INFERENCE", "1") == "1"

SPECTRAL_SAVE = BASE_DIR / os.environ.get("SPECTRAL_SAVE_FILE", "spectral_realworld_production.pt")
DROUGHT_SAVE = BASE_DIR / os.environ.get("DROUGHT_SAVE_FILE", "drought_realworld_production_model.pt")
RESULTS_SAVE = BASE_DIR / os.environ.get("RESULTS_SAVE_FILE", "drought_realworld_results.json")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def split_indices(
    labels: torch.Tensor,
    regions=None,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    holdout_groups: bool = False,
):
    labels_np = labels.cpu().numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    if regions is not None and holdout_groups:
        regions_np = np.asarray(regions)
        all_region_values = np.unique(regions_np)
        region_values = np.asarray(
            [region for region in all_region_values if not str(region).startswith("dataset_seq_forecast_")]
        )
        if len(region_values) < 2:
            raise ValueError("Region holdout needs at least two named geographic regions.")
        val_count = min(len(region_values) - 1, max(1, int(round(len(region_values) * val_fraction))))
        for _ in range(200):
            val_regions = rng.choice(region_values, size=val_count, replace=False)
            val_mask = np.isin(regions_np, val_regions)
            if len(np.unique(labels_np[val_mask])) == 2 and len(np.unique(labels_np[~val_mask])) == 2:
                return (
                    torch.tensor(np.flatnonzero(~val_mask), dtype=torch.long),
                    torch.tensor(np.flatnonzero(val_mask), dtype=torch.long),
                )
        raise ValueError("Could not make a region-disjoint split containing both classes in train and validation.")

    train_idx, val_idx = [], []

    if regions is None:
        groups = [(None, label_value) for label_value in np.unique(labels_np)]
    else:
        regions_np = np.asarray(regions)
        groups = [(region_value, label_value) for region_value in np.unique(regions_np) for label_value in np.unique(labels_np)]

    for region_value, label_value in groups:
        if regions is None:
            idx = np.where(labels_np == label_value)[0]
        else:
            idx = np.where((regions_np == region_value) & (labels_np == label_value))[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        val_count = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:val_count].tolist())
        train_idx.extend(idx[val_count:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


def run_spectral_model(model, rgb_seq: torch.Tensor, batch_size: int):
    loader = DataLoader(TensorDataset(rgb_seq), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            bsz, steps, _, height, width = xb.shape
            flat = xb.view(bsz * steps, 3, height, width)
            pred = model(flat) if FAST_SPECTRAL_INFERENCE else predict_spectral_with_tta(model, flat)
            outputs.append(pred.view(bsz, steps, 6, height, width).cpu())
    return torch.cat(outputs, dim=0)


def _apply_spectral_augmentation(xb, yb):
    if torch.rand(1).item() > 0.5:
        xb = torch.flip(xb, dims=[-1])
        yb = torch.flip(yb, dims=[-1])
    if torch.rand(1).item() > 0.5:
        xb = torch.flip(xb, dims=[-2])
        yb = torch.flip(yb, dims=[-2])
    return xb, yb


def _spectral_loss(pred, target, aux_mid=None, aux_low=None):
    weights = torch.tensor([1.0, 1.0, 1.0, 1.20, 1.20, 1.20], device=target.device).view(1, 6, 1, 1)
    main_loss = torch.mean(((pred - target) ** 2) * weights) + 0.1 * F.smooth_l1_loss(pred, target)
    if aux_mid is not None and aux_low is not None:
        aux_loss = F.mse_loss(aux_mid, target) + F.mse_loss(aux_low, target)
        return main_loss + 0.30 * aux_loss
    return main_loss


def _focal_loss(logits, labels, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    focal_weight = alpha_t * ((1.0 - p_t) ** gamma)
    return focal_weight * bce


def fine_tune_spectral_model(bundle):
    model = load_spectral_model(DEFAULT_SPECTRAL_MODEL, device=DEVICE)
    split_regions = bundle.region_name if REGION_AWARE_SPLIT or REGION_HOLDOUT_SPLIT else None
    train_idx, val_idx = split_indices(
        bundle.y_hard,
        regions=split_regions,
        holdout_groups=REGION_HOLDOUT_SPLIT,
    )
    train_rgb = bundle.rgb[train_idx].reshape(-1, 3, 64, 64)
    train_spec = bundle.spec[train_idx].reshape(-1, 6, 64, 64)
    val_rgb = bundle.rgb[val_idx].reshape(-1, 3, 64, 64)
    val_spec = bundle.spec[val_idx].reshape(-1, 6, 64, 64)

    train_loader = DataLoader(TensorDataset(train_rgb, train_spec), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_rgb, val_spec), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, SPECTRAL_EPOCHS))
    best_val = float("inf")
    best_state = None
    best_metrics = None

    for epoch in range(SPECTRAL_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            xb, yb = _apply_spectral_augmentation(xb, yb)
            pred, aux_mid, aux_low = model(xb, return_aux=True)
            loss = _spectral_loss(pred, yb, aux_mid=aux_mid, aux_low=aux_low)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        scheduler.step()
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred, aux_mid, aux_low = model(xb, return_aux=True)
                loss = _spectral_loss(pred, yb, aux_mid=aux_mid, aux_low=aux_low)
                val_loss += loss.item() * xb.size(0)
                val_predictions.append(pred.cpu())
                val_targets.append(yb.cpu())

        val_loss /= max(len(val_loader.dataset), 1)
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
        val_metrics = compute_reconstruction_metrics(val_predictions, val_targets)
        print(
            f"[spectral] epoch {epoch + 1}/{SPECTRAL_EPOCHS} "
            f"train={train_loss / max(len(train_loader.dataset), 1):.5f} "
            f"val={val_loss:.5f} mse={val_metrics['mse']:.5f} mae={val_metrics['mae']:.5f} rmse={val_metrics['rmse']:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            best_metrics = val_metrics.copy()

    if best_state is not None:
        torch.save(
            {
                "architecture": "spectral_v10_multiscale",
                "model_state_dict": best_state,
                "training": {
                    "spectral_epochs": SPECTRAL_EPOCHS,
                    "learning_rate": LEARNING_RATE * 0.5,
                    "batch_size": BATCH_SIZE,
                    "region_aware_split": REGION_AWARE_SPLIT,
                },
                "validation_metrics": best_metrics,
            },
            SPECTRAL_SAVE,
        )
    elif not SPECTRAL_SAVE.exists():
        torch.save(
            {
                "architecture": "spectral_v10_multiscale",
                "model_state_dict": model.state_dict(),
            },
            SPECTRAL_SAVE,
        )

    return model


def train_drought_model(bundle, spectral_model):
    reconstructed = run_spectral_model(spectral_model, bundle.rgb, batch_size=BATCH_SIZE)
    split_regions = bundle.region_name if REGION_AWARE_SPLIT or REGION_HOLDOUT_SPLIT else None
    train_idx, val_idx = split_indices(
        bundle.y_hard,
        regions=split_regions,
        holdout_groups=REGION_HOLDOUT_SPLIT,
    )

    X_train = reconstructed[train_idx]
    y_train = bundle.y_hard[train_idx]
    y_train_soft = bundle.y_soft[train_idx]
    y_train_conf = bundle.y_conf[train_idx]

    X_val = reconstructed[val_idx]
    y_val = bundle.y_hard[val_idx]
    y_val_soft = bundle.y_soft[val_idx]
    y_val_conf = bundle.y_conf[val_idx]
    heldout_regions = sorted(set(bundle.region_name[val_idx].astype(str).tolist())) if REGION_HOLDOUT_SPLIT else []

    model = PhysicsDroughtWrapper().to(DEVICE)
    if DEFAULT_DROUGHT_MODEL.exists():
        state = torch.load(DEFAULT_DROUGHT_MODEL, map_location=DEVICE)
        if isinstance(state, dict) and state.get("architecture") == "physics_temporal_v1":
            model.load_state_dict(state["model_state_dict"])

    positive_count = float(y_train.sum().item())
    negative_count = float(len(y_train) - positive_count)
    pos_weight = torch.tensor([max(negative_count / max(positive_count, 1.0), 1.0)], device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sample_weights = torch.where(y_train > 0.5, torch.full_like(y_train, pos_weight.item()), torch.ones_like(y_train))
    sampler = WeightedRandomSampler(weights=sample_weights.double(), num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, y_train_soft, y_train_conf),
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val, y_val_soft, y_val_conf), batch_size=BATCH_SIZE, shuffle=False)

    best_f1 = -1.0
    best_metrics = None
    best_threshold = 0.5
    physics_baseline_probs = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X_val), batch_size=BATCH_SIZE, shuffle=False):
            xb = xb.to(DEVICE)
            physics_frames = torch.stack([build_drought_physics_features(xb[:, t]) for t in range(xb.shape[1])], dim=1)
            _, physics_prior = build_sequence_physics_vector(physics_frames)
            physics_baseline_probs.extend(physics_prior.cpu().numpy().tolist())
    physics_threshold, physics_metrics = find_best_threshold(physics_baseline_probs, y_val.cpu().numpy(), metric="f1")
    physics_metrics["auc"] = compute_auc(physics_baseline_probs, y_val.cpu().numpy())
    physics_metrics["threshold"] = physics_threshold
    print(
        f"[physics baseline] heldout_auc={physics_metrics['auc']:.4f} "
        f"heldout_f1={physics_metrics['f1']:.4f} threshold={physics_threshold:.2f}"
    )

    for epoch in range(DROUGHT_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb, yb_soft, yb_conf in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yb_soft = yb_soft.to(DEVICE)
            yb_conf = yb_conf.to(DEVICE)
            logits, conf_logits = model(xb)
            probs = torch.sigmoid(logits)
            conf_probs = torch.sigmoid(conf_logits)
            physics_frames = torch.stack([build_drought_physics_features(xb[:, t]) for t in range(xb.shape[1])], dim=1)
            _, physics_prior = build_sequence_physics_vector(physics_frames)

            cls_loss = _focal_loss(logits, yb)
            reg_loss = torch.abs(probs - yb_soft)
            conf_target = torch.clamp(0.65 * torch.abs(yb_soft - 0.5) * 2.0 + 0.35 * torch.abs(physics_prior - 0.5) * 2.0, 0.0, 1.0)
            conf_loss = torch.abs(conf_probs - conf_target)
            prior_loss = torch.abs(probs - physics_prior)
            weights = 0.35 + 0.65 * yb_conf
            loss = torch.mean((0.55 * cls_loss + 0.20 * reg_loss + 0.15 * conf_loss + 0.10 * prior_loss) * weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        val_probs = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, yb_soft, yb_conf in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                yb_soft = yb_soft.to(DEVICE)
                yb_conf = yb_conf.to(DEVICE)
                logits, conf_logits = model(xb)
                probs = torch.sigmoid(logits)
                conf_probs = torch.sigmoid(conf_logits)
                physics_frames = torch.stack([build_drought_physics_features(xb[:, t]) for t in range(xb.shape[1])], dim=1)
                _, physics_prior = build_sequence_physics_vector(physics_frames)
                cls_loss = _focal_loss(logits, yb)
                reg_loss = torch.abs(probs - yb_soft)
                conf_target = torch.clamp(0.65 * torch.abs(yb_soft - 0.5) * 2.0 + 0.35 * torch.abs(physics_prior - 0.5) * 2.0, 0.0, 1.0)
                conf_loss = torch.abs(conf_probs - conf_target)
                prior_loss = torch.abs(probs - physics_prior)
                weights = 0.35 + 0.65 * yb_conf
                loss = torch.mean((0.55 * cls_loss + 0.20 * reg_loss + 0.15 * conf_loss + 0.10 * prior_loss) * weights)
                val_loss += loss.item() * xb.size(0)
                val_probs.extend(probs.cpu().numpy().tolist())
                val_labels.extend(yb.cpu().numpy().tolist())

        train_loss /= max(len(train_loader.dataset), 1)
        val_loss /= max(len(val_loader.dataset), 1)
        val_auc = compute_auc(val_probs, val_labels)

        epoch_best_threshold, epoch_best_metrics = find_best_threshold(val_probs, val_labels, metric="f1")
        epoch_best_f1 = epoch_best_metrics["f1"] if epoch_best_metrics is not None else -1.0
        if epoch_best_metrics is not None:
            epoch_best_metrics["auc"] = val_auc
            epoch_best_metrics["threshold"] = epoch_best_threshold

        print(
            f"[drought] epoch {epoch + 1}/{DROUGHT_EPOCHS} "
            f"train={train_loss:.5f} val={val_loss:.5f} f1={epoch_best_f1:.4f} "
            f"auc={val_auc:.4f} thr={epoch_best_threshold:.2f}"
        )

        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            best_threshold = epoch_best_threshold
            best_metrics = {
                **epoch_best_metrics,
                "physics_baseline": physics_metrics,
                "heldout_regions": heldout_regions,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "physics_temporal_v1",
                    "dataset": bundle.dataset_path.name,
                    "best_threshold": best_threshold,
                    "validation_metrics": best_metrics,
                    "region_holdout_split": REGION_HOLDOUT_SPLIT,
                    "heldout_regions": heldout_regions,
                },
                DROUGHT_SAVE,
            )

    return reconstructed, best_metrics, best_threshold


def build_region_report(bundle, probabilities):
    records = []
    for region in sorted(set(bundle.region_name.tolist())):
        mask = bundle.region_name == region
        idx = np.where(mask)[0]
        records.append(
            {
                "region": region,
                "samples": int(len(idx)),
                "mean_probability": float(np.mean(probabilities[idx])),
                "mean_hard_label": float(torch.mean(bundle.y_hard[idx]).item()),
                "mean_soft_label": float(torch.mean(bundle.y_soft[idx]).item()),
                "mean_lat": float(np.nanmean(bundle.lat[idx])) if np.isfinite(bundle.lat[idx]).any() else float("nan"),
                "mean_lon": float(np.nanmean(bundle.lon[idx])) if np.isfinite(bundle.lon[idx]).any() else float("nan"),
            }
        )
    return sorted(records, key=lambda item: item["mean_probability"])


def main():
    preferred_dataset = os.environ.get("REAL_DATASET_FILE")
    dataset_path = choose_dataset_path(Path(preferred_dataset) if preferred_dataset else None)
    print("=" * 80)
    print("REAL-DATA DROUGHT PRODUCTION TRAINER")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print("Note: perfect drought prediction is not scientifically realistic; this script reports measured validation performance only.")
    print(f"Dataset: {dataset_path.name}")

    bundle = load_real_dataset(dataset_path)
    print(f"Samples: {len(bundle.y_hard)}")
    print(f"Positive rate: {bundle.y_hard.mean().item():.4f}")
    print(f"Soft severity mean: {bundle.y_soft.mean().item():.4f}")

    spectral_model = fine_tune_spectral_model(bundle) if SPECTRAL_EPOCHS > 0 else load_spectral_model(DEFAULT_SPECTRAL_MODEL, device=DEVICE)
    reconstructed, val_metrics, best_threshold = train_drought_model(bundle, spectral_model)

    final_model = PhysicsDroughtWrapper().to(DEVICE)
    state = torch.load(DROUGHT_SAVE, map_location=DEVICE)
    final_model.load_state_dict(state["model_state_dict"])
    final_model.eval()

    loader = DataLoader(TensorDataset(reconstructed), batch_size=BATCH_SIZE, shuffle=False)
    probabilities = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits, _ = final_model(xb)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    region_report = build_region_report(bundle, np.asarray(probabilities))
    results = {
        "dataset": bundle.dataset_path.name,
        "device": str(DEVICE),
        "spectral_model_input": DEFAULT_SPECTRAL_MODEL.name,
        "spectral_model_output": SPECTRAL_SAVE.name,
        "drought_model_output": DROUGHT_SAVE.name,
        "spectral_epochs": SPECTRAL_EPOCHS,
        "drought_epochs": DROUGHT_EPOCHS,
        "region_aware_split": REGION_AWARE_SPLIT,
        "region_holdout_split": REGION_HOLDOUT_SPLIT,
        "fast_spectral_inference": FAST_SPECTRAL_INFERENCE,
        "best_threshold": best_threshold,
        "validation_metrics": val_metrics,
        "region_report": region_report,
        "note": "Labels in this cache are mostly proxy_only because TerraClimate targets were unavailable during collection.",
    }
    with open(RESULTS_SAVE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Saved spectral model: {SPECTRAL_SAVE.name}")
    print(f"Saved drought model:  {DROUGHT_SAVE.name}")
    print(f"Saved results:        {RESULTS_SAVE.name}")


if __name__ == "__main__":
    main()
