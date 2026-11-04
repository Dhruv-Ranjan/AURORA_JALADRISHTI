from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_FILE = BASE_DIR / "dataset_seq_forecast_v1.npz"
DEFAULT_SPECTRAL_MODEL_FILE = BASE_DIR / "spectral_v8_spatial.pt"
DEFAULT_DROUGHT_MODEL_FILE = BASE_DIR / "drought_v8_temporal.pt"
DEFAULT_DROUGHT_FEATURE_CACHE = BASE_DIR / "drought_features_v8.pt"
DEFAULT_DROUGHT_V9_MODEL_FILE = BASE_DIR / "drought_v9_delta_attention.pt"
DEFAULT_DROUGHT_V9_FEATURE_CACHE = BASE_DIR / "drought_features_v9.pt"
DEFAULT_DROUGHT_V9_META_FILE = BASE_DIR / "drought_v9_metadata.json"
DEFAULT_SPECTRAL_V10_MODEL_FILE = BASE_DIR / "spectral_v10_hires.pt"
DEFAULT_DROUGHT_V10_MODEL_FILE = BASE_DIR / "drought_v10_fusion.pt"
DEFAULT_DROUGHT_V10_FEATURE_CACHE = BASE_DIR / "drought_features_v10.pt"
DEFAULT_DROUGHT_V10_META_FILE = BASE_DIR / "drought_v10_metadata.json"
DEFAULT_CACHE_FILE_V2 = BASE_DIR / "dataset_seq_forecast_v2.npz"
DEFAULT_CACHE_FILE_V3 = BASE_DIR / "dataset_seq_forecast_v3.npz"


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU()
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.se(x)
        return self.act(x + residual)


class SpectralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.e1 = ConvBlock(3, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.mid = ConvBlock(128, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = ConvBlock(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = ConvBlock(128 + 64, 64)

        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d0 = ConvBlock(64 + 32, 32)

        self.head = nn.Conv2d(32, 6, 1)

    def forward(self, x):
        rgb_as_spec = torch.stack([x[:, 2], x[:, 1], x[:, 0]], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))

        d2 = self.up2(m)
        d2 = self.d2(torch.cat([d2, e3], dim=1))

        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, e2], dim=1))

        d0 = self.up0(d1)
        d0 = self.d0(torch.cat([d0, e1], dim=1))

        out = self.head(d0)

        visible_delta = 0.1 * torch.tanh(out[:, :3])
        visible = torch.clamp(rgb_as_spec + visible_delta, 0.0, 1.0)
        hidden = torch.sigmoid(out[:, 3:])

        return torch.cat([visible, hidden], dim=1)


class SpectralNetV10(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2)

        self.e1 = ResidualConvBlock(3, 48)
        self.e2 = ResidualConvBlock(48, 96)
        self.e3 = ResidualConvBlock(96, 160)
        self.e4 = ResidualConvBlock(160, 224)
        self.mid = ResidualConvBlock(224, 320)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = ResidualConvBlock(320 + 224, 224)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = ResidualConvBlock(224 + 160, 160)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1 = ResidualConvBlock(160 + 96, 96)
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d0 = ResidualConvBlock(96 + 48, 64)

        self.low_head = nn.Conv2d(224, 6, 1)
        self.mid_head = nn.Conv2d(160, 6, 1)
        self.head = nn.Conv2d(64, 6, 1)

    def _format_output(self, out, rgb_input):
        rgb_as_spec = torch.stack([rgb_input[:, 2], rgb_input[:, 1], rgb_input[:, 0]], dim=1)
        visible_delta = 0.08 * torch.tanh(out[:, :3])
        visible = torch.clamp(rgb_as_spec + visible_delta, 0.0, 1.0)
        hidden = torch.sigmoid(out[:, 3:])
        return torch.cat([visible, hidden], dim=1)

    def forward(self, x, return_aux=False):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        mid = self.mid(self.pool(e4))

        d3 = self.up3(mid)
        d3 = self.d3(torch.cat([d3, e4], dim=1))
        d2 = self.up2(d3)
        d2 = self.d2(torch.cat([d2, e3], dim=1))
        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, e2], dim=1))
        d0 = self.up0(d1)
        d0 = self.d0(torch.cat([d0, e1], dim=1))

        out = self._format_output(self.head(d0), x)

        if not return_aux:
            return out

        aux_low = self._format_output(
            nn.functional.interpolate(self.low_head(d3), size=x.shape[-2:], mode="bilinear", align_corners=False),
            x,
        )
        aux_mid = self._format_output(
            nn.functional.interpolate(self.mid_head(d2), size=x.shape[-2:], mode="bilinear", align_corners=False),
            x,
        )
        return out, aux_mid, aux_low


def build_spectral_features(spec):
    b = spec[:, 0]
    g = spec[:, 1]
    r = spec[:, 2]
    n = spec[:, 3]
    s1 = spec[:, 4]
    s2 = spec[:, 5]

    ndvi = (n - r) / (n + r + 1e-3) #Normalised Difference Vegetation Index
    ndmi = (n - s1) / (n + s1 + 1e-3) # Normalised Difference Moisture Index
    ndwi = (g - n) / (g + n + 1e-3) # Normalised Difference Water Index
    savi = 1.5 * (n - r) / (n + r + 0.5) # Soil Adjusted Vegetation Index
    evi = 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0 + 1e-3) # Enhanced Vegetation Index
    nbr = (n - s2) / (n + s2 + 1e-3) # Normalised Burn Ratio 
    moisture = n - s1
    chlorophyll = g - r

    return torch.stack(
        [b, g, r, n, s1, s2, ndvi, ndmi, ndwi, savi, evi, nbr, moisture, chlorophyll],
        dim=1,
    )


def append_temporal_delta_features(sequence_features):
    deltas = torch.zeros_like(sequence_features)
    deltas[:, 1:] = sequence_features[:, 1:] - sequence_features[:, :-1]
    return torch.cat([sequence_features, deltas], dim=2)


def append_temporal_context_features(sequence_features):
    deltas = torch.zeros_like(sequence_features)
    deltas[:, 1:] = sequence_features[:, 1:] - sequence_features[:, :-1]

    seq_mean = sequence_features.mean(dim=1, keepdim=True).expand(-1, sequence_features.shape[1], -1, -1, -1)
    seq_span = (sequence_features.max(dim=1, keepdim=True).values - sequence_features.min(dim=1, keepdim=True).values)
    seq_span = seq_span.expand(-1, sequence_features.shape[1], -1, -1, -1)

    return torch.cat([sequence_features, deltas, seq_mean, seq_span], dim=2)


class TemporalAttentionDroughtModel(nn.Module):
    def __init__(self, in_channels=14, hidden_dim=160, lstm_layers=2, dropout=0.25):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feature_norm = nn.LayerNorm(96 + in_channels * 2)
        self.lstm = nn.LSTM(
            input_size=96 + in_channels * 2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        bsz, steps, channels, height, width = x.shape

        flat = x.view(bsz * steps, channels, height, width)
        spatial = self.frame_encoder(flat).flatten(1)
        mean_stats = flat.mean(dim=(2, 3))
        std_stats = flat.std(dim=(2, 3), unbiased=False)
        frame_features = torch.cat([spatial, mean_stats, std_stats], dim=1)
        frame_features = self.feature_norm(frame_features)
        frame_features = frame_features.view(bsz, steps, -1)

        seq, _ = self.lstm(frame_features)
        attn_scores = self.attn(seq).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(seq * attn_weights, dim=1)
        last = seq[:, -1]
        return self.head(torch.cat([context, last], dim=1)).squeeze(-1)


class TemporalFusionDroughtModelV10(nn.Module):
    def __init__(self, in_channels=56, hidden_dim=224, dropout=0.30):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            ResidualConvBlock(in_channels, 64),
            nn.AvgPool2d(2),
            ResidualConvBlock(64, 96),
            nn.AvgPool2d(2),
            ResidualConvBlock(96, 128),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feature_norm = nn.LayerNorm(128 + in_channels * 2)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128 + in_channels * 2, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=128 + in_channels * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x):
        bsz, steps, channels, height, width = x.shape
        flat = x.view(bsz * steps, channels, height, width)

        spatial = self.spatial_encoder(flat).flatten(1)
        mean_stats = flat.mean(dim=(2, 3))
        std_stats = flat.std(dim=(2, 3), unbiased=False)
        frame_features = torch.cat([spatial, mean_stats, std_stats], dim=1)
        frame_features = self.feature_norm(frame_features)
        frame_features = frame_features.view(bsz, steps, -1)

        conv_branch = self.temporal_conv(frame_features.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.gru(frame_features)
        attn_scores = self.attn(seq).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_context = torch.sum(seq * attn_weights, dim=1)
        conv_pool = conv_branch.mean(dim=1)
        last = seq[:, -1]
        fused = torch.cat([attn_context, last, conv_pool], dim=1)
        return self.head(fused).squeeze(-1)


def load_sequence_dataset(cache_file=DEFAULT_CACHE_FILE):
    data = np.load(cache_file, allow_pickle=True)
    rgb_seq = torch.tensor(data["rgb"], dtype=torch.float32)
    spec_seq = torch.tensor(data["spec"], dtype=torch.float32)
    labels = torch.tensor(data["y"], dtype=torch.float32)
    return rgb_seq, spec_seq, labels


def load_sequence_dataset_extended(cache_file=DEFAULT_CACHE_FILE):
    data = np.load(cache_file, allow_pickle=True)
    payload = {
        "rgb": torch.tensor(data["rgb"], dtype=torch.float32),
        "spec": torch.tensor(data["spec"], dtype=torch.float32),
        "y": torch.tensor(data["y"], dtype=torch.float32),
    }
    if "y_soft" in data.files:
        payload["y_soft"] = torch.tensor(data["y_soft"], dtype=torch.float32)
    if "y_confidence" in data.files:
        payload["y_confidence"] = torch.tensor(data["y_confidence"], dtype=torch.float32)
    if "y_proxy" in data.files:
        payload["y_proxy"] = torch.tensor(data["y_proxy"], dtype=torch.float32)
    if "y_proxy_confidence" in data.files:
        payload["y_proxy_confidence"] = torch.tensor(data["y_proxy_confidence"], dtype=torch.float32)
    if "y_real" in data.files:
        payload["y_real"] = torch.tensor(data["y_real"], dtype=torch.float32)
    if "y_real_confidence" in data.files:
        payload["y_real_confidence"] = torch.tensor(data["y_real_confidence"], dtype=torch.float32)
    if "region_id" in data.files:
        payload["region_id"] = torch.tensor(data["region_id"], dtype=torch.long)
    for key in [
        "region_name",
        "future_date",
        "label_source",
        "vegetation_fraction",
        "water_fraction",
        "non_vegetated_fraction",
        "usable_fraction",
        "lon",
        "lat",
    ]:
        if key in data.files:
            payload[key] = data[key]
    return payload


def stratified_split_indices(labels, val_fraction=0.2, seed=42):
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for label_value in np.unique(labels_np):
        idx = np.where(labels_np == label_value)[0]
        rng.shuffle(idx)
        val_count = max(1, int(round(len(idx) * val_fraction)))
        val_indices.extend(idx[:val_count].tolist())
        train_indices.extend(idx[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def binary_classification_metrics(probs, labels):
    return binary_classification_metrics_with_threshold(probs, labels, threshold=0.5)


def binary_classification_metrics_with_threshold(probs, labels, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    preds = (probs >= threshold).astype(np.float32)

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(labels), 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(probs, labels, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in np.linspace(0.1, 0.9, 81):
        metrics = binary_classification_metrics_with_threshold(probs, labels, threshold=float(threshold))
        score = metrics.get(metric, -1.0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def compute_auc(probs, labels):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    labels = labels.astype(np.int64, copy=False)

    if len(np.unique(labels)) < 2:
        return float("nan")

    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    sorted_labels = labels[order]

    n = len(sorted_probs)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_probs[j] == sorted_probs[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j

    positive_mask = sorted_labels == 1
    positive_count = int(positive_mask.sum())
    negative_count = int(len(sorted_labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    rank_sum = float(ranks[positive_mask].sum())
    auc = (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(np.clip(auc, 0.0, 1.0))


def predict_spectral_with_tta(model, xb):
    variants = [
        xb,
        torch.flip(xb, dims=[-1]),
        torch.flip(xb, dims=[-2]),
        torch.flip(xb, dims=[-1, -2]),
    ]

    preds = []
    with torch.no_grad():
        for idx, variant in enumerate(variants):
            pred = model(variant)
            if idx == 1:
                pred = torch.flip(pred, dims=[-1])
            elif idx == 2:
                pred = torch.flip(pred, dims=[-2])
            elif idx == 3:
                pred = torch.flip(pred, dims=[-1, -2])
            preds.append(pred)

    return torch.stack(preds, dim=0).mean(dim=0)


def blend_spectral_predictions(primary_pred, secondary_pred, alpha=0.7):
    alpha = float(alpha)
    blended = alpha * primary_pred + (1.0 - alpha) * secondary_pred
    visible = torch.clamp(blended[:, :3], 0.0, 1.0)
    hidden = torch.clamp(blended[:, 3:], 0.0, 1.0)
    return torch.cat([visible, hidden], dim=1)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
