import torch
import torch.nn as nn


def build_drought_physics_features(spec):
    # spec: (B, 6, H, W)
    b = spec[:, 0]
    g = spec[:, 1]
    r = spec[:, 2]
    n = spec[:, 3]
    s1 = spec[:, 4]
    s2 = spec[:, 5]

    eps = 1e-6
    ndvi = (n - r) / (n + r + eps)
    ndmi = (n - s1) / (n + s1 + eps)
    ndwi = (g - n) / (g + n + eps)
    nbr = (n - s2) / (n + s2 + eps)
    savi = 1.5 * (n - r) / (n + r + 0.5 + eps)
    evi = 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0 + eps)
    msi = s1 / (n + eps)
    gci = (n / (g + eps)) - 1.0
    moisture = n - s1
    chlorophyll = g - r
    vegetation_water = (n - s2) / (n + s2 + eps)
    dry_soil = s1 - n

    return torch.stack(
        [
            b,
            g,
            r,
            n,
            s1,
            s2,
            ndvi,
            ndmi,
            ndwi,
            nbr,
            savi,
            evi,
            msi,
            gci,
            moisture,
            chlorophyll,
            vegetation_water,
            dry_soil,
        ],
        dim=1,
    )


def _normalize_component(value, good, bad):
    if bad > good:
        scaled = (value - good) / (bad - good)
    else:
        scaled = (good - value) / (good - bad)
    return torch.clamp(scaled, 0.0, 1.0)


def build_sequence_physics_vector(sequence_features):
    # sequence_features: (B, T, C, H, W), built from build_drought_physics_features
    means = sequence_features.mean(dim=(-1, -2))
    last = means[:, -1]
    first = means[:, 0]
    seq_mean = means.mean(dim=1)
    seq_span = means.max(dim=1).values - means.min(dim=1).values
    drop = first - last

    ndvi_idx = 6
    ndmi_idx = 7
    ndwi_idx = 8
    nbr_idx = 9
    savi_idx = 10
    evi_idx = 11
    msi_idx = 12
    moisture_idx = 14
    veg_water_idx = 16
    dry_soil_idx = 17

    low_vegetation = _normalize_component(last[:, ndvi_idx], good=0.58, bad=0.12)
    low_moisture = _normalize_component(last[:, ndmi_idx], good=0.20, bad=-0.12)
    low_surface_water = _normalize_component(last[:, ndwi_idx], good=0.10, bad=-0.18)
    burn_dryness = _normalize_component(last[:, nbr_idx], good=0.48, bad=0.06)
    canopy_loss = _normalize_component(last[:, savi_idx], good=0.46, bad=0.08)
    moisture_stress = _normalize_component(last[:, msi_idx], good=0.55, bad=1.35)
    dry_soil = _normalize_component(last[:, dry_soil_idx], good=-0.18, bad=0.18)
    vegetation_drop = _normalize_component(drop[:, ndvi_idx], good=0.00, bad=0.16)
    moisture_drop = _normalize_component(drop[:, ndmi_idx], good=0.00, bad=0.12)
    water_drop = _normalize_component(drop[:, ndwi_idx], good=0.00, bad=0.10)
    nbr_drop = _normalize_component(drop[:, nbr_idx], good=0.00, bad=0.12)
    savi_drop = _normalize_component(drop[:, savi_idx], good=0.00, bad=0.16)

    physics_prior = torch.clamp(
        0.20 * low_vegetation +
        0.17 * low_moisture +
        0.07 * low_surface_water +
        0.07 * burn_dryness +
        0.08 * canopy_loss +
        0.08 * moisture_stress +
        0.05 * dry_soil +
        0.12 * vegetation_drop +
        0.08 * moisture_drop +
        0.03 * water_drop +
        0.03 * nbr_drop +
        0.02 * savi_drop,
        0.0,
        1.0,
    )

    vector = torch.cat(
        [
            last[:, [ndvi_idx, ndmi_idx, ndwi_idx, nbr_idx, savi_idx, evi_idx, msi_idx, moisture_idx, veg_water_idx, dry_soil_idx]],
            drop[:, [ndvi_idx, ndmi_idx, ndwi_idx, nbr_idx, savi_idx, evi_idx, msi_idx, moisture_idx, veg_water_idx, dry_soil_idx]],
            seq_mean[:, [ndvi_idx, ndmi_idx, ndwi_idx, nbr_idx, savi_idx, msi_idx]],
            seq_span[:, [ndvi_idx, ndmi_idx, ndwi_idx, nbr_idx, savi_idx, msi_idx]],
            physics_prior.unsqueeze(1),
        ],
        dim=1,
    )
    return vector, physics_prior


class PhysicsTemporalDroughtModel(nn.Module):
    def __init__(self, in_channels, physics_dim, hidden_dim=192, dropout=0.30):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 72, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(72, 112, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(112, 144, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(144, 192, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        frame_dim = 192 + in_channels * 2
        self.frame_norm = nn.LayerNorm(frame_dim)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(frame_dim, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=frame_dim,
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
        self.physics_head = nn.Sequential(
            nn.LayerNorm(physics_dim),
            nn.Linear(physics_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        fused_dim = hidden_dim * 8 + hidden_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, physics_vector):
        bsz, steps, channels, height, width = x.shape
        flat = x.view(bsz * steps, channels, height, width)

        spatial = self.spatial_encoder(flat).flatten(1)
        mean_stats = flat.mean(dim=(2, 3))
        std_stats = flat.std(dim=(2, 3), unbiased=False)
        frame_features = torch.cat([spatial, mean_stats, std_stats], dim=1)
        frame_features = self.frame_norm(frame_features)
        frame_features = frame_features.view(bsz, steps, -1)

        conv_branch = self.temporal_conv(frame_features.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.gru(frame_features)
        attn_scores = self.attn(seq).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_context = torch.sum(seq * attn_weights, dim=1)
        last = seq[:, -1]
        conv_pool = conv_branch.mean(dim=1)
        conv_last = conv_branch[:, -1]
        physics_embed = self.physics_head(physics_vector)

        fused = torch.cat([attn_context, last, conv_pool, conv_last, physics_embed], dim=1)
        return self.head(fused).squeeze(-1)


class TemporalDrought(nn.Module):
    """Legacy model kept for backward compatibility."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(64 * 16 * 16, 128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        bsz, steps, channels, height, width = x.shape
        flat = x.view(bsz * steps, channels, height, width)
        feat = self.encoder(flat)
        feat = feat.view(bsz, steps, -1)
        _, (hn, _) = self.lstm(feat)
        return torch.sigmoid(self.fc(hn[-1])).squeeze()


def build_features(spec):
    return torch.stack(
        [
            spec[:, 0],
            spec[:, 1],
            spec[:, 2],
            spec[:, 3],
            spec[:, 4],
            spec[:, 5],
            (spec[:, 3] - spec[:, 2]) / (spec[:, 3] + spec[:, 2] + 1e-3),
            (spec[:, 1] - spec[:, 3]) / (spec[:, 1] + spec[:, 3] + 1e-3),
            (spec[:, 3] - spec[:, 4]) / (spec[:, 3] + spec[:, 4] + 1e-3),
        ],
        dim=1,
    )
