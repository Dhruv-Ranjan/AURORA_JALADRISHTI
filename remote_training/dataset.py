from __future__ import annotations

import logging
import random
import json
import hashlib
import time
from dataclasses import dataclass
from typing import Iterator
from io import BytesIO

import numpy as np

from .accounting import SampleAccounting
from .cache import BoundedFileCache
from .config import RemoteTrainingConfig

LOGGER = logging.getLogger(__name__)
REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")
INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}


@dataclass(frozen=True)
class RemoteSample:
    """A single manifest row; imagery is fetched only during iteration."""

    item_id: str
    region: str
    acquired: str
    longitude: float
    latitude: float
    asset_hrefs: dict[str, str]


class RemoteWindowReader:
    """Read only a patch window from signed cloud-optimized raster assets."""

    def __init__(self, config: RemoteTrainingConfig):
        self.config = config
        self.cache = BoundedFileCache(
            config.cache_directory, config.cache_limit_bytes, config.cache_enabled
        )

    def _open(self, href: str):
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("rasterio is required for remote window reads") from exc
        return rasterio.open(href)

    def _read_band(self, href: str, longitude: float, latitude: float, size: int) -> np.ndarray:
        import rasterio
        from rasterio.windows import Window
        from rasterio.warp import transform

        cache_key = f"{href}|{longitude:.7f}|{latitude:.7f}|{size}"
        cached = self.cache.get(cache_key, ".npy")
        if cached is not None:
            return np.load(BytesIO(cached), allow_pickle=False)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with self._open(href) as source:
                    xs, ys = transform("EPSG:4326", source.crs, [longitude], [latitude])
                    row, col = source.index(xs[0], ys[0])
                    half = size // 2
                    window = Window(col - half, row - half, size, size)
                    if (
                        window.col_off < 0
                        or window.row_off < 0
                        or window.col_off + size > source.width
                        or window.row_off + size > source.height
                    ):
                        raise ValueError("window outside raster")
                    data = source.read(
                        1,
                        window=window,
                        out_shape=(size, size),
                        boundless=False,
                        masked=True,
                    )
                    if np.ma.isMaskedArray(data):
                        result = data.filled(np.nan).astype(np.float32)
                    else:
                        result = np.asarray(data, dtype=np.float32)
                break
            except (OSError, TimeoutError, ConnectionError, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    if isinstance(exc, ValueError):
                        raise
                    raise RuntimeError("network") from exc
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        else:
            raise RuntimeError("network") from last_error
        buffer = BytesIO()
        np.save(buffer, result, allow_pickle=False)
        self.cache.put(cache_key, buffer.getvalue(), ".npy")
        return result

    def read(self, sample: RemoteSample) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.patch_size
        arrays: dict[str, np.ndarray] = {}
        try:
            for band in REQUIRED_BANDS:
                if band not in sample.asset_hrefs:
                    raise KeyError(band)
                band_size = size if band in {"B02", "B03", "B04", "B08"} else size // 2
                arrays[band] = self._read_band(
                    sample.asset_hrefs[band],
                    sample.longitude,
                    sample.latitude,
                    band_size,
                )
                if band_size != size:
                    arrays[band] = np.repeat(np.repeat(arrays[band], 2, axis=0), 2, axis=1)
        except KeyError as exc:
            raise RuntimeError("missing_bands") from exc
        except (OSError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError("network") from exc
        except ValueError as exc:
            raise RuntimeError("quality") from exc

        scl = np.where(np.isfinite(arrays["SCL"]), np.rint(arrays["SCL"]), -1).astype(np.int16)
        valid = np.isfinite(arrays["B02"])
        for band in REQUIRED_BANDS[:-1]:
            valid &= np.isfinite(arrays[band])
        valid &= ~np.isin(scl, list(INVALID_SCL))
        for band in REQUIRED_BANDS[:-1]:
            valid &= (arrays[band] >= 0) & (arrays[band] <= 10000)
        if float(valid.mean()) < 0.80:
            raise RuntimeError("invalid_pixels")

        rgb = np.stack([arrays["B04"], arrays["B03"], arrays["B02"]], axis=0) / 10000.0
        target = np.stack([arrays[band] for band in REQUIRED_BANDS[:-1]], axis=0) / 10000.0
        rgb[:, ~valid] = 0
        target[:, ~valid] = 0
        return rgb.astype(np.float32), target.astype(np.float32)


class RemoteSequenceDataset:
    """Yield valid samples until the requested count is reached."""

    def __init__(self, config: RemoteTrainingConfig, samples: list[RemoteSample]):
        config.validate()
        self.config = config
        self.samples = samples
        self.reader = RemoteWindowReader(config)
        self.accounting = SampleAccounting()
        self._rng = random.Random(config.seed)
        self._seen: set[tuple[str, float, float]] = set()
        self._manifest = config.manifest_path
        self._manifest.parent.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[dict[str, object]]:
        order = list(self.samples)
        self._rng.shuffle(order)
        target = self.config.target_valid_samples
        for sample in order:
            if self.accounting.valid_samples >= target:
                break
            self.accounting.replacement_candidates_requested += 1
            key = (sample.item_id, sample.longitude, sample.latitude)
            if key in self._seen:
                self.accounting.record("duplicate")
                continue
            self._seen.add(key)
            try:
                rgb, target_spec = self.reader.read(sample)
            except RuntimeError as exc:
                self.accounting.record(str(exc))
                continue
            except Exception:
                LOGGER.exception("Unexpected failure reading %s", sample.item_id)
                self.accounting.record("other")
                continue
            self.accounting.record(None)
            with self._manifest.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "item_id": sample.item_id,
                            "region": sample.region,
                            "acquired": sample.acquired,
                            "longitude": sample.longitude,
                            "latitude": sample.latitude,
                            "split": "train",
                            "bands": list(REQUIRED_BANDS),
                            "patch_size": self.config.patch_size,
                        }
                    )
                    + "\n"
                )
            yield {
                "rgb": rgb,
                "target": target_spec,
                "metadata": {
                    "item_id": sample.item_id,
                    "region": sample.region,
                    "acquired": sample.acquired,
                    "latitude": sample.latitude,
                    "longitude": sample.longitude,
                },
            }

    @property
    def exhausted(self) -> bool:
        return self.accounting.valid_samples < self.config.target_valid_samples


def discover_samples(config: RemoteTrainingConfig) -> list[RemoteSample]:
    """Discover metadata only; no imagery is downloaded during discovery."""

    try:
        import planetary_computer
        import pystac_client
    except ImportError as exc:
        raise RuntimeError("pystac-client and planetary-computer are required") from exc

    catalog = pystac_client.Client.open(config.stac_url)
    regions = config.regions or ({"name": "global", "bbox": [-180, -60, 180, 60]},)
    discovered: list[RemoteSample] = []
    for region in regions:
        search = catalog.search(
            collections=[config.collection],
            bbox=region["bbox"],
            datetime=config.date_range,
            query={"eo:cloud_cover": {"lt": config.cloud_cover_max}},
        )
        for item in search.items():
            signed = planetary_computer.sign(item)
            if any(band not in signed.assets for band in REQUIRED_BANDS):
                continue
            bbox = signed.bbox or region["bbox"]
            if len(bbox) != 4:
                continue
            for patch_index in range(config.patches_per_item):
                digest = hashlib.sha256(f"{signed.id}:{patch_index}".encode("utf-8")).digest()
                u_lon = int.from_bytes(digest[:8], "big") / 2**64
                u_lat = int.from_bytes(digest[8:16], "big") / 2**64
                discovered.append(
                    RemoteSample(
                        item_id=f"{signed.id}#patch-{patch_index}",
                        region=str(region["name"]),
                        acquired=signed.datetime.isoformat() if signed.datetime else "",
                        longitude=float(bbox[0] + u_lon * (bbox[2] - bbox[0])),
                        latitude=float(bbox[1] + u_lat * (bbox[3] - bbox[1])),
                        asset_hrefs={band: signed.assets[band].href for band in REQUIRED_BANDS},
                    )
                )
                if config.max_candidates and len(discovered) >= config.max_candidates:
                    return discovered[: config.max_candidates]
    return discovered
