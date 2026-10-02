from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


class BoundedFileCache:
    """A size-limited cache with oldest-access eviction."""

    def __init__(self, directory: Path, limit_bytes: int, enabled: bool = True):
        self.directory = Path(directory)
        self.limit_bytes = max(0, int(limit_bytes))
        self.enabled = enabled and self.limit_bytes > 0
        self._peak_bytes = 0
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str, suffix: str = ".bin") -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}{suffix}"

    def get(self, key: str, suffix: str = ".bin") -> bytes | None:
        if not self.enabled:
            return None
        path = self.path_for(key, suffix)
        try:
            value = path.read_bytes()
            os.utime(path, None)
            return value
        except FileNotFoundError:
            return None

    def put(self, key: str, value: bytes, suffix: str = ".bin") -> None:
        if not self.enabled or len(value) > self.limit_bytes:
            return
        path = self.path_for(key, suffix)
        path.write_bytes(value)
        self._peak_bytes = max(self._peak_bytes, self.usage_bytes())
        self.evict()

    def usage_bytes(self) -> int:
        if not self.enabled or not self.directory.exists():
            return 0
        return sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())

    def evict(self) -> None:
        while self.usage_bytes() > self.limit_bytes:
            files = [path for path in self.directory.iterdir() if path.is_file()]
            if not files:
                return
            oldest = min(files, key=lambda path: path.stat().st_atime)
            oldest.unlink(missing_ok=True)

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes
