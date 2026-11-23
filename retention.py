"""Safe age-based cleanup for Digital Eye recordings and event snapshots."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

from settings import INCIDENTS_DIR, RECORDINGS_DIR, RETENTION_DAYS

MEDIA_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}


def cleanup_expired_media(
    retention_days: int = RETENTION_DAYS,
    protected_paths: Iterable[str] = (),
    now: float | None = None,
) -> list[str]:
    """Delete old media under the recordings and incidents folders.

    Returns paths successfully deleted. Active/in-progress files can be passed
    through ``protected_paths`` and are never removed.
    """
    if not isinstance(retention_days, int) or retention_days < 1:
        raise ValueError("retention_days must be a positive integer")

    cutoff = (time.time() if now is None else now) - retention_days * 86400
    protected = {os.path.abspath(path) for path in protected_paths}
    deleted: list[str] = []

    for root in (RECORDINGS_DIR, INCIDENTS_DIR):
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            absolute = os.path.abspath(str(path))
            if absolute in protected:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted.append(str(path))
            except OSError as error:
                print(f"Retention cleanup could not remove {path}: {error}")

    return deleted
