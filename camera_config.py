"""Persistent, validated configuration for Digital Eye camera slots.

Camera source strings (including RTSP URLs) are stored locally in JSON.
Do not share camera_config.json: RTSP URLs may contain credentials.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "camera_config.json"
MAX_CAMERAS = 4
_DEFAULTS = {
    str(number): {
        "name": f"Camera {number}",
        "source": "0" if number == 1 else "",
        "enabled": number == 1,
        "width": 640,
        "height": 480,
        "fps": 15,
        "confidence": 0.40,
    }
    for number in range(1, MAX_CAMERAS + 1)
}
_LOCK = RLock()


def _normalise_slot(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {}
    name = str(value.get("name", "")).strip()[:64]
    source = str(value.get("source", "")).strip()[:2048]
    try:
        width = int(value.get("width", 640))
        height = int(value.get("height", 480))
        fps = int(value.get("fps", 15))
        confidence = float(value.get("confidence", 0.40))
    except (TypeError, ValueError):
        width, height, fps, confidence = 640, 480, 15, 0.40
    return {
        "name": name or "Camera",
        "source": source,
        "enabled": bool(value.get("enabled", False)) and bool(source),
        "width": min(max(width, 160), 3840),
        "height": min(max(height, 120), 2160),
        "fps": min(max(fps, 1), 60),
        "confidence": min(max(confidence, 0.05), 0.99),
    }


class CameraConfigStore:
    """Thread-safe JSON store for four independent camera configurations."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else CONFIG_PATH

    def load_all(self) -> dict[str, dict]:
        with _LOCK:
            data = {}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            except FileNotFoundError:
                pass
            except (OSError, json.JSONDecodeError):
                # Keep the application usable; defaults are returned and the
                # malformed file is not overwritten automatically.
                return {key: dict(value) for key, value in _DEFAULTS.items()}
            return {
                str(number): _normalise_slot(data.get(str(number), defaults))
                for number, defaults in ((n, _DEFAULTS[str(n)]) for n in range(1, MAX_CAMERAS + 1))
            }

    def get(self, number: int) -> dict:
        self._validate_number(number)
        return self.load_all()[str(number)]

    def save(self, number: int, **changes) -> dict:
        self._validate_number(number)
        with _LOCK:
            all_configs = self.load_all()
            current = dict(all_configs[str(number)])
            current.update(changes)
            all_configs[str(number)] = _normalise_slot(current)
            self._atomic_write(all_configs)
            return dict(all_configs[str(number)])

    def _atomic_write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="camera_config_", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _validate_number(number: int) -> None:
        if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= MAX_CAMERAS:
            raise ValueError("Camera number must be an integer from 1 to 4")


def parse_camera_source(value: Any) -> int | str:
    """Convert a numeric USB camera source to int; preserve stream URLs."""
    value = str(value).strip()
    if not value:
        raise ValueError("Camera source cannot be empty")
    if value.isdecimal():
        return int(value)
    return value
