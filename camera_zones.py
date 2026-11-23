"""Camera-specific zone storage for Digital Eye.

Camera 1 keeps using the existing zones.json file for backward compatibility.
Cameras 2-4 use separate JSON files in the project directory.

This module only manages persistence and normalization. It does not change the
zone editor or detection pipeline by itself.
"""

import json
import os
import tempfile
from pathlib import Path

from settings import ZONES_FILE


def _camera_number(camera_id):
    """Return a validated camera number (1-4) from an int or common label."""
    if isinstance(camera_id, bool):
        raise ValueError("Camera number must be between 1 and 4")

    if isinstance(camera_id, int):
        number = camera_id
    else:
        text = str(camera_id).strip().lower()
        if text.startswith("camera"):
            text = text[6:].strip(" _-#")
        try:
            number = int(text)
        except (TypeError, ValueError):
            raise ValueError("Camera must be specified as 1, 2, 3, or 4")

    if number not in (1, 2, 3, 4):
        raise ValueError("Camera number must be between 1 and 4")
    return number


def zone_file(camera_id):
    """Get the zone JSON path for a camera without creating the file."""
    number = _camera_number(camera_id)
    base_path = Path(ZONES_FILE)
    if number == 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_camera_{number}{base_path.suffix}")


def _normalize_zones(zones):
    if not isinstance(zones, list):
        return []

    normalized = []
    for index, zone in enumerate(zones, start=1):
        if not isinstance(zone, dict):
            continue

        try:
            x1 = float(zone.get("x1", 0))
            y1 = float(zone.get("y1", 0))
            x2 = float(zone.get("x2", 0))
            y2 = float(zone.get("y2", 0))
        except (TypeError, ValueError):
            continue

        # Coordinates are stored as normalized fractions, not pixels.
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        zone_type = str(zone.get("type", "Restricted")).strip().title()
        if zone_type not in ("Restricted", "Warning"):
            zone_type = "Restricted"

        try:
            timer = max(0.0, float(zone.get("timer", 0)))
        except (TypeError, ValueError):
            timer = 0.0

        normalized.append({
            "id": zone.get("id", index),
            "name": str(zone.get("name", f"Zone {index}")).strip() or f"Zone {index}",
            "type": zone_type,
            "timer": timer,
            "enabled": bool(zone.get("enabled", True)),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return normalized


def load_camera_zones(camera_id):
    """Load one camera's zones. Missing or invalid files return an empty list."""
    path = zone_file(camera_id)
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    # Accept both the historical list format and a future {"zones": [...]} form.
    if isinstance(data, dict):
        data = data.get("zones", [])
    return _normalize_zones(data)


def save_camera_zones(camera_id, zones):
    """Atomically save zones for one camera and return the saved file path."""
    path = zone_file(camera_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_zones(zones)

    fd, temp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return str(path)
